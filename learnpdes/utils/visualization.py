"""Evaluate visualization samples independently of the collocation grid."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from matplotlib.tri import Triangulation, UniformTriRefiner

from learnpdes import (
    COSINUS_SCENARIO,
    EXPONENTIAL_SCENARIO,
    LAPLACE_SCENARIO,
    POTENTIAL_FLOW_SCENARIO,
    SOLENOIDAL_FLOW_SCENARIO,
)


@dataclass
class VisualizationGrid:
    coordinates: np.ndarray
    triangulation: Triangulation | None = None
    boundary_edges: np.ndarray | None = None


def airfoil_grid(mesh_path: str | Path, subdivisions: int = 1) -> VisualizationGrid:
    """Refine the SU2 fluid cells, retaining the hole and boundary connectivity.

    Refining the existing mesh preserves its concentration of samples near the
    airfoil. Re-triangulating only the points would fill the solid interior.
    """
    lines = Path(mesh_path).read_text().splitlines()
    triangles, edges, coordinates = [], [], []
    for index, line in enumerate(lines):
        if line.startswith('NELEM='):
            count = int(line.split('=')[1])
            for element in lines[index + 1 : index + 1 + count]:
                values = [int(value) for value in element.split()]
                if values[0] == 5:
                    triangles.append(values[1:4])
                elif values[0] == 9:
                    a, b, c, d = values[1:5]
                    triangles.extend([(a, b, c), (a, c, d)])
                else:
                    raise ValueError(f'Unsupported SU2 cell type: {values[0]}')
        elif line.startswith('NPOIN='):
            count = int(line.split('=')[1].split()[0])
            coordinates = [
                [float(value) for value in point.split()[:2]]
                for point in lines[index + 1 : index + 1 + count]
            ]
        elif line.startswith('MARKER_TAG=') and line.split('=')[1].strip() == 'airfoil':
            count = int(lines[index + 1].split('=')[1])
            edges = [
                [int(value) for value in edge.split()[1:3]]
                for edge in lines[index + 2 : index + 2 + count]
            ]
    if not coordinates or not triangles or not edges:
        raise ValueError(
            'Expected SU2 fluid cells, coordinates, and an airfoil marker.'
        )
    xy = np.asarray(coordinates)
    mesh = Triangulation(xy[:, 0], xy[:, 1], np.asarray(triangles))
    refined = UniformTriRefiner(mesh).refine_triangulation(subdiv=subdivisions)
    return VisualizationGrid(
        np.column_stack((refined.x, refined.y)), refined, xy[np.asarray(edges)]
    )


def visualization_grid(
    scenario: str,
    input_space: torch.Tensor,
    resolution: int = 300,
    mesh_path: str | Path = 'meshes/mesh_airfoil_ch10sm.su2',
) -> VisualizationGrid:
    """Build plotting coordinates without changing any training samples."""
    if resolution < 2:
        raise ValueError('Visualization resolution must be at least 2.')
    bounds = input_space.detach().cpu().numpy()
    if scenario in (EXPONENTIAL_SCENARIO, COSINUS_SCENARIO):
        return VisualizationGrid(
            np.linspace(bounds.min(), bounds.max(), resolution)[:, None]
        )
    if scenario == LAPLACE_SCENARIO:
        x, y = np.meshgrid(
            np.linspace(bounds[:, 0].min(), bounds[:, 0].max(), resolution),
            np.linspace(bounds[:, 1].min(), bounds[:, 1].max(), resolution),
            indexing='ij',
        )
        return VisualizationGrid(np.column_stack((x.ravel(), y.ravel())))
    if scenario in (POTENTIAL_FLOW_SCENARIO, SOLENOIDAL_FLOW_SCENARIO):
        return airfoil_grid(mesh_path)
    raise ValueError(f'Unknown visualization scenario: {scenario}')


class ModelEvaluator:
    """Bound memory use and differentiate only when the plotted field needs it."""

    def __init__(self, model, scenario, grid, density=1.225, batch_size=8192):
        self.model = model
        self.scenario = scenario
        self.grid = grid
        self.density = float(density)
        self.batch_size = batch_size
        if batch_size < 1:
            raise ValueError('Evaluation batch size must be positive.')

    def __call__(self) -> dict:
        parameter = next(self.model.parameters())
        flow = self.scenario in (POTENTIAL_FLOW_SCENARIO, SOLENOIDAL_FLOW_SCENARIO)
        batches = []
        was_training = self.model.training
        self.model.eval()
        try:
            for start in range(0, len(self.grid.coordinates), self.batch_size):
                xy = torch.as_tensor(
                    self.grid.coordinates[start : start + self.batch_size],
                    dtype=parameter.dtype,
                    device=parameter.device,
                )
                with torch.set_grad_enabled(flow):
                    xy.requires_grad_(flow)
                    output = self.model(xy)
                    if flow:
                        derivative = torch.autograd.grad(output[:, 0].sum(), xy)[0]
                        if self.scenario == POTENTIAL_FLOW_SCENARIO:
                            u, v = derivative[:, 0], derivative[:, 1]
                            p = self.density * (u.square() + v.square()) / 2
                        else:
                            u, v = derivative[:, 1], -derivative[:, 0]
                            p = torch.zeros_like(u)
                        output = torch.stack((u, v, p), dim=1)
                batches.append(output.detach().cpu().numpy())
        finally:
            self.model.train(was_training)
        values = np.concatenate(batches)
        result = {
            'inputs': self.grid.coordinates,
            'f': tuple(values[:, i] for i in range(3)) if flow else values,
            'geometry_mask': None,
        }
        if self.scenario == SOLENOIDAL_FLOW_SCENARIO:
            result['pressure_label'] = 'Pressure p (not modeled)'
        if self.grid.triangulation is not None:
            result.update(
                triangulation=self.grid.triangulation,
                boundary_edges=self.grid.boundary_edges,
            )
        return result
