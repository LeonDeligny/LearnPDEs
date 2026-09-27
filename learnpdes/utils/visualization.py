"""Evaluate visualization samples independently of the collocation grid."""

from dataclasses import dataclass
from pathlib import Path

from learnpdes.scenarios import DEFAULT_MESH

import numpy as np
import torch

from learnpdes import (
    cosinus,
    COSINUS_SCENARIO,
    CYLINDER_SCENARIO,
    EXPONENTIAL_SCENARIO,
    LAPLACE_SCENARIO,
    KOVASZNAY_SCENARIO,
    POISEUILLE_SCENARIO,
    POTENTIAL_FLOW_SCENARIO,
    SOLENOIDAL_FLOW_SCENARIO,
)


@dataclass
class VisualizationGrid:
    coordinates: np.ndarray
    triangles: np.ndarray | None = None
    boundary_edges: np.ndarray | None = None


def airfoil_grid(mesh_path: str | Path, subdivisions: int = 1) -> VisualizationGrid:
    """Refine the SU2 fluid cells, retaining the hole and boundary connectivity.

    Refining the existing mesh preserves its concentration of samples near the
    airfoil. Re-triangulating only the points would fill the solid interior.
    """
    if subdivisions < 0:
        raise ValueError('Mesh subdivisions cannot be negative.')
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
    boundary_edges = xy[np.asarray(edges)]
    triangles = np.asarray(triangles)
    for _ in range(subdivisions):
        # Give adjacent cells the same midpoint, preserving connectivity and
        # refining only existing fluid cells rather than filling the solid hole.
        cell_edges = triangles[:, [[0, 1], [1, 2], [2, 0]]]
        unique_edges, inverse = np.unique(
            np.sort(cell_edges.reshape(-1, 2), axis=1), axis=0, return_inverse=True
        )
        midpoints = inverse.reshape(-1, 3) + len(xy)
        xy = np.vstack((xy, xy[unique_edges].mean(axis=1)))
        a, b, c = triangles.T
        ab, bc, ca = midpoints.T
        triangles = np.vstack(
            (
                np.column_stack((a, ab, ca)),
                np.column_stack((ab, b, bc)),
                np.column_stack((ca, bc, c)),
                np.column_stack((ab, bc, ca)),
            )
        )
    return VisualizationGrid(xy, triangles, boundary_edges)


def rectangular_triangles(coordinates: np.ndarray) -> np.ndarray:
    """Connect a complete rectangular grid without triangulating across holes."""
    xy = np.asarray(coordinates)
    x, y = np.unique(xy[:, 0]), np.unique(xy[:, 1])
    if (
        len(x) < 2
        or len(y) < 2
        or len(x) * len(y) != len(xy)
        or len(np.unique(xy, axis=0)) != len(xy)
    ):
        raise ValueError('Nonrectangular flow grids require explicit fluid triangles.')
    nodes = np.lexsort((xy[:, 0], xy[:, 1])).reshape(len(y), len(x))
    a, b = nodes[:-1, :-1].ravel(), nodes[:-1, 1:].ravel()
    c, d = nodes[1:, 1:].ravel(), nodes[1:, :-1].ravel()
    return np.vstack((np.column_stack((a, b, c)), np.column_stack((a, c, d))))


def visualization_grid(
    scenario: str,
    input_space: torch.Tensor,
    resolution: int = 300,
    mesh_path: str | Path = DEFAULT_MESH,
) -> VisualizationGrid:
    """Build plotting coordinates without changing any training samples."""
    if resolution < 2:
        raise ValueError('Visualization resolution must be at least 2.')
    bounds = input_space.detach().cpu().numpy()
    if scenario == CYLINDER_SCENARIO:
        from learnpdes.fluid import CylinderProblem

        problem = CylinderProblem()
        count = max(64, 4 * resolution)
        angles = np.arange(count) * (2 * np.pi / count)
        directions = np.column_stack((np.cos(angles), np.sin(angles)))
        center = np.asarray(problem.center)
        # Connect nested rings to their intersections with the channel walls.
        # The innermost polygon is the cylinder; no cell spans its interior.
        with np.errstate(divide='ignore'):
            distances = (
                np.where(directions >= 0, np.asarray([22.0, 4.1]) - center, -center)
                / directions
            )
        outer = distances.min(axis=1)
        fraction = np.linspace(0, 1, max(3, resolution)) ** 1.5
        radii = problem.radius + fraction[:, None] * (outer - problem.radius)
        coordinates = (center + radii[:, :, None] * directions).reshape(-1, 2)
        a = np.arange((len(fraction) - 1) * count)
        b = a // count * count + (a + 1) % count
        triangles = np.vstack(
            (
                np.column_stack((a, b, b + count)),
                np.column_stack((a, b + count, a + count)),
            )
        )
        edges = coordinates[
            np.column_stack((np.arange(count), (np.arange(count) + 1) % count))
        ]
        return VisualizationGrid(coordinates, triangles, edges)
    if scenario == COSINUS_SCENARIO:
        return VisualizationGrid(
            np.linspace(*cosinus.EVALUATION_BOUNDS, resolution)[:, None]
        )
    if scenario == EXPONENTIAL_SCENARIO:
        return VisualizationGrid(
            np.linspace(bounds.min(), bounds.max(), resolution)[:, None]
        )
    if scenario in (LAPLACE_SCENARIO, POISEUILLE_SCENARIO, KOVASZNAY_SCENARIO):
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
        differentiate = self.scenario in (
            POTENTIAL_FLOW_SCENARIO,
            SOLENOIDAL_FLOW_SCENARIO,
        )
        flow = differentiate or self.scenario in (
            CYLINDER_SCENARIO,
            POISEUILLE_SCENARIO,
            KOVASZNAY_SCENARIO,
        )
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
                with torch.set_grad_enabled(differentiate):
                    xy.requires_grad_(differentiate)
                    output = self.model(xy)
                    if differentiate:
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
        elif self.scenario in (
            KOVASZNAY_SCENARIO,
            POISEUILLE_SCENARIO,
            CYLINDER_SCENARIO,
        ):
            result['pressure_label'] = 'Pressure p (dimensionless)'
        if self.grid.triangles is not None:
            result.update(
                triangles=self.grid.triangles,
                boundary_edges=self.grid.boundary_edges,
            )
        return result
