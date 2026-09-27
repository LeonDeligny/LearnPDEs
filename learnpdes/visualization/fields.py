"""Batched prediction and physical field decoding for plots."""

from __future__ import annotations

import numpy as np
import torch

from learnpdes.types import FieldEvaluation
from learnpdes.visualization.grids import VisualizationGrid


class ModelEvaluator:
    """Bound memory use and differentiate only when the plotted field needs it."""

    def __init__(
        self,
        model: torch.nn.Module,
        scenario: str,
        grid: VisualizationGrid,
        density: float = 1.225,
        batch_size: int = 8192,
    ) -> None:
        """Configure batched field evaluation on an independent display grid."""
        self.model = model
        self.scenario = scenario
        self.grid = grid
        self.density = float(density)
        self.batch_size = batch_size
        if batch_size < 1:
            raise ValueError('Evaluation batch size must be positive.')

    def __call__(self) -> FieldEvaluation:
        parameter = next(self.model.parameters())
        from learnpdes.scenarios.registry import get_scenario

        kind = get_scenario(self.scenario).output_kind
        differentiate = kind in ('potential', 'streamfunction')
        flow = kind != 'scalar'
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
                        if kind == 'potential':
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
        result: FieldEvaluation = {
            'inputs': self.grid.coordinates,
            'f': tuple(values[:, i] for i in range(3)) if flow else values,
            'geometry_mask': None,
        }
        if kind == 'streamfunction':
            result['pressure_label'] = 'Pressure p (not modeled)'
        elif kind == 'velocity_pressure':
            result['pressure_label'] = 'Pressure p (dimensionless)'
        if self.grid.triangles is not None:
            result.update(
                triangles=self.grid.triangles,
                boundary_edges=self.grid.boundary_edges,
            )
        return result
