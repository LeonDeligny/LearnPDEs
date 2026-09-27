"""Cosine training domain and evaluation-only extrapolation diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import Tensor
from torch.nn import Module

from learnpdes.evaluation.metrics import compare_exact
from learnpdes.model.encodings import identity
from learnpdes.model.objectives import CollocationObjective
from learnpdes.scenarios._builders import build_collocation
from learnpdes.scenarios._sampling import load_real_space
from learnpdes.scenarios.base import Scenario
from learnpdes.types import Analytical, CollocationData
from learnpdes.visualization.grids import VisualizationGrid, line_grid

if TYPE_CHECKING:
    from learnpdes.config import RunConfig
    from learnpdes.model.pinn import PINN


TRAINING_BOUNDS = (-np.pi, np.pi)
EVALUATION_BOUNDS = (-3 * np.pi, 3 * np.pi)
EVALUATION_POINTS = 601
REGION_LABELS = {
    'inside': 'Inside [−π, π]',
    'outside': 'Outside: π < |x| ≤ 2π',
    'outside_3pi': 'Outside to 3π: π < |x| ≤ 3π',
    'far_outside': 'Far outside: 2π < |x| ≤ 3π',
    'full': 'Full [−2π, 2π]',
    'full_3pi': 'Full [−3π, 3π]',
}
REGION_TITLES = {
    'inside': 'Inside MSE',
    'outside': 'Outside to 2π MSE',
    'full': 'Full ±2π MSE',
    'outside_3pi': 'Outside to 3π MSE',
    'far_outside': 'Far band MSE',
    'full_3pi': 'Full ±3π MSE',
}


def validate_order(order: int) -> None:
    if isinstance(order, bool) or not isinstance(order, int) or order < 2 or order % 2:
        raise ValueError('Cosinus derivative order must be an even integer >= 2.')


def region_masks(coordinates: ArrayLike) -> dict[str, np.ndarray]:
    """Separate the near and far bands; tolerate float32 interval endpoints."""
    x = np.asarray(coordinates).ravel()
    tolerance = 4 * np.finfo(np.float32).eps
    inside = np.abs(x) <= np.pi + tolerance
    full = np.abs(x) <= 2 * np.pi + tolerance
    extended = np.abs(x) <= 3 * np.pi + tolerance
    masks = {'inside': inside, 'outside': full & ~inside, 'full': full}
    # Old recordings only reach ±2π. Do not label their truncated interval ±3π.
    if (extended & ~full).any():
        masks.update(
            outside_3pi=extended & ~inside,
            far_outside=extended & ~full,
            full_3pi=extended,
        )
    return masks


def region_mse(coordinates: ArrayLike, prediction: ArrayLike) -> dict[str, float]:
    """Compare to cos(x), never supplying these targets to the optimizer."""
    x = np.asarray(coordinates).ravel()
    values = np.asarray(prediction).ravel()
    if values.shape != x.shape:
        raise ValueError('Cosinus predictions must match the evaluation coordinates.')
    error = (values - np.cos(x)) ** 2
    return {
        region: float(error[mask].mean())
        for region, mask in region_masks(x).items()
        if mask.any()
    }


def load_cosinus(
    num_inputs: int,
) -> CollocationData:
    """Load cosine collocation points on [-pi, pi], including the initial point."""
    # Define constants
    output_dim = 1
    input_homeo = identity
    output_homeo = identity
    encoding = identity
    analytical = np.cos

    # Load space
    x, mesh_masks = load_real_space(num_inputs, TRAINING_BOUNDS)

    return (
        x,
        mesh_masks,
        output_dim,
        analytical,
        input_homeo,
        output_homeo,
        encoding,
    )


class Objective(CollocationObjective):
    supports_derivative_order = True

    def cosinus_loss(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Cumulative even-order residuals, sampled only on the training domain.

        Order 2 preserves the original initial conditions. Each added order n
        contributes f^(n) + f^(n-2) = 0 and f^(n)(0) = (-1)^(n/2).
        """
        if self.cosinus_derivatives is None:
            derivatives = [self.forward(self.inputs)]
            for _ in range(self.cosinus_order):
                derivatives.append(self.partial_derivative(derivatives[-1], self.x))
        else:
            derivatives = self.cosinus_derivatives(self.inputs, self.cosinus_order)
        f, df_dx, ddf_dxdx = derivatives[:3]

        # f'' = -f
        physics_loss = self.mse_loss(f, -ddf_dxdx)

        # f(0) = 1, f'(0) = 0
        boundary_loss = self.mse_loss(
            f[self.zero_mask].view(-1, 1),
            self.one_tensor,
        ) + self.mse_loss(df_dx[self.zero_mask].view(-1, 1), self.zero_tensor)
        previous = ddf_dxdx
        for order in range(4, self.cosinus_order + 1, 2):
            current = derivatives[order]
            physics_loss = physics_loss + (current + previous).square().mean()
            boundary_loss = (
                boundary_loss
                + (current[self.zero_mask] - (-1) ** (order // 2)).square().mean()
            )
            previous = current
        return (
            self.process(physics_loss, boundary_loss),
            self.inputs,
            f,
            self.inputs_mask,
        )

    loss = cosinus_loss


def build(
    config: 'RunConfig',
) -> tuple['PINN', CollocationObjective, Analytical | None]:
    if config.points is None:
        raise ValueError('Cosinus requires a point count.')
    return build_collocation(config, load_cosinus(config.points), Objective)


def evaluate(model: 'Module', analytical: Analytical | None) -> dict[str, float]:
    coordinates = torch.linspace(*EVALUATION_BOUNDS, EVALUATION_POINTS).view(-1, 1)
    return compare_exact(
        model,
        coordinates,
        analytical,
        diagnostics=lambda x, y: {
            f'{region}_mse': value for region, value in region_mse(x, y).items()
        },
    )


def visualization_grid(
    input_space: Tensor, resolution: int, **kwargs: object
) -> VisualizationGrid:
    return line_grid(input_space, resolution, bounds=EVALUATION_BOUNDS)


SCENARIO = Scenario(
    'cosinus',
    "Harmonic oscillator: f''+f=0 on [-pi, pi]",
    64,
    build=build,
    grid=visualization_grid,
    evaluate=evaluate,
    reference_type='analytical',
    supports_derivative_order=True,
)
