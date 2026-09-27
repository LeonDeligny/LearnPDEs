"""Laplace problem definition, sampling, objective, and evaluation."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from learnpdes import pi_tensor
from learnpdes.config import RunConfig
from learnpdes.evaluation.metrics import compare_exact
from learnpdes.model.encodings import identity
from learnpdes.model.objectives import CollocationObjective
from learnpdes.model.pinn import PINN
from learnpdes.scenarios._builders import build_collocation
from learnpdes.scenarios.base import Scenario
from learnpdes.types import Analytical, Array, CollocationData
from learnpdes.visualization.grids import rectangle_grid


def analytical(x: Array, y: Array) -> Array:
    """Exact solution for evaluation only."""
    return np.sin(np.pi * x) * np.sinh(np.pi * y) / np.sinh(np.pi)


def load_laplace(
    num_inputs: int,
) -> CollocationData:
    """Load a square [0, 1] x [0, 1] as input space for laplace PDE."""
    # Define constants
    output_dim = 1
    input_homeo = identity
    output_homeo = identity
    encoding = identity
    reference = analytical

    # Load space
    xy = torch.cartesian_prod(
        torch.linspace(0, 1, num_inputs),
        torch.linspace(0, 1, num_inputs),
    )

    # Create masks for each boundary
    x = xy[:, 0]
    y = xy[:, 1]
    mesh_masks = {
        'inlet': x == 0,
        'outlet': x == 1,
        'bottom': y == 0,
        'top': y == 1,
    }

    return (
        xy,
        mesh_masks,
        output_dim,
        reference,
        input_homeo,
        output_homeo,
        encoding,
    )


class Objective(CollocationObjective):
    def generate_boundaries(self) -> None:
        super().generate_boundaries()
        self.generate_laplace_boundary()

    def generate_laplace_boundary(self) -> None:
        self.sin = torch.sin(pi_tensor * self.x[self.top_mask]).view(-1, 1)

    def laplace_loss(self) -> tuple[Tensor, Tensor, Tensor, None]:
        f = self.forward(self.inputs)

        # Compute the second derivatives
        df_dx = self.partial_derivative(f, self.x)
        ddf_dxdx = self.partial_derivative(df_dx, self.x)

        df_dy = self.partial_derivative(f, self.y)
        ddf_dydy = self.partial_derivative(df_dy, self.y)

        # Delta f = 0
        physics_loss = self.mse_loss(ddf_dxdx, -ddf_dydy)

        # Dirichlet boundary conditions
        boundary_loss = (
            self.mse_loss(
                # f(bottom) = 0
                f[self.bottom_mask].view(-1, 1),
                self.bottom_zero_tensor,
            )
            + self.mse_loss(
                # f(top) = sin(pi x)
                f[self.top_mask].view(-1, 1),
                self.sin,
            )
            + self.mse_loss(
                # f(inlet) = 0
                f[self.inlet_mask].view(-1, 1),
                self.inlet_zero_tensor,
            )
            + self.mse_loss(
                # f(outlet) = 0
                f[self.outlet_mask].view(-1, 1),
                self.outlet_zero_tensor,
            )
        )
        return self.process(physics_loss, boundary_loss), self.inputs, f, None

    loss = laplace_loss


def build(config: RunConfig) -> tuple[PINN, CollocationObjective, Analytical | None]:
    if config.points is None:
        raise ValueError('Resolved configuration is missing a point count.')
    return build_collocation(config, load_laplace(config.points), Objective)


def evaluate(model: torch.nn.Module, analytical: Analytical | None) -> dict[str, float]:
    coordinates = torch.cartesian_prod(
        torch.linspace(0, 1, 41), torch.linspace(0, 1, 41)
    )
    return compare_exact(model, coordinates, analytical)


SCENARIO = Scenario(
    'laplace',
    'Laplace equation on the unit square',
    21,
    build=build,
    grid=rectangle_grid,
    evaluate=evaluate,
    reference_type='analytical',
)
