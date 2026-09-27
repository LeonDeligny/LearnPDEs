"""Exponential problem definition, sampling, objective, and evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module

from learnpdes.evaluation.metrics import compare_exact
from learnpdes.model.encodings import identity
from learnpdes.model.objectives import CollocationObjective
from learnpdes.scenarios._builders import build_collocation
from learnpdes.scenarios._sampling import load_real_space
from learnpdes.scenarios.base import Scenario
from learnpdes.types import Analytical, CollocationData
from learnpdes.visualization.grids import line_grid

if TYPE_CHECKING:
    from learnpdes.config import RunConfig
    from learnpdes.model.pinn import PINN


def load_exponential(
    num_inputs: int,
) -> CollocationData:
    """Load configuration space (around 0) for exponential PDE."""
    # Define constants
    output_dim = 1
    input_homeo = identity
    output_homeo = identity
    encoding = identity
    analytical = np.exp

    # Load space
    x, mesh_masks = load_real_space(num_inputs)

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
    def exponential_loss(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        f = self.forward(self.x)
        df_dx = self.partial_derivative(f, self.x)

        # f' = f
        physics_loss = self.mse_loss(f, df_dx)

        # f(0) = 1
        boundary_loss = self.mse_loss(
            f[self.zero_mask].view(-1, 1),
            self.one_tensor,
        )

        return (
            self.process(physics_loss, boundary_loss),
            self.inputs,
            f,
            self.inputs_mask,
        )

    loss = exponential_loss


def build(
    config: 'RunConfig',
) -> tuple['PINN', CollocationObjective, Analytical | None]:
    if config.points is None:
        raise ValueError('Exponential requires a point count.')
    return build_collocation(config, load_exponential(config.points), Objective)


def evaluate(model: 'Module', analytical: Analytical | None) -> dict[str, float]:
    coordinates = torch.linspace(-3, 3, 201).view(-1, 1)
    return compare_exact(model, coordinates, analytical)


SCENARIO = Scenario(
    'exponential',
    "Exponential ODE: f'=f, f(0)=1",
    256,
    10000,
    build=build,
    grid=line_grid,
    evaluate=evaluate,
    reference_type='analytical',
)
