"""Logistic initial-value problem: f' = f(1-f), f(0) = 1/2 on [0, 2]."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import ArrayLike

from learnpdes.config import RunConfig
from learnpdes.evaluation.metrics import compare_exact
from learnpdes.model.encodings import identity
from learnpdes.model.objectives import CollocationObjective
from learnpdes.model.pinn import PINN
from learnpdes.scenarios._builders import build_collocation
from learnpdes.scenarios.base import Scenario
from learnpdes.types import Analytical, Array, CollocationData, LossResult
from learnpdes.visualization.grids import line_grid


def analytical(t: ArrayLike) -> Array:
    """Exact solution for evaluation only."""
    return 1 / (1 + np.exp(-np.asarray(t)))


def load_logistic(num_inputs: int) -> CollocationData:
    t = torch.linspace(0, 2, num_inputs)
    return t, {'zero': t == 0}, 1, analytical, identity, identity, identity


class Objective(CollocationObjective):
    def loss(self) -> LossResult:
        f = self.forward(self.x)
        df_dt = self.partial_derivative(f, self.x)
        physics_loss = self.mse_loss(df_dt, f * (1 - f))
        boundary_loss = self.mse_loss(f[self.zero_mask], self.one_tensor / 2)
        return (
            self.process(physics_loss, boundary_loss),
            self.inputs,
            f,
            self.inputs_mask,
        )


def build(config: RunConfig) -> tuple[PINN, CollocationObjective, Analytical | None]:
    if config.points is None:
        raise ValueError('Resolved configuration is missing a point count.')
    return build_collocation(config, load_logistic(config.points), Objective)


def evaluate(model: torch.nn.Module, analytical: Analytical | None) -> dict[str, float]:
    coordinates = torch.linspace(0, 2, 201).view(-1, 1)
    return compare_exact(model, coordinates, analytical)


SCENARIO = Scenario(
    'logistic',
    "Logistic ODE: f'=f(1-f), f(0)=1/2",
    64,
    5000,
    build=build,
    grid=line_grid,
    evaluate=evaluate,
    reference_type='analytical',
)
