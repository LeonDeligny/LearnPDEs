"""Runnable cylinder scenario and its shared-runner adapters."""

from __future__ import annotations

import numpy as np
from torch import Tensor
from torch.nn import Module

from learnpdes.config import RunConfig
from learnpdes.evaluation.fluid import FluidEvaluation
from learnpdes.model.fluid import FluidObjective, build_fluid
from learnpdes.model.pinn import PINN
from learnpdes.scenarios.base import Scenario
from learnpdes.scenarios.cylinder.evaluation import cylinder_observables
from learnpdes.scenarios.cylinder.problem import CylinderProblem
from learnpdes.types import Analytical
from learnpdes.visualization.grids import VisualizationGrid, ring_grid

EVALUATION = FluidEvaluation(
    pressure_gauge='Outlet normal stress u_x/Re-p=0',
    hard_boundaries=True,
    observables=cylinder_observables,
    diagnostic_limits=(('outlet_flow_relative_error', 0.01),),
)


def build(config: RunConfig) -> tuple[PINN, FluidObjective, Analytical | None]:
    return build_fluid(CylinderProblem(), config, None)


def evaluate(model: Module, analytical: Analytical | None = None) -> dict[str, float]:
    return EVALUATION.evaluate(model, CylinderProblem())


def visualization_grid(
    input_space: Tensor, resolution: int, **kwargs: object
) -> VisualizationGrid:
    problem = CylinderProblem()
    center = np.asarray(problem.center, dtype=float)

    def outer(directions: np.ndarray) -> np.ndarray:
        directions = np.asarray(directions, dtype=float)
        with np.errstate(divide='ignore'):
            distances = (
                np.where(directions >= 0, np.asarray([22.0, 4.1]) - center, -center)
                / directions
            )
        return distances.min(axis=1)

    return ring_grid(center, problem.radius, outer, resolution)


SCENARIO = Scenario(
    'cylinder',
    'Exploratory DFG cylinder, Re=20; no exact reference',
    45,
    hidden_dim=64,
    fluid=True,
    build=build,
    grid=visualization_grid,
    evaluate=evaluate,
    fluid_evaluation=EVALUATION,
    output_kind='velocity_pressure',
)
