"""Airfoil geometry with potential and streamfunction formulation variants."""

from __future__ import annotations

from pathlib import Path

from torch import Tensor

from learnpdes import POTENTIAL_FLOW_SCENARIO, SOLENOIDAL_FLOW_SCENARIO
from learnpdes.config import RunConfig
from learnpdes.model.objectives import CollocationObjective
from learnpdes.model.pinn import PINN
from learnpdes.resources import DEFAULT_MESH
from learnpdes.scenarios._builders import build_collocation
from learnpdes.scenarios.airfoil.objective import (
    PotentialObjective,
    StreamfunctionObjective,
)
from learnpdes.scenarios.airfoil.sampling import load_2d_mesh
from learnpdes.scenarios.base import Scenario
from learnpdes.types import Analytical
from learnpdes.visualization.grids import VisualizationGrid, airfoil_grid


def build_potential(
    config: RunConfig,
) -> tuple[PINN, CollocationObjective, Analytical | None]:
    if config.points is None:
        raise ValueError('Airfoil flow requires a point count.')
    points = config.points
    return build_collocation(
        config,
        load_2d_mesh(points, filepath=config.mesh_path or DEFAULT_MESH),
        PotentialObjective,
        formulation=POTENTIAL_FLOW_SCENARIO,
    )


def build_streamfunction(
    config: RunConfig,
) -> tuple[PINN, CollocationObjective, Analytical | None]:
    if config.points is None:
        raise ValueError('Airfoil flow requires a point count.')
    points = config.points
    return build_collocation(
        config,
        load_2d_mesh(points, filepath=config.mesh_path or DEFAULT_MESH),
        StreamfunctionObjective,
        formulation=SOLENOIDAL_FLOW_SCENARIO,
    )


def visualization_grid(
    input_space: Tensor, resolution: int, *, mesh_path: str | Path | None = DEFAULT_MESH
) -> VisualizationGrid:
    path = DEFAULT_MESH if mesh_path is None else mesh_path
    return airfoil_grid(path)


POTENTIAL = Scenario(
    'potential-flow',
    'Exploratory airfoil potential flow; no exact reference',
    32,
    mesh=True,
    formulation=POTENTIAL_FLOW_SCENARIO,
    build=build_potential,
    grid=visualization_grid,
    output_kind='potential',
)
STREAMFUNCTION = Scenario(
    'solenoidal-flow',
    'Exploratory airfoil streamfunction; no exact reference',
    32,
    mesh=True,
    formulation=SOLENOIDAL_FLOW_SCENARIO,
    build=build_streamfunction,
    grid=visualization_grid,
    output_kind='streamfunction',
)
