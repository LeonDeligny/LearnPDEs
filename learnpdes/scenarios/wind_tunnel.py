"""Potential flow through a rectangular tunnel without an obstacle."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from learnpdes import POTENTIAL_FLOW_SCENARIO
from learnpdes.model.encodings import identity
from learnpdes.model.objectives import CollocationObjective
from learnpdes.scenarios._builders import build_collocation
from learnpdes.scenarios.airfoil.objective import PotentialObjective
from learnpdes.scenarios.base import Scenario
from learnpdes.types import Analytical, LossFunction
from learnpdes.visualization.grids import (
    VisualizationGrid,
    rectangle_grid,
    rectangular_triangles,
)

if TYPE_CHECKING:
    from learnpdes.config import RunConfig
    from learnpdes.model.pinn import PINN


WIND_TUNNEL_SCENARIO = 'wind_tunnel_no_geometry'


def load_wind_tunnel(
    num_inputs: int,
) -> tuple[
    Tensor,
    dict[str, Tensor],
    int,
    Analytical | None,
    Callable[[Tensor], Tensor],
    Callable[[Tensor], Tensor],
    Callable[[Tensor], Tensor],
]:
    """Load a square [0, 4] x [0, 1] as input space."""
    # Define constants
    output_dim = 1
    input_homeo = identity
    output_homeo = identity
    encoding = identity

    # Load space
    xy = torch.cartesian_prod(
        torch.linspace(0, 4, num_inputs),
        torch.linspace(0, 1, num_inputs),
    )

    # Create masks for each boundary
    x = xy[:, 0]
    y = xy[:, 1]
    mesh_masks = {
        'inlet': x == 0,
        'outlet': x == 4,
        'wall': ((y == 0) | (y == 1)),
    }

    return (
        xy,
        mesh_masks,
        output_dim,
        None,
        input_homeo,
        output_homeo,
        encoding,
    )


class Objective(PotentialObjective):
    def get_loss(self, scenario: str) -> LossFunction:
        super().get_loss(scenario)  # Validate the requested scenario.
        return partial(self.loss, pre=True)


def build(
    config: 'RunConfig',
) -> tuple['PINN', CollocationObjective, Analytical | None]:
    if config.points is None:
        raise ValueError('Wind tunnel requires a point count.')
    data = load_wind_tunnel(config.points)
    data[1]['airfoil'] = torch.zeros(len(data[0]), dtype=torch.bool)
    return build_collocation(
        config, data, Objective, formulation=POTENTIAL_FLOW_SCENARIO
    )


def visualization_grid(
    input_space: Tensor, resolution: int, **kwargs: object
) -> VisualizationGrid:
    grid = rectangle_grid(input_space, resolution)
    grid.triangles = rectangular_triangles(grid.coordinates)
    return grid


SCENARIO = Scenario(
    WIND_TUNNEL_SCENARIO,
    'Potential flow in a channel without an obstacle',
    32,
    1000,
    formulation=POTENTIAL_FLOW_SCENARIO,
    build=build,
    grid=visualization_grid,
    output_kind='potential',
)
