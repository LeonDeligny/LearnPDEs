"""Independent exact evaluator for steady circular Couette flow.

Inner radius 1 is stationary; outer radius 2 has tangential speed 1. Pressure
is zero at the outer wall. With a=2/3 and b=-2/3, u_theta=a*r+b/r and
p=a²*r²/2+2*a*b*log(r)-b²/(2*r²)+C. No forcing is required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import overload

import numpy as np
import torch

from learnpdes.config import RunConfig
from learnpdes.evaluation.fluid import FluidEvaluation
from learnpdes.model.fluid import FluidObjective, FluidSamples, build_fluid
from learnpdes.model.pinn import PINN
from learnpdes.physics.navier_stokes import navier_stokes
from learnpdes.scenarios.base import Scenario
from learnpdes.types import Analytical, Array, TensorFunction
from learnpdes.visualization.grids import VisualizationGrid, ring_grid


@overload
def analytical(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


@overload
def analytical(x: Array, y: Array) -> tuple[Array, Array, Array]: ...


def analytical(
    x: torch.Tensor | Array, y: torch.Tensor | Array
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[Array, Array, Array]:
    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        r2 = x * x + y * y
        angular_velocity = (2 / 3) * (1 - 1 / r2)
        pressure = (2 * r2 - 4 * torch.log(r2) - 2 / r2) / 9
        outer_pressure = (8 - 4 * np.log(4) - 0.5) / 9
        return -y * angular_velocity, x * angular_velocity, pressure - outer_pressure
    x_array, y_array = np.asarray(x), np.asarray(y)
    r2_array = x_array * x_array + y_array * y_array
    angular_velocity_array = (2 / 3) * (1 - 1 / r2_array)
    pressure_array = (2 * r2_array - 4 * np.log(r2_array) - 2 / r2_array) / 9
    outer_pressure = (8 - 4 * np.log(4) - 0.5) / 9
    return (
        -y_array * angular_velocity_array,
        x_array * angular_velocity_array,
        pressure_array - outer_pressure,
    )


@dataclass(frozen=True)
class CircularCouetteProblem:
    """A curved-wall NS verification case with an exact solution, Re=10."""

    name: str = 'circular-couette'
    reynolds: float = 10.0
    x_bounds: tuple[float, float] = (-2.0, 2.0)
    y_bounds: tuple[float, float] = (-2.0, 2.0)
    inner_radius: float = 1.0
    outer_radius: float = 2.0

    def contains(self, xy: torch.Tensor) -> torch.Tensor:
        r2 = xy.square().sum(1)
        return (r2 > self.inner_radius**2) & (r2 < self.outer_radius**2)

    def residuals(
        self, xy: torch.Tensor, fields: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return navier_stokes(xy, fields, self.reynolds)

    def output_transform(self, xy: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        # A polynomial extension of the prescribed wall velocities. This is
        # not the rational exact interior solution, which lives in the evaluator.
        r2 = xy.square().sum(1, keepdim=True)
        fraction = (r2 - 1) / 3
        wall_extension = 0.5 * fraction * torch.cat((-xy[:, 1:], xy[:, :1]), 1)
        velocity = wall_extension + fraction * (1 - fraction) * raw[:, :2]
        return torch.cat((velocity, raw[:, 2:]), 1)

    def boundary_residuals(
        self, forward: TensorFunction, boundary: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        terms = {}
        for name, xy in boundary.items():
            fields = forward(xy)
            if name == 'pressure':
                terms['pressure'] = fields[:, 2:]
            else:
                target = (
                    0.5 * torch.cat((-xy[:, 1:], xy[:, :1]), 1)
                    if name == 'outer'
                    else torch.zeros_like(xy)
                )
                terms[f'{name}_u'] = fields[:, :1] - target[:, :1]
                terms[f'{name}_v'] = fields[:, 1:2] - target[:, 1:2]
        return terms

    def sample(
        self,
        interior_count: int,
        boundary_count: int,
        *,
        generator: torch.Generator,
        near_cylinder: bool = True,
    ) -> FluidSamples:
        if interior_count < 1 or boundary_count < 1:
            raise ValueError('Interior and boundary sample counts must be positive.')

        def rand(count: int, dimensions: int) -> torch.Tensor:
            return torch.rand(
                count, dimensions, generator=generator, dtype=torch.float64
            )

        def circle(count: int, radii: float | torch.Tensor) -> torch.Tensor:
            angle = 2 * math.pi * rand(count, 1)
            return radii * torch.cat((angle.cos(), angle.sin()), 1)

        radius = (1 + 3 * rand(interior_count, 1)).sqrt()
        return FluidSamples(
            circle(interior_count, radius),
            {
                'inner': circle(boundary_count, 1),
                'outer': circle(boundary_count, 2),
                'pressure': torch.tensor([[2.0, 0.0]], dtype=torch.float64),
            },
        )

    def conservation_residuals(
        self, forward: TensorFunction, template: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return {}


EVALUATION = FluidEvaluation(
    pressure_gauge='p(2, 0)=0',
    analytical=analytical,
    hard_boundaries=True,
    implementation='learnpdes/scenarios/circular_couette.py:analytical',
    formula='r^2=x^2+y^2; u=-2*y*(1-1/r^2)/3; v=2*x*(1-1/r^2)/3; p=(2*r^2-4*log(r^2)-2/r^2)/9-(8-4*log(4)-0.5)/9',
    source='Project-derived annular Couette formula; substitution and hand checkpoints in tests/scenarios/test_circular_couette.py',
)


def build(config: RunConfig) -> tuple[PINN, FluidObjective, Analytical | None]:
    return build_fluid(CircularCouetteProblem(), config, analytical)


def evaluate(
    model: torch.nn.Module, analytical: Analytical | None = None
) -> dict[str, float]:
    return EVALUATION.evaluate(model, CircularCouetteProblem())


def visualization_grid(
    input_space: torch.Tensor, resolution: int, **kwargs: object
) -> VisualizationGrid:
    return ring_grid((0, 0), 1.0, 2.0, resolution, outer_boundary=True)


SCENARIO = Scenario(
    'circular-couette',
    'Exact viscous annular flow; curved-wall verification, Re=10',
    32,
    1500,
    64,
    fluid=True,
    build=build,
    grid=visualization_grid,
    evaluate=evaluate,
    reference_type='analytical',
    fluid_evaluation=EVALUATION,
    output_kind='velocity_pressure',
)
