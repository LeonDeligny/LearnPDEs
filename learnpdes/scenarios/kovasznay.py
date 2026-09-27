"""Dimensionless Kovasznay benchmark: steady incompressible Navier–Stokes.

The network predicts (u, v, p) directly, with unit density, no body force,
and kinematic viscosity 1 / Re. The rectangle contains no solid obstacle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast, overload

import numpy as np
import torch

from learnpdes.config import RunConfig
from learnpdes.evaluation.fluid import FluidEvaluation
from learnpdes.model.fluid import FluidObjective, FluidSamples, build_fluid
from learnpdes.model.pinn import PINN
from learnpdes.physics.navier_stokes import navier_stokes
from learnpdes.scenarios.base import Scenario
from learnpdes.types import Analytical, Array, TensorFunction
from learnpdes.visualization.grids import rectangle_grid

REYNOLDS = 40.0
VISCOSITY = 1.0 / REYNOLDS
X_BOUNDS = (-0.5, 1.0)
Y_BOUNDS = (-0.5, 1.5)
DECAY_RATE = REYNOLDS / 2 - math.sqrt(REYNOLDS**2 / 4 + 4 * math.pi**2)


@overload
def analytical(
    x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


@overload
def analytical(x: Array, y: Array) -> tuple[Array, Array, Array]: ...


def analytical(
    x: torch.Tensor | np.ndarray, y: torch.Tensor | np.ndarray
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[np.ndarray, np.ndarray, np.ndarray]
):
    """Return exact (u, v, p) arrays or tensors, retaining Torch autograd.

    Pressure uses the gauge p(0, y) = 0. Training fixes this same additive
    constant by prescribing the exact pressure at the bottom-right corner.
    """
    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        x_tensor, y_tensor = torch.broadcast_tensors(x, y)
        decay = torch.exp(DECAY_RATE * cast(torch.Tensor, x_tensor))
        phase = 2 * math.pi * cast(torch.Tensor, y_tensor)
        u = 1 - decay * torch.cos(phase)
        v = DECAY_RATE / (2 * math.pi) * decay * torch.sin(phase)
        p = (1 - decay**2) / 2
        return u, v, p
    x_array, y_array = np.broadcast_arrays(np.asarray(x), np.asarray(y))
    decay_array = np.exp(DECAY_RATE * x_array)
    phase_array = 2 * math.pi * y_array
    u_array = 1 - decay_array * np.cos(phase_array)
    v_array = DECAY_RATE / (2 * math.pi) * decay_array * np.sin(phase_array)
    p_array = (1 - decay_array**2) / 2
    return u_array, v_array, p_array


@dataclass(frozen=True)
class KovasznayProblem:
    name: str = 'kovasznay'
    reynolds: float = REYNOLDS
    x_bounds: tuple[float, float] = X_BOUNDS
    y_bounds: tuple[float, float] = Y_BOUNDS

    def contains(self, xy: torch.Tensor) -> torch.Tensor:
        x, y = xy.unbind(1)
        return (
            (x > self.x_bounds[0])
            & (x < self.x_bounds[1])
            & (y > self.y_bounds[0])
            & (y < self.y_bounds[1])
        )

    def residuals(
        self, xy: torch.Tensor, fields: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return navier_stokes(xy, fields, self.reynolds)

    def boundary_residuals(
        self, forward: TensorFunction, boundary: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # Only prescribed boundary data enter training. No interior evaluator.
        terms = {}
        decay_rate = self.reynolds / 2 - math.sqrt(
            self.reynolds**2 / 4 + 4 * math.pi**2
        )
        for name, xy in boundary.items():
            fields = forward(xy)
            decay = torch.exp(decay_rate * xy[:, :1])
            if name == 'pressure':
                terms['pressure'] = fields[:, 2:] - (1 - decay.square()) / 2
            else:
                phase = 2 * math.pi * xy[:, 1:]
                terms[f'{name}_u'] = fields[:, :1] - (1 - decay * torch.cos(phase))
                terms[f'{name}_v'] = fields[:, 1:2] - decay_rate / (
                    2 * math.pi
                ) * decay * torch.sin(phase)
        return terms

    def conservation_residuals(
        self, forward: TensorFunction, template: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return {}

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

        lower = torch.tensor([self.x_bounds[0], self.y_bounds[0]], dtype=torch.float64)
        upper = torch.tensor([self.x_bounds[1], self.y_bounds[1]], dtype=torch.float64)
        batches, remaining = [], interior_count
        while remaining:
            candidates = lower + (upper - lower) * rand(max(remaining * 2, 32), 2)
            accepted = candidates[self.contains(candidates)][:remaining]
            batches.append(accepted)
            remaining -= len(accepted)
        interior = torch.cat(batches)
        boundary = {}
        for name, axis, value in (
            ('inlet', 0, self.x_bounds[0]),
            ('outlet', 0, self.x_bounds[1]),
            ('bottom', 1, self.y_bounds[0]),
            ('top', 1, self.y_bounds[1]),
        ):
            xy = lower + (upper - lower) * rand(boundary_count, 2)
            xy[:, axis] = value
            boundary[name] = xy
        boundary['pressure'] = upper.new_tensor([[self.x_bounds[1], self.y_bounds[0]]])
        return FluidSamples(interior, boundary)


EVALUATION = FluidEvaluation(
    pressure_gauge='Exact pressure at (1, -0.5)',
    analytical=analytical,
    implementation='learnpdes/scenarios/kovasznay.py:analytical',
    formula='lambda=Re/2-sqrt(Re^2/4+4*pi^2); u=1-exp(lambda*x)*cos(2*pi*y); v=lambda/(2*pi)*exp(lambda*x)*sin(2*pi*y); p=(1-exp(2*lambda*x))/2',
    source='https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/Kovasznay.flow.html',
)


def build(config: RunConfig) -> tuple[PINN, FluidObjective, Analytical | None]:
    return build_fluid(KovasznayProblem(), config, analytical)


def evaluate(
    model: torch.nn.Module, analytical: Analytical | None = None
) -> dict[str, float]:
    return EVALUATION.evaluate(model, KovasznayProblem())


SCENARIO = Scenario(
    'kovasznay',
    'Navier-Stokes exact benchmark, Re=40',
    31,
    10000,
    64,
    fluid=True,
    build=build,
    grid=rectangle_grid,
    evaluate=evaluate,
    reference_type='analytical',
    fluid_evaluation=EVALUATION,
    output_kind='velocity_pressure',
)
