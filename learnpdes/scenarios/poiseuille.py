"""Plane Poiseuille benchmark in nondimensional model units.

Stationary plates bound 0 <= y <= H; pressure decreases along 0 <= x <= L.
The reference is for evaluation only, never a velocity target during training.
"""

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
from learnpdes.scenarios.base import Scenario
from learnpdes.types import Analytical, CollocationData
from learnpdes.visualization.grids import rectangle_grid

if TYPE_CHECKING:
    from learnpdes.config import RunConfig
    from learnpdes.model.pinn import PINN


LENGTH = 4.0
HEIGHT = 1.0
DENSITY = 1.0
VISCOSITY = 0.1  # Dynamic viscosity mu, not kinematic viscosity nu = mu / rho.
PRESSURE_DROP = 3.2
OUTLET_PRESSURE = 0.0
INLET_PRESSURE = OUTLET_PRESSURE + PRESSURE_DROP


def analytical(x: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (u, v, p): u = G y(H-y)/(2 mu), p = p_out + G(L-x)."""
    x, y = np.broadcast_arrays(x, y)
    forcing = PRESSURE_DROP / LENGTH  # G = -dp/dx > 0.
    u = forcing * y * (HEIGHT - y) / (2 * VISCOSITY)
    v = np.zeros_like(u)
    p = OUTLET_PRESSURE + forcing * (LENGTH - x)
    return u, v, p


def load_poiseuille(
    num_inputs: int,
) -> CollocationData:
    """Load a flat channel with primitive outputs (u, v, p)."""
    if num_inputs < 3:
        raise ValueError('Poiseuille flow requires at least 3 points per axis.')
    xy = torch.cartesian_prod(
        torch.linspace(0, LENGTH, num_inputs),
        torch.linspace(0, HEIGHT, num_inputs),
    )
    x, y = xy.unbind(dim=1)
    mesh_masks = {
        'inlet': x == 0,
        'outlet': x == LENGTH,
        'wall': (y == 0) | (y == HEIGHT),
    }
    return xy, mesh_masks, 3, analytical, identity, identity, identity


class Objective(CollocationObjective):
    rho = torch.full_like(CollocationObjective.rho, DENSITY)

    def poiseuille_loss(
        self,
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor, Tensor], None]:
        """Steady incompressible Navier–Stokes with pressure-driven open ends."""
        u, v, p = self.forward(self.inputs).split(1, dim=1)
        derivative = self.partial_derivative
        u_x, u_y = derivative(u, self.x), derivative(u, self.y)
        v_x, v_y = derivative(v, self.x), derivative(v, self.y)
        p_x, p_y = derivative(p, self.x), derivative(p, self.y)
        lap_u = derivative(u_x, self.x) + derivative(u_y, self.y)
        lap_v = derivative(v_x, self.x) + derivative(v_y, self.y)

        residuals = (
            u_x + v_y,
            self.rho * (u * u_x + v * u_y) + p_x - VISCOSITY * lap_u,
            self.rho * (u * v_x + v * v_y) + p_y - VISCOSITY * lap_v,
        )
        physics_loss = sum(
            (residual.square().mean() for residual in residuals),
            torch.zeros((), device=self.inputs.device),
        )

        # Stationary plates: no tangential slip and no penetration.
        boundary_loss = (
            u[self.wall_mask].square().mean() + v[self.wall_mask].square().mean()
        )
        # Fully developed ends: du/dx = 0 and v = 0. The prescribed pressures
        # drive the flow and fix its gauge without supplying a velocity profile.
        for mask, pressure in (
            (self.inlet_mask, INLET_PRESSURE),
            (self.outlet_mask, OUTLET_PRESSURE),
        ):
            boundary_loss = (
                boundary_loss
                + (p[mask] - pressure).square().mean()
                + u_x[mask].square().mean()
                + v[mask].square().mean()
            )
        return self.process(physics_loss, boundary_loss), self.inputs, (u, v, p), None

    loss = poiseuille_loss


def build(
    config: 'RunConfig',
) -> tuple['PINN', CollocationObjective, Analytical | None]:
    if config.points is None:
        raise ValueError('Poiseuille requires a point count.')
    return build_collocation(config, load_poiseuille(config.points), Objective)


def evaluate(model: 'Module', analytical: Analytical | None) -> dict[str, float]:
    coordinates = torch.cartesian_prod(
        torch.linspace(0, LENGTH, 41), torch.linspace(0, HEIGHT, 41)
    )
    return compare_exact(model, coordinates, analytical, components=('u', 'v', 'p'))


SCENARIO = Scenario(
    'poiseuille',
    'Pressure-driven viscous channel flow',
    21,
    build=build,
    grid=rectangle_grid,
    evaluate=evaluate,
    reference_type='analytical',
    output_kind='velocity_pressure',
)
