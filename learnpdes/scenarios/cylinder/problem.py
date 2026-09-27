"""Cylinder geometry, equations, prescribed boundaries, and flux constraint."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import torch

from learnpdes.model.fluid import FluidSamples
from learnpdes.physics.navier_stokes import derivative, navier_stokes
from learnpdes.scenarios.cylinder.sampling import sample
from learnpdes.types import TensorFunction


@dataclass(frozen=True)
class CylinderProblem:
    """DFG 2D-1, Re=20; rectangle minus a stationary, no-slip cylinder.

    The do-nothing outlet (nu*u_x-p, nu*v_x)=0 also fixes the pressure
    reference. Adding a separate arbitrary pressure pin would overconstrain it.
    """

    name: str = 'cylinder'
    reynolds: float = 20.0
    x_bounds: tuple[float, float] = (0.0, 22.0)
    y_bounds: tuple[float, float] = (0.0, 4.1)
    center: tuple[float, float] = (2.0, 2.0)
    radius: float = 0.5

    def contains(self, xy: torch.Tensor) -> torch.Tensor:
        x, y = xy.unbind(1)
        radius_squared = (xy - xy.new_tensor(self.center)).square().sum(1)
        return (
            (x > 0) & (x < 22) & (y > 0) & (y < 4.1) & (radius_squared > self.radius**2)
        )

    def residuals(
        self, xy: torch.Tensor, fields: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return navier_stokes(xy, fields, self.reynolds)

    def inlet_velocity(self, xy: torch.Tensor) -> torch.Tensor:
        y = xy[:, 1:2]
        return 6 * y * (self.y_bounds[1] - y) / self.y_bounds[1] ** 2

    def output_transform(self, xy: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        """Smooth boundary extension enforcing inlet and stationary walls exactly."""
        x, y = xy.split(1, dim=1)
        radius_squared = (x - self.center[0]) ** 2 + (y - self.center[1]) ** 2
        distance = -torch.expm1(-(radius_squared - self.radius**2))
        inlet_distance = -torch.expm1(
            -(self.center[0] ** 2 + (y - self.center[1]) ** 2 - self.radius**2)
        )
        profile = self.inlet_velocity(xy)
        correction = x / (1 + x)
        u = profile * distance / inlet_distance * (1 + correction * raw[:, :1])
        v = profile * distance * correction * raw[:, 1:2]
        return torch.cat((u, v, raw[:, 2:]), dim=1)

    def boundary_residuals(
        self, forward: TensorFunction, boundary: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        terms = {}
        for name, xy in boundary.items():
            fields = forward(xy)
            if name == 'outlet':
                du, dv = (derivative(fields[:, i : i + 1], xy) for i in (0, 1))
                terms['outlet_normal'] = du[:, :1] / self.reynolds - fields[:, 2:]
                terms['outlet_tangent'] = dv[:, :1] / self.reynolds
            else:
                target = self.inlet_velocity(xy) if name == 'inlet' else 0
                terms[f'{name}_u'] = fields[:, :1] - target
                terms[f'{name}_v'] = fields[:, 1:2]
        return terms

    @cached_property
    def flux_quadrature(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Integrate through fluid cross-sections, including the cylinder gap.

        Integral continuity supplies the same inlet flux at every section. This
        prevents low-flow fits that hide continuity defects between PDE samples.
        All targets follow from the inlet condition, not an interior reference.
        """
        nodes, weights = np.polynomial.legendre.leggauss(32)
        sections: list[np.ndarray] = []
        section_weights: list[np.ndarray] = []
        for x in (0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 10.0, 16.0, 22.0):
            half_gap = math.sqrt(max(self.radius**2 - (x - self.center[0]) ** 2, 0))
            segments = (
                (self.y_bounds[0], self.center[1] - half_gap),
                (self.center[1] + half_gap, self.y_bounds[1]),
            )
            coordinates, quadrature_weights = [], []
            for low, high in segments:
                y = low + (nodes + 1) * (high - low) / 2
                coordinates.append(np.column_stack((np.full_like(y, x), y)))
                quadrature_weights.append(weights * (high - low) / 2)
            sections.append(np.concatenate(coordinates))
            section_weights.append(np.concatenate(quadrature_weights))
        return torch.tensor(np.stack(sections)), torch.tensor(np.stack(section_weights))

    def conservation_residuals(
        self, forward: TensorFunction, template: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        xy, weights = self.flux_quadrature
        fields = forward(xy.to(template).flatten(0, 1)).reshape(*weights.shape, 3)
        flux = (fields[:, :, 0] * weights.to(template)).sum(1)
        return {'mass_flux': flux / (self.y_bounds[1] - self.y_bounds[0]) - 1}

    def collocation_quadrature(self) -> dict[str, torch.Tensor]:
        return {'mass_flux': self.flux_quadrature[0].flatten(0, 1)}

    def sample(
        self,
        interior_count: int,
        boundary_count: int,
        *,
        generator: torch.Generator,
        near_cylinder: bool = True,
    ) -> FluidSamples:
        return sample(
            self,
            interior_count,
            boundary_count,
            generator=generator,
            near_cylinder=near_cylinder,
        )
