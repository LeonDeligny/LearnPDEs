"""Cylinder diagnostics; no simulation-derived or exact full-field reference."""

from __future__ import annotations

import math

import numpy as np
import torch

from learnpdes.physics.navier_stokes import derivative
from learnpdes.scenarios.cylinder.problem import CylinderProblem


def cylinder_observables(
    model: torch.nn.Module, problem: CylinderProblem, count: int = 1024
) -> dict[str, float]:
    """Periodic surface quadrature using normals pointing out of the solid."""
    parameter = next(model.parameters())
    angle = torch.arange(count, device=parameter.device, dtype=parameter.dtype) * (
        2 * math.pi / count
    )
    normal = torch.stack((angle.cos(), angle.sin()), dim=1)
    xy = (
        (normal * problem.radius + normal.new_tensor(problem.center))
        .detach()
        .requires_grad_(True)
    )
    fields = model(xy)
    du, dv = (derivative(fields[:, i : i + 1], xy) for i in (0, 1))
    # Benchmark stress nu*grad(u)-p*I. At an incompressible no-slip wall,
    # the symmetric-gradient stress has the same integrated force.
    traction = (
        torch.cat(
            ((du * normal).sum(1, keepdim=True), (dv * normal).sum(1, keepdim=True)),
            dim=1,
        )
        / problem.reynolds
        - fields[:, 2:] * normal
    )
    force = traction.mean(0) * (2 * math.pi * problem.radius)
    # D=1, U_mean=1, rho=1: C = 2*F.
    front_back = xy.new_tensor(
        [
            [problem.center[0] - problem.radius, problem.center[1]],
            [problem.center[0] + problem.radius, problem.center[1]],
        ]
    )
    with torch.no_grad():
        pressure = model(front_back)[:, 2]
        nodes, weights = np.polynomial.legendre.leggauss(64)
        height = problem.y_bounds[1] - problem.y_bounds[0]
        y = xy.new_tensor((nodes + 1) * height / 2 + problem.y_bounds[0])
        quadrature_weights = xy.new_tensor(weights * height / 2)
        flux = []
        for x in problem.x_bounds:
            section = torch.stack((torch.full_like(y, x), y), dim=1)
            flux.append((model(section)[:, 0] * quadrature_weights).sum().item())
    return {
        'drag': 2 * force[0].item(),
        'lift': 2 * force[1].item(),
        'pressure_drop': (pressure[0] - pressure[1]).item(),
        'inlet_flow_rate': flux[0],
        'outlet_flow_rate': flux[1],
        'mass_balance_relative_error': abs(flux[1] - flux[0]) / height,
        'outlet_flow_relative_error': abs(flux[1] - height) / height,
    }
