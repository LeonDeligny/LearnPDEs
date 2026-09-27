"""Autograd residuals for dimensionless incompressible Navier–Stokes."""

import torch


def derivative(value: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    """Differentiate a scalar field, including constant and affine test fields."""
    if not value.requires_grad:
        return coordinates * 0
    result = torch.autograd.grad(
        value.sum(), coordinates, create_graph=True, allow_unused=True
    )[0]
    return coordinates * 0 if result is None else result + coordinates * 0


def navier_stokes(
    coordinates: torch.Tensor, fields: torch.Tensor, reynolds: float
) -> dict[str, torch.Tensor]:
    """Continuity and both momentum equations; at most second derivatives."""
    u, v, p = fields.split(1, dim=1)
    du, dv, dp = (derivative(f, coordinates) for f in (u, v, p))
    lap_u, lap_v = (
        derivative(d[:, :1], coordinates)[:, :1]
        + derivative(d[:, 1:], coordinates)[:, 1:]
        for d in (du, dv)
    )
    return {
        'continuity': du[:, :1] + dv[:, 1:],
        'momentum_u': u * du[:, :1] + v * du[:, 1:] + dp[:, :1] - lap_u / reynolds,
        'momentum_v': u * dv[:, :1] + v * dv[:, 1:] + dp[:, 1:] - lap_v / reynolds,
    }
