"""Cylinder training refinement and independent uniform evaluation sampling."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from learnpdes.scenarios.cylinder.problem import CylinderProblem

import math

import torch

from learnpdes.model.fluid import FluidSamples


def sample(
    problem: CylinderProblem,
    interior_count: int,
    boundary_count: int,
    *,
    generator: torch.Generator,
    near_cylinder: bool = True,
) -> FluidSamples:
    """Independent random interior/edge/circle samples, generated on CPU.

    Cylinder training uses stratified Sobol points with inlet/obstacle refinement.
    Independent uniform random validation disables this stratification.
    """
    if interior_count < 1 or boundary_count < 1:
        raise ValueError('Interior and boundary sample counts must be positive.')

    def rand(count: int, dimensions: int) -> torch.Tensor:
        return torch.rand(count, dimensions, generator=generator, dtype=torch.float64)

    lower = torch.tensor(
        [problem.x_bounds[0], problem.y_bounds[0]], dtype=torch.float64
    )
    upper = torch.tensor(
        [problem.x_bounds[1], problem.y_bounds[1]], dtype=torch.float64
    )
    cylinder_training = near_cylinder
    sobol = (
        torch.quasirandom.SobolEngine(
            2, scramble=True, seed=int(torch.randint(2**31, (), generator=generator))
        )
        if cylinder_training
        else None
    )

    def rejection(count: int, hi: torch.Tensor) -> torch.Tensor:
        batches, remaining = [], count
        while remaining:
            draw_count = max(remaining * 2, 32)
            unit = (
                sobol.draw(draw_count, dtype=torch.float64)
                if sobol is not None
                else rand(draw_count, 2)
            )
            candidates = lower + (hi - lower) * unit
            accepted = candidates[problem.contains(candidates)][:remaining]
            batches.append(accepted)
            remaining -= len(accepted)
        return torch.cat(batches)

    if cylinder_training and interior_count >= 5:
        near_count = interior_count // 5
        angle = 2 * math.pi * rand(near_count, 1)
        radius = (
            problem.radius**2 + (1.2**2 - problem.radius**2) * rand(near_count, 1)
        ).sqrt()
        ring = torch.tensor(problem.center) + radius * torch.cat(
            (angle.cos(), angle.sin()), 1
        )
        interior = torch.cat(
            (
                rejection(interior_count - 3 * near_count, upper),
                rejection(near_count, upper.new_tensor([5.0, 4.1])),
                rejection(near_count, upper.new_tensor([0.75, 4.1])),
                ring,
            )
        )
    else:
        interior = rejection(interior_count, upper)
    boundary = {}
    for name, axis, value in (
        ('inlet', 0, problem.x_bounds[0]),
        ('outlet', 0, problem.x_bounds[1]),
        ('bottom', 1, problem.y_bounds[0]),
        ('top', 1, problem.y_bounds[1]),
    ):
        xy = lower + (upper - lower) * rand(boundary_count, 2)
        xy[:, axis] = value
        boundary[name] = xy
    angle = 2 * math.pi * rand(boundary_count, 1)
    boundary['cylinder'] = torch.tensor(problem.center) + problem.radius * torch.cat(
        (angle.cos(), angle.sin()), 1
    )
    return FluidSamples(interior, boundary)
