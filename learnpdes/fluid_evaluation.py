"""Exact-reference verification and physics diagnostics; no simulation data.

Kovasznay has an analytical reference. Cylinder observables are predictions,
not accuracy comparisons: no admissible reference is selected for that setup.
"""

import math

import numpy as np
import torch

from learnpdes import kovasznay
from learnpdes.fluid import CylinderProblem, derivative, sample_fluid


def cylinder_observables(model, problem, count=1024):
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


def evaluate_fluid(model, problem, *, count=2048, seed=1729):
    """Unseen uniform interior points, independent boundaries, no gauge fitting."""
    parameter = next(model.parameters())
    generator = torch.Generator().manual_seed(seed)
    samples = sample_fluid(
        problem, count, 256, generator=generator, near_cylinder=False
    )
    was_training = model.training
    model.eval()
    try:
        metrics = {}
        sums = {'continuity': 0.0, 'momentum_u': 0.0, 'momentum_v': 0.0}
        prediction_batches = []
        with torch.enable_grad():
            for batch in samples.interior.split(256):
                xy = batch.to(parameter).detach().requires_grad_(True)
                fields = model(xy)
                prediction_batches.append(fields.detach())
                for name, residual in problem.residuals(xy, fields).items():
                    sums[name] += residual.detach().square().sum().item()
            metrics.update(
                {
                    f'{name}_rms': math.sqrt(value / count)
                    for name, value in sums.items()
                }
            )
            boundary = {
                name: xy.to(parameter).detach().requires_grad_(True)
                for name, xy in samples.boundary.items()
            }
            for name, residual in problem.boundary_residuals(model, boundary).items():
                metrics[f'{name}_rms'] = residual.detach().square().mean().sqrt().item()
            if isinstance(problem, CylinderProblem):
                # The DFG force/pressure tables are simulation data and forbidden
                # even for evaluation. Retain only predictions and physics checks.
                metrics.update(cylinder_observables(model, problem))
                return metrics
        xy = samples.interior.to(parameter)
        with torch.no_grad():
            reference = torch.stack(kovasznay.analytical(*xy.unbind(1)), dim=1)
            error = torch.cat(prediction_batches) - reference
            metrics.update(
                {
                    'rmse': error.square().mean().sqrt().item(),
                    'max_error': error.abs().max().item(),
                    'relative_l2': (error.norm() / reference.norm()).item(),
                }
            )
            for index, name in enumerate(('u', 'v', 'p')):
                metrics[f'{name}_rmse'] = error[:, index].square().mean().sqrt().item()
                metrics[f'{name}_max_error'] = error[:, index].abs().max().item()
                metrics[f'{name}_relative_l2'] = (
                    error[:, index].norm() / reference[:, index].norm()
                ).item()
        return metrics
    finally:
        model.train(was_training)
