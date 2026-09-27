"""Steady fluid problems and physics-only objectives, independent of evaluation.

Coordinates, velocities and pressure are nondimensional. The cylinder uses
diameter D=0.1, mean inlet speed U=0.2 and density rho=1 as reference scales.
"""

from dataclasses import dataclass
from functools import cached_property
import math

import numpy as np
import torch

from learnpdes import kovasznay


def derivative(value, coordinates):
    """Differentiate a scalar field, including constant and affine test fields."""
    if not value.requires_grad:
        return coordinates * 0
    result = torch.autograd.grad(
        value.sum(), coordinates, create_graph=True, allow_unused=True
    )[0]
    return coordinates * 0 if result is None else result + coordinates * 0


def navier_stokes(coordinates, fields, reynolds):
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


@dataclass
class FluidSamples:
    interior: torch.Tensor
    boundary: dict[str, torch.Tensor]


@dataclass(frozen=True)
class KovasznayProblem:
    name: str = 'kovasznay'
    reynolds: float = kovasznay.REYNOLDS
    x_bounds: tuple = kovasznay.X_BOUNDS
    y_bounds: tuple = kovasznay.Y_BOUNDS

    def contains(self, xy):
        x, y = xy.unbind(1)
        return (
            (x > self.x_bounds[0])
            & (x < self.x_bounds[1])
            & (y > self.y_bounds[0])
            & (y < self.y_bounds[1])
        )

    def residuals(self, xy, fields):
        return navier_stokes(xy, fields, self.reynolds)

    def boundary_residuals(self, forward, boundary):
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


@dataclass(frozen=True)
class CylinderProblem:
    """DFG 2D-1, Re=20; rectangle minus a stationary, no-slip cylinder.

    The do-nothing outlet (nu*u_x-p, nu*v_x)=0 also fixes the pressure
    reference. Adding a separate arbitrary pressure pin would overconstrain it.
    """

    name: str = 'cylinder'
    reynolds: float = 20.0
    x_bounds: tuple = (0.0, 22.0)
    y_bounds: tuple = (0.0, 4.1)
    center: tuple = (2.0, 2.0)
    radius: float = 0.5

    def contains(self, xy):
        x, y = xy.unbind(1)
        radius_squared = (xy - xy.new_tensor(self.center)).square().sum(1)
        return (
            (x > 0) & (x < 22) & (y > 0) & (y < 4.1) & (radius_squared > self.radius**2)
        )

    def residuals(self, xy, fields):
        return navier_stokes(xy, fields, self.reynolds)

    def inlet_velocity(self, xy):
        y = xy[:, 1:2]
        return 6 * y * (self.y_bounds[1] - y) / self.y_bounds[1] ** 2

    def output_transform(self, xy, raw):
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

    def boundary_residuals(self, forward, boundary):
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
    def flux_quadrature(self):
        """Integrate through fluid cross-sections, including the cylinder gap.

        Integral continuity supplies the same inlet flux at every section. This
        prevents low-flow fits that hide continuity defects between PDE samples.
        All targets follow from the inlet condition, not an interior reference.
        """
        nodes, weights = np.polynomial.legendre.leggauss(32)
        sections, section_weights = [], []
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

    def conservation_residuals(self, forward, template):
        xy, weights = self.flux_quadrature
        fields = forward(xy.to(template).flatten(0, 1)).reshape(*weights.shape, 3)
        flux = (fields[:, :, 0] * weights.to(template)).sum(1)
        return {'mass_flux': flux / (self.y_bounds[1] - self.y_bounds[0]) - 1}


def sample_fluid(
    problem, interior_count, boundary_count, *, generator, near_cylinder=True
):
    """Independent random interior/edge/circle samples, generated on CPU.

    Cylinder training uses stratified Sobol points with inlet/obstacle refinement.
    Independent uniform random validation disables this stratification.
    """
    if interior_count < 1 or boundary_count < 1:
        raise ValueError('Interior and boundary sample counts must be positive.')

    def rand(count, dimensions):
        return torch.rand(count, dimensions, generator=generator, dtype=torch.float64)

    lower = torch.tensor(
        [problem.x_bounds[0], problem.y_bounds[0]], dtype=torch.float64
    )
    upper = torch.tensor(
        [problem.x_bounds[1], problem.y_bounds[1]], dtype=torch.float64
    )
    cylinder_training = isinstance(problem, CylinderProblem) and near_cylinder
    sobol = (
        torch.quasirandom.SobolEngine(
            2, scramble=True, seed=int(torch.randint(2**31, (), generator=generator))
        )
        if cylinder_training
        else None
    )

    def rejection(count, hi):
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
    if isinstance(problem, CylinderProblem):
        angle = 2 * math.pi * rand(boundary_count, 1)
        boundary['cylinder'] = torch.tensor(
            problem.center
        ) + problem.radius * torch.cat((angle.cos(), angle.sin()), 1)
    else:
        boundary['pressure'] = upper.new_tensor(
            [[problem.x_bounds[1], problem.y_bounds[0]]]
        )
    return FluidSamples(interior, boundary)


class FluidObjective:
    """Evaluate fresh graphs on samples owned/refreshed by the trainer."""

    def __init__(self, problem, model, points, *, boundary_points=None):
        self.problem, self.model = problem, model
        self.interior_count = points**2
        self.boundary_count = boundary_points or 4 * points
        self.generator = torch.Generator().manual_seed(torch.initial_seed())
        self.components = {}
        self.rho = torch.tensor(1.0)
        self.resample()

    def resample(self):
        self.samples = sample_fluid(
            self.problem,
            self.interior_count,
            self.boundary_count,
            generator=self.generator,
        )
        # Domain corners are for metadata/visualization only, never PDE samples.
        self.input_space = torch.tensor(
            [[x, y] for x in self.problem.x_bounds for y in self.problem.y_bounds]
        )
        self.inputs = self.samples.interior

    def get_loss(self, scenario):
        if scenario != self.problem.name:
            raise ValueError('Scenario does not match the fluid problem.')
        return self.loss

    def loss(self):
        parameter = next(self.model.parameters())

        def fresh(xy):
            return xy.to(parameter).detach().clone().requires_grad_(True)

        xy = fresh(self.samples.interior)
        fields = self.model(xy)
        residuals = self.problem.residuals(xy, fields)
        residuals.update(
            self.problem.boundary_residuals(
                self.model,
                {name: fresh(values) for name, values in self.samples.boundary.items()},
            )
        )
        if isinstance(self.problem, CylinderProblem):
            residuals.update(self.problem.conservation_residuals(self.model, xy))
        terms = {name: value.square().mean() for name, value in residuals.items()}
        self.components = {name: value.detach().item() for name, value in terms.items()}
        # Velocity boundary constraints receive extra weight in Kovasznay.
        total = sum(
            value * (1 if name in ('continuity', 'momentum_u', 'momentum_v') else 10)
            for name, value in terms.items()
        )
        return total, xy, tuple(fields.split(1, dim=1)), None


def build_fluid_problem(scenario, points, *, hidden_dim=64, hidden_layers=4):
    from learnpdes import device
    from learnpdes.model.encodings import identity
    from learnpdes.model.pinn import PINN

    if points < 3:
        raise ValueError('Fluid flow requires at least 3 points per axis.')
    if scenario not in ('cylinder', 'kovasznay'):
        raise ValueError(f'Unknown fluid scenario: {scenario}')
    problem = CylinderProblem() if scenario == 'cylinder' else KovasznayProblem()
    model = PINN(
        {
            'input_dim': 2,
            'hidden_dim': hidden_dim,
            'output_dim': 3,
            'num_hidden_layers': hidden_layers,
            'activation': torch.nn.Tanh,
        },
        identity,
        identity,
        identity,
        output_transform=problem.output_transform if scenario == 'cylinder' else None,
    ).to(device)
    return (
        model,
        FluidObjective(problem, model, points),
        (kovasznay.analytical if scenario == 'kovasznay' else None),
    )
