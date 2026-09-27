"""Cylinder geometry, boundary conditions, conservation, and force diagnostics."""

import math
import unittest

import torch

from learnpdes.model.fluid import FluidObjective
from learnpdes.scenarios.cylinder.evaluation import cylinder_observables
from learnpdes.scenarios.cylinder.problem import CylinderProblem
from learnpdes.scenarios.registry import get_scenario
from learnpdes.training import build_problem


class TestCylinder(unittest.TestCase):
    def test_curved_wall_and_equations_against_exact_circular_couette_flow(
        self,
    ) -> None:
        # A separate exact steady NS solution around a stationary inner cylinder.
        # It validates curved-wall differentiation, not the channel's outer BCs.
        problem = CylinderProblem()
        samples = problem.sample(100, 100, generator=torch.Generator().manual_seed(17))

        def exact(xy: torch.Tensor) -> torch.Tensor:
            x, y = (xy - xy.new_tensor(problem.center)).unbind(1)
            r2, a2 = x * x + y * y, problem.radius**2
            speed = 1 - a2 / r2
            p = r2 / 2 - a2 * torch.log(r2) - a2 * a2 / (2 * r2)
            return torch.stack((-y * speed, x * speed, p), dim=1)

        xy = samples.interior.requires_grad_()
        for residual in problem.residuals(xy, exact(xy)).values():
            self.assertLess(residual.abs().max().item(), 1e-12)
        self.assertLess(
            exact(samples.boundary['cylinder'])[:, :2].abs().max().item(), 1e-14
        )

    def test_samples_exclude_solid_and_boundaries_and_are_reproducible(self) -> None:
        problem = CylinderProblem()
        first = problem.sample(1000, 64, generator=torch.Generator().manual_seed(8))
        repeat = problem.sample(1000, 64, generator=torch.Generator().manual_seed(8))
        torch.testing.assert_close(first.interior, repeat.interior)
        self.assertTrue(problem.contains(first.interior).all())
        self.assertEqual(len(first.interior), 1000)
        for name, xy in first.boundary.items():
            self.assertEqual(len(xy), 64)
            torch.testing.assert_close(xy, repeat.boundary[name])
            self.assertFalse(torch.isin(first.interior[:, 0], xy[:, 0]).all())
        radii = (first.boundary['cylinder'] - torch.tensor(problem.center)).norm(dim=1)
        torch.testing.assert_close(radii, torch.full_like(radii, problem.radius))

    def test_boundary_transform_and_outlet_pressure_reference(self) -> None:
        model, objective, reference = build_problem('cylinder', 5)
        assert isinstance(objective, FluidObjective)
        model = model.cpu().double()
        self.assertIsNone(reference)
        self.assertEqual(
            (model.hidden_dim, model.num_hidden_layers, model.output_dim), (64, 4, 3)
        )
        boundary = {
            name: xy.detach().clone().requires_grad_(True)
            for name, xy in objective.samples.boundary.items()
        }
        residuals = objective.problem.boundary_residuals(model, boundary)
        for name, value in residuals.items():
            if not name.startswith('outlet'):
                self.assertLess(value.abs().max().item(), 1e-14)

        def shifted(xy: torch.Tensor) -> torch.Tensor:
            return model(xy) + xy.new_tensor([0, 0, 0.3])

        shifted_residuals = objective.problem.boundary_residuals(shifted, boundary)
        torch.testing.assert_close(
            shifted_residuals['outlet_normal'], residuals['outlet_normal'] - 0.3
        )

    def test_integral_continuity_uses_only_fluid_cross_sections(self) -> None:
        problem = CylinderProblem()
        xy, weights = problem.flux_quadrature
        # A unit horizontal field integrates to the fluid section height.
        # At x=2 the diameter-one solid must be excluded from integration.
        self.assertTrue(
            ((xy - xy.new_tensor(problem.center)).norm(dim=2) > problem.radius).all()
        )
        at_center = xy[:, 0, 0] == problem.center[0]
        torch.testing.assert_close(
            weights[at_center].sum(1), torch.tensor([3.1], dtype=torch.float64)
        )
        self.assertAlmostEqual(weights[-1].sum().item(), 4.1, places=12)

        def uniform(points: torch.Tensor) -> torch.Tensor:
            return torch.stack(
                (points[:, 0] * 0 + 1, points[:, 0] * 0, points[:, 0] * 0), 1
            )

        residual = problem.conservation_residuals(uniform, xy)['mass_flux']
        self.assertAlmostEqual(residual[-1].item(), 0.0, places=12)
        self.assertAlmostEqual(residual[at_center].item(), -1 / 4.1, places=12)

    def test_force_sign_pressure_scaling_and_evaluator_restores_mode(self) -> None:
        class LinearPressure(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))

            def forward(self, xy: torch.Tensor) -> torch.Tensor:
                return self.scale * torch.stack(
                    (xy[:, 0] * 0, xy[:, 0] * 0, -xy[:, 0]), 1
                )

        model, problem = LinearPressure(), CylinderProblem()
        values = cylinder_observables(model, problem)
        self.assertAlmostEqual(
            values['drag'], 2 * math.pi * problem.radius**2, places=12
        )
        self.assertAlmostEqual(values['lift'], 0.0, places=12)
        self.assertAlmostEqual(values['pressure_drop'], 1.0, places=12)
        self.assertEqual(values['outlet_flow_relative_error'], 1.0)
        self.assertEqual(values['mass_balance_relative_error'], 0.0)
        with torch.no_grad():
            evaluator = get_scenario('cylinder').fluid_evaluation
            assert evaluator is not None
            metrics = evaluator.evaluate(model, problem, count=64)
        self.assertTrue(model.training)
        self.assertIsNone(model.scale.grad)
        self.assertAlmostEqual(metrics['momentum_u_rms'], 1.0, places=12)
        # Predictions remain available, but no comparison to a simulated
        # cylinder solution may be exported as an accuracy metric.
        for name in ('drag', 'lift', 'pressure_drop'):
            self.assertIn(name, metrics)
            self.assertNotIn(f'{name}_absolute_error', metrics)
            self.assertNotIn(f'{name}_relative_error', metrics)

    def test_force_quadrature_includes_viscosity_and_ignores_pressure_offset(
        self,
    ) -> None:
        class Polynomial(torch.nn.Module):
            def __init__(self, pressure: float) -> None:
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
                self.pressure = pressure

            def forward(self, xy: torch.Tensor) -> torch.Tensor:
                r2 = (xy - xy.new_tensor([2.0, 2.0])).square().sum(1)
                return self.scale * torch.stack((r2, 2 * r2, r2 * 0 + self.pressure), 1)

        problem = CylinderProblem()
        expected = 8 * math.pi * problem.radius**2 / problem.reynolds
        for pressure in (0.0, 17.0):
            values = cylinder_observables(Polynomial(pressure), problem)
            self.assertAlmostEqual(values['drag'], expected, places=12)
            self.assertAlmostEqual(values['lift'], 2 * expected, places=12)


if __name__ == '__main__':
    unittest.main()
