"""Geometry, independent physics checks, and the two-phase training contract."""

import csv
import math
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from learnpdes import kovasznay
from learnpdes.fluid import (
    CylinderProblem,
    KovasznayProblem,
    FluidObjective,
    build_fluid_problem,
    navier_stokes,
    sample_fluid,
)
from learnpdes.fluid_evaluation import cylinder_observables, evaluate_fluid
from learnpdes.model.trainer import Trainer
from learnpdes.utils.artifacts import TrainingRun
from learnpdes.utils.visualization import visualization_grid, ModelEvaluator


class TestFluid(unittest.TestCase):
    def test_equations_against_exact_kovasznay_and_polynomial(self):
        xy = torch.rand(100, 2, dtype=torch.float64, requires_grad=True)
        fields = torch.stack(kovasznay.analytical(*xy.unbind(1)), dim=1)
        for residual in navier_stokes(xy, fields, 40).values():
            self.assertLess(residual.abs().max().item(), 2e-14)
        x, y = xy.unbind(1)
        fields = torch.stack(
            (x * x + y * y, x * x + 3 * y * y, x**3 + x * y * y), dim=1
        )
        expected = {
            'continuity': 2 * x + 6 * y,
            'momentum_u': 2 * x**3
            + 2 * x * y * y
            + 2 * x * x * y
            + 6 * y**3
            + 3 * x * x
            + y * y
            - 4 / 20,
            'momentum_v': 2 * x**3
            + 2 * x * y * y
            + 6 * x * x * y
            + 18 * y**3
            + 2 * x * y
            - 8 / 20,
        }
        for name, residual in navier_stokes(xy, fields, 20).items():
            torch.testing.assert_close(
                residual[:, 0], expected[name], atol=1e-13, rtol=1e-13
            )

    def test_curved_wall_and_equations_against_exact_circular_couette_flow(self):
        # A separate exact steady NS solution around a stationary inner cylinder.
        # It validates curved-wall differentiation, not the channel's outer BCs.
        problem = CylinderProblem()
        samples = sample_fluid(
            problem, 100, 100, generator=torch.Generator().manual_seed(17)
        )

        def exact(xy):
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

    def test_samples_exclude_solid_and_boundaries_and_are_reproducible(self):
        problem = CylinderProblem()
        first = sample_fluid(
            problem, 1000, 64, generator=torch.Generator().manual_seed(8)
        )
        repeat = sample_fluid(
            problem, 1000, 64, generator=torch.Generator().manual_seed(8)
        )
        torch.testing.assert_close(first.interior, repeat.interior)
        self.assertTrue(problem.contains(first.interior).all())
        self.assertEqual(len(first.interior), 1000)
        for name, xy in first.boundary.items():
            self.assertEqual(len(xy), 64)
            torch.testing.assert_close(xy, repeat.boundary[name])
            self.assertFalse(torch.isin(first.interior[:, 0], xy[:, 0]).all())
        radii = (first.boundary['cylinder'] - torch.tensor(problem.center)).norm(dim=1)
        torch.testing.assert_close(radii, torch.full_like(radii, problem.radius))

    def test_boundary_transform_and_outlet_pressure_reference(self):
        model, objective, reference = build_fluid_problem('cylinder', 5)
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

        def shifted(xy):
            return model(xy) + xy.new_tensor([0, 0, 0.3])

        shifted_residuals = objective.problem.boundary_residuals(shifted, boundary)
        torch.testing.assert_close(
            shifted_residuals['outlet_normal'], residuals['outlet_normal'] - 0.3
        )

    def test_integral_continuity_uses_only_fluid_cross_sections(self):
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

        def uniform(points):
            return torch.stack(
                (points[:, 0] * 0 + 1, points[:, 0] * 0, points[:, 0] * 0), 1
            )

        residual = problem.conservation_residuals(uniform, xy)['mass_flux']
        self.assertAlmostEqual(residual[-1].item(), 0.0, places=12)
        self.assertAlmostEqual(residual[at_center].item(), -1 / 4.1, places=12)

    def test_exact_kovasznay_conditions_and_no_interior_reference_in_loss(self):
        class Exact(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))

            def forward(self, xy):
                return self.scale * torch.stack(
                    kovasznay.analytical(*xy.unbind(1)), dim=1
                )

        objective = FluidObjective(KovasznayProblem(), Exact(), 7)
        value, *_ = objective.loss()
        self.assertLess(value.item(), 1e-25)
        metrics = evaluate_fluid(objective.model, objective.problem, count=100)
        self.assertLess(metrics['relative_l2'], 1e-14)
        model, objective, _ = build_fluid_problem('kovasznay', 5)
        with patch(
            'learnpdes.kovasznay.analytical',
            side_effect=AssertionError('Interior reference leaked into training'),
        ):
            objective.loss()[0].backward()

    def test_force_sign_pressure_scaling_and_evaluator_restores_mode(self):
        class LinearPressure(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))

            def forward(self, xy):
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
            metrics = evaluate_fluid(model, problem, count=64)
        self.assertTrue(model.training)
        self.assertIsNone(model.scale.grad)
        self.assertAlmostEqual(metrics['momentum_u_rms'], 1.0, places=12)
        # Predictions remain available, but no comparison to a simulated
        # cylinder solution may be exported as an accuracy metric.
        for name in ('drag', 'lift', 'pressure_drop'):
            self.assertIn(name, metrics)
            self.assertNotIn(f'{name}_absolute_error', metrics)
            self.assertNotIn(f'{name}_relative_error', metrics)

    def test_fresh_graphs_fixed_lbfgs_samples_and_saved_components(self):
        torch.manual_seed(0)
        model, objective, _ = build_fluid_problem('cylinder', 5)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        previous_xy = None
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            value, xy, *_ = objective.loss()
            self.assertIsNot(xy, previous_xy)
            previous_xy = xy
            value.backward()  # Deliberately no retain_graph.
            self.assertTrue(torch.isfinite(model.network[-1].weight.grad).all())
            self.assertTrue((model.network[-1].weight.grad.abs().sum(1) > 0).all())
            optimizer.step()
        sample_ids = []
        original_loss = objective.loss

        def observed_loss():
            sample_ids.append(id(objective.samples))
            return original_loss()

        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(objective, 'resample', wraps=objective.resample) as resample,
        ):
            run = TrainingRun(folder, 'cylinder')
            trainer = Trainer(
                model.parameters,
                observed_loss,
                {
                    'learning_rate': 0.001,
                    'epochs': 3,
                    'lbfgs_steps': 3,
                    'resample_every': 1,
                },
                {
                    'plot_func': lambda *args, **kwargs: None,
                    'output_dir': folder,
                    'max_frames': 2,
                },
                objective=objective,
                model=model,
                run=run,
            )
            trainer.train()
            self.assertEqual(
                resample.call_count, 4
            )  # Three Adam batches + one fixed L-BFGS batch.
            self.assertEqual(len(set(sample_ids[3:])), 1)
            self.assertEqual(trainer.completed_steps, 6)
            self.assertEqual(len(trainer.loss_history), 7)
            self.assertEqual(trainer.component_history[-1]['phase'], 'lbfgs')
            with (run.directory / 'residuals.csv').open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 7)
            self.assertIn('momentum_u', rows[0])
            self.assertIn('residuals.csv', run.manifest['artifacts'])

    def test_force_quadrature_includes_viscosity_and_ignores_pressure_offset(self):
        class Polynomial(torch.nn.Module):
            def __init__(self, pressure):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
                self.pressure = pressure

            def forward(self, xy):
                r2 = (xy - xy.new_tensor([2.0, 2.0])).square().sum(1)
                return self.scale * torch.stack((r2, 2 * r2, r2 * 0 + self.pressure), 1)

        problem = CylinderProblem()
        expected = 8 * math.pi * problem.radius**2 / problem.reynolds
        for pressure in (0.0, 17.0):
            values = cylinder_observables(Polynomial(pressure), problem)
            self.assertAlmostEqual(values['drag'], expected, places=12)
            self.assertAlmostEqual(values['lift'], 2 * expected, places=12)

    def test_visualization_retains_cylinder_hole(self):
        model, objective, _ = build_fluid_problem('cylinder', 5)
        grid = visualization_grid('cylinder', objective.input_space, resolution=9)
        self.assertTrue(np.isfinite(grid.coordinates).all())
        centers = grid.coordinates[grid.triangles].mean(axis=1)
        self.assertTrue((np.linalg.norm(centers - (2, 2), axis=1) > 0.5).all())
        output = ModelEvaluator(model, 'cylinder', grid)()
        self.assertEqual(len(output['f']), 3)
        self.assertTrue(all(np.isfinite(value).all() for value in output['f']))

    def test_lbfgs_line_search_actually_reduces_a_stiff_objective(self):
        # One logged iteration must still allow enough closure evaluations for
        # line search. A max_eval of one silently makes zero-length steps.
        parameter = torch.nn.Parameter(torch.tensor([3.0, -2.0]))

        class Objective:
            components = {}

            def resample(self):
                pass

            def loss(self):
                value = 1000 * parameter[0].square() + parameter[1].square()
                self.components = {'quadratic': value.item()}
                return value, parameter[:, None], parameter[:, None], None

        objective = Objective()
        with tempfile.TemporaryDirectory() as folder:
            trainer = Trainer(
                lambda: [parameter],
                objective.loss,
                {'learning_rate': 0.001, 'epochs': 0, 'lbfgs_steps': 12},
                {
                    'plot_func': lambda *args, **kwargs: None,
                    'output_dir': folder,
                    'max_frames': 2,
                },
                objective=objective,
            )
            trainer.train()
        self.assertLess(trainer.loss_history[-1][1], 1e-8)


if __name__ == '__main__':
    unittest.main()
