"""Verify the viscous benchmark independently of a trained approximation."""

import unittest
from unittest.mock import patch

import numpy as np
import torch

from examples.train_pinn import build_problem, evaluate
from learnpdes import POISEUILLE_SCENARIO, device, poiseuille
from learnpdes.model.loss import Loss
from learnpdes.utils.interactive import InteractivePlot
from learnpdes.utils.loadscenarios import load_scenario
from learnpdes.utils.visualization import ModelEvaluator, visualization_grid


class ExactFlow(torch.nn.Module):
    """Independent polynomial for the default benchmark, with perturbations."""

    def __init__(self, *, slip=0.0, pressure_shift=0.0, curvature=1.0, reverse=False):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.slip = slip
        self.pressure_shift = pressure_shift
        self.curvature = curvature
        self.reverse = reverse

    def forward(self, xy):
        x, y = xy[:, :1], xy[:, 1:2]
        u = self.curvature * 4 * y * (1 - y) + self.slip
        v = torch.zeros_like(u)
        p = 0.8 * (x if self.reverse else 4 - x) + self.pressure_shift
        return self.scale * torch.cat((u, v, p), dim=1)


class TestPoiseuille(unittest.TestCase):
    def make_loss(self, model):
        coordinates, masks, *_ = load_scenario(POISEUILLE_SCENARIO, num_inputs=7)
        model = model.to(device)
        return Loss(POISEUILLE_SCENARIO, coordinates, 2, model.forward, masks)

    def test_channel_masks_and_exact_solution(self):
        coordinates, masks, outputs, analytical, *_ = load_scenario(
            POISEUILLE_SCENARIO, num_inputs=7
        )
        self.assertEqual(outputs, 3)
        self.assertEqual(coordinates.shape, (49, 2))
        self.assertEqual(masks['inlet'].sum().item(), 7)
        self.assertEqual(masks['outlet'].sum().item(), 7)
        self.assertEqual(masks['wall'].sum().item(), 14)
        u, v, p = analytical(*coordinates.numpy().T)
        np.testing.assert_array_equal(u[masks['wall']], 0)
        np.testing.assert_array_equal(v, 0)
        np.testing.assert_allclose(p[masks['inlet']], 3.2)
        np.testing.assert_array_equal(p[masks['outlet']], 0)
        center_u, _, center_p = analytical(2.0, 0.5)
        self.assertAlmostEqual(float(center_u), 1.0)
        self.assertAlmostEqual(float(center_p), 1.6)
        np.testing.assert_allclose(u.reshape(7, 7), u.reshape(7, 7)[:, ::-1], atol=1e-6)
        with self.assertRaisesRegex(ValueError, 'at least 3'):
            load_scenario(POISEUILLE_SCENARIO, num_inputs=2)

    def test_exact_flow_has_zero_residual_and_boundary_loss(self):
        objective = self.make_loss(ExactFlow())
        total, coordinates, fields, geometry = objective.get_loss(POISEUILLE_SCENARIO)()
        self.assertLess(total.item(), 1e-12)
        self.assertIsNone(geometry)
        np.testing.assert_allclose(
            np.array([field.detach().cpu().numpy().ravel() for field in fields]),
            poiseuille.analytical(*coordinates.detach().cpu().numpy().T),
            atol=1e-6,
        )

    def test_wall_slip_pressure_gauge_and_wrong_forcing_are_penalized(self):
        # Constant slip and pressure shifts leave the PDE unchanged; only the
        # respective wall/pressure boundary conditions can detect them.
        for changes, expected in (
            ({'slip': 0.5}, 0.25),
            ({'pressure_shift': 0.5}, 0.5),
            ({'curvature': 0.0}, 3 * 0.8**2),
        ):
            with self.subTest(changes=changes):
                total, *_ = self.make_loss(ExactFlow(**changes)).poiseuille_loss()
                self.assertAlmostEqual(total.item(), expected, places=5)
        total, *_ = self.make_loss(ExactFlow(reverse=True)).poiseuille_loss()
        self.assertGreater(total.item(), 1.0)

    def test_affine_and_constant_fields_have_valid_second_derivatives(self):
        def affine(xy):
            return torch.cat((xy[:, :1], xy[:, 1:2], torch.ones_like(xy[:, :1])), dim=1)

        coordinates, masks, *_ = load_scenario(POISEUILLE_SCENARIO, num_inputs=3)
        objective = Loss(POISEUILLE_SCENARIO, coordinates, 2, affine, masks)
        total, *_ = objective.poiseuille_loss()
        self.assertTrue(torch.isfinite(total))
        self.assertGreater(total.item(), 0)

    def test_both_momentum_equations_and_continuity(self):
        # This field has nonzero convection, diffusion, pressure gradients,
        # and divergence, unlike the exact unidirectional Poiseuille solution.
        def polynomial(xy):
            x, y = xy[:, :1], xy[:, 1:2]
            return torch.cat((x**2 + y**2, x * y, x + y), dim=1)

        coordinates, masks, *_ = load_scenario(POISEUILLE_SCENARIO, num_inputs=5)
        objective = Loss(POISEUILLE_SCENARIO, coordinates, 2, polynomial, masks)
        x, y = coordinates.unbind(dim=1)
        expected = sum(
            residual.square().mean()
            for residual in (
                3 * x,
                2 * x**3 + 4 * x * y**2 + 0.6,
                2 * x**2 * y + y**3 + 1,
            )
        )
        with patch.object(objective, 'process', wraps=objective.process) as process:
            objective.poiseuille_loss()
        physics, _ = process.call_args.args
        torch.testing.assert_close(physics.cpu(), expected)

    def test_optimizer_updates_all_three_output_fields(self):
        torch.manual_seed(0)
        model, objective, _ = build_problem(POISEUILLE_SCENARIO, 7)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        before = model.network[-1].weight.detach().clone()
        losses = []
        for _ in range(3):
            optimizer.zero_grad()
            total, *_ = objective.poiseuille_loss()
            losses.append(total.item())
            total.backward(retain_graph=True)
            for parameter in model.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
            optimizer.step()
        self.assertTrue(np.isfinite(losses).all())
        for old, new in zip(before, model.network[-1].weight):
            self.assertFalse(torch.equal(old, new))

    def test_evaluation_uses_primitive_fields_on_separate_channel_grid(self):
        model = ExactFlow().to(device)
        coordinates, *_ = load_scenario(POISEUILLE_SCENARIO, num_inputs=7)
        grid = visualization_grid(POISEUILLE_SCENARIO, coordinates, resolution=9)
        self.assertEqual(grid.coordinates.shape, (81, 2))
        result = ModelEvaluator(model, POISEUILLE_SCENARIO, grid, batch_size=10)()
        np.testing.assert_allclose(
            result['f'], poiseuille.analytical(*grid.coordinates.T), atol=1e-6
        )
        self.assertTrue(model.training)
        self.assertIsNone(model.scale.grad)
        metrics = evaluate(model, POISEUILLE_SCENARIO, poiseuille.analytical)
        for value in metrics.values():
            self.assertLess(value, 1e-6)

    def test_plot_reference_errors_and_fixed_scales(self):
        coordinates, *_ = load_scenario(POISEUILLE_SCENARIO, num_inputs=3)
        xy = coordinates.numpy()
        reference = np.asarray(poiseuille.analytical(*xy.T))
        plotter = InteractivePlot(POISEUILLE_SCENARIO)
        for step, offset in ((0, -0.5), (2, 1.5)):
            plotter(
                '.',
                epoch=step,
                inputs=xy,
                f=reference + offset,
                loss=1.0,
                analytical=poiseuille.analytical,
                loss_history=[(0, 1.0), (2, 1.0)],
            )
        figure = plotter.figure()
        buttons = figure.layout.updatemenus[1].buttons
        self.assertEqual(
            [button.label for button in buttons],
            ['Prediction', 'Reference', 'Signed error'],
        )
        for index, ref_index, error_index in zip(range(3), (5, 7, 9), (6, 8, 10)):
            prediction, exact, error = (
                figure.data[i] for i in (index, ref_index, error_index)
            )
            np.testing.assert_allclose(
                np.asarray(exact.z).ravel(),
                reference[index][np.lexsort((xy[:, 0], xy[:, 1]))],
            )
            self.assertEqual(
                (prediction.zmin, prediction.zmax), (exact.zmin, exact.zmax)
            )
            self.assertEqual(error.zmin, -error.zmax)
            self.assertGreaterEqual(error.zmax, 1.5)
            for frame, offset in zip(figure.frames, (-0.5, 1.5)):
                delta = frame.data[list(frame.traces).index(error_index)]
                np.testing.assert_allclose(delta.z, offset, atol=1e-6)
                self.assertIsNone(delta.visible)
        for button in buttons:
            self.assertEqual(sum(button.args[0]['visible']), 5)


if __name__ == '__main__':
    unittest.main()
