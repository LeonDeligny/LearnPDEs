"""Verify the coupled Navier–Stokes benchmark independently of convergence."""

import unittest
from unittest.mock import patch

import numpy as np
import torch

from examples.train_pinn import build_problem, evaluate
from learnpdes import KOVASZNAY_SCENARIO, device, kovasznay
from learnpdes.model.loss import Loss
from learnpdes.utils.loadscenarios import load_scenario
from learnpdes.utils.plot import get_plot_func
from learnpdes.utils.visualization import ModelEvaluator, visualization_grid


def exact_fields(xy):
    return torch.stack(kovasznay.analytical(*xy.unbind(dim=1)), dim=1)


class ExactFlow(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, xy):
        return self.scale * exact_fields(xy)


class TestKovasznay(unittest.TestCase):
    def objective(self, forward=exact_fields, points=9):
        xy, masks, *_ = load_scenario(KOVASZNAY_SCENARIO, points)
        # Double precision on CPU checks cancellation independently of MPS.
        with patch.object(Loss, 'device', torch.device('cpu')):
            return Loss(KOVASZNAY_SCENARIO, xy.double(), 2, forward, masks)

    def test_rectangle_boundaries_and_three_outputs(self):
        xy, masks, outputs, analytical, *_ = load_scenario(KOVASZNAY_SCENARIO, 9)
        self.assertEqual(xy.shape, (81, 2))
        self.assertEqual(outputs, 3)
        self.assertEqual(set(masks), {'inlet', 'outlet', 'bottom', 'top'})
        torch.testing.assert_close(xy.min(dim=0).values, torch.tensor([-0.5, -0.5]))
        torch.testing.assert_close(xy.max(dim=0).values, torch.tensor([1.0, 1.5]))
        for mask in masks.values():
            self.assertEqual(mask.dtype, torch.bool)
            self.assertEqual(mask.sum().item(), 9)
        boundary = torch.stack(list(masks.values())).any(dim=0)
        self.assertEqual((~boundary).sum().item(), 49)
        reference = np.column_stack(analytical(*xy.double().numpy().T))
        np.testing.assert_allclose(
            reference, exact_fields(xy.double()).numpy(), atol=1e-14
        )
        for points in (0, 1, 2):
            with self.assertRaisesRegex(ValueError, 'at least 3'):
                load_scenario(KOVASZNAY_SCENARIO, points)

    def test_exact_solution_satisfies_each_equation_and_all_conditions(self):
        objective = self.objective()
        for residual in objective.kovasznay_residuals(exact_fields(objective.inputs)):
            self.assertLess(residual.abs().max().item(), 1e-12)
        total, inputs, fields, geometry = objective.get_loss(KOVASZNAY_SCENARIO)()
        self.assertLess(total.item(), 1e-24)
        self.assertEqual(inputs.shape, (81, 2))
        self.assertEqual(len(fields), 3)
        self.assertIsNone(geometry)
        self.assertEqual(objective.pressure_mask.sum().item(), 1)
        self.assertFalse(objective.boundary_velocity.requires_grad)
        self.assertFalse(objective.pressure_target.requires_grad)

    def test_residuals_include_nonlinear_advection_pressure_and_viscosity(self):
        def polynomial(xy):
            x, y = xy.unbind(dim=1)
            return torch.stack((x**2 + y**2, x**2 + 3 * y**2, x**3 + x * y**2), dim=1)

        objective = self.objective(polynomial)
        x, y = objective.inputs.split(1, dim=1)
        expected = (
            2 * x**3
            + 2 * x * y**2
            + 2 * x**2 * y
            + 6 * y**3
            + 3 * x**2
            + y**2
            - 4 / 40,
            2 * x**3 + 2 * x * y**2 + 6 * x**2 * y + 18 * y**3 + 2 * x * y - 8 / 40,
            2 * x + 6 * y,
        )
        for actual, reference in zip(
            objective.kovasznay_residuals(polynomial(objective.inputs)), expected
        ):
            torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)

    def test_pressure_gauge_detects_an_otherwise_invisible_constant_shift(self):
        def shifted(xy):
            return exact_fields(xy) + xy.new_tensor([0.0, 0.0, 0.37])

        objective = self.objective(shifted)
        for residual in objective.kovasznay_residuals(shifted(objective.inputs)):
            self.assertLess(residual.abs().max().item(), 1e-12)
        total, *_ = objective.kovasznay_loss()
        self.assertAlmostEqual(total.item(), 0.37**2, places=12)

    def test_both_velocity_components_are_prescribed_on_every_edge(self):
        for edge in ('inlet', 'outlet', 'top', 'bottom'):
            for component in (0, 1):
                with self.subTest(edge=edge, component=component):
                    objective = self.objective()
                    mask = objective.mesh_masks[edge]
                    offset = (
                        torch.zeros_like(objective.inputs[:, :1]).expand(-1, 3).clone()
                    )
                    offset[mask, component] = 0.5
                    objective.forward = lambda xy: exact_fields(xy) + offset
                    # Isolate the boundary penalty from the PDE residuals.
                    with patch.object(
                        objective, 'kovasznay_residuals', return_value=(torch.zeros(1),)
                    ):
                        total, *_ = objective.kovasznay_loss()
                    expected = (
                        0.25 * mask.sum().item() / objective.boundary_mask.sum().item()
                    )
                    self.assertAlmostEqual(total.item(), expected, places=12)

    def test_repeated_optimizer_steps_train_all_three_outputs(self):
        torch.manual_seed(0)
        model, objective, analytical = build_problem(KOVASZNAY_SCENARIO, 7)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        output_layer = model.network[-1]
        before = output_layer.weight.detach().clone()
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss, *_ = objective.get_loss(KOVASZNAY_SCENARIO)()
            self.assertTrue(torch.isfinite(loss))
            loss.backward(retain_graph=True)
            self.assertTrue(torch.isfinite(output_layer.weight.grad).all())
            self.assertTrue((output_layer.weight.grad.abs().sum(dim=1) > 0).all())
            optimizer.step()
        self.assertTrue(((before - output_layer.weight).abs().sum(dim=1) > 0).all())
        metrics = evaluate(model, KOVASZNAY_SCENARIO, analytical)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        self.assertTrue(
            {'u_relative_l2', 'v_relative_l2', 'p_relative_l2'} <= metrics.keys()
        )

    def test_visualization_uses_direct_fields_on_a_separate_rectangle(self):
        xy, *_ = load_scenario(KOVASZNAY_SCENARIO, 5)
        grid = visualization_grid(KOVASZNAY_SCENARIO, xy, resolution=11)
        self.assertEqual(grid.coordinates.shape, (121, 2))
        self.assertIsNone(grid.boundary_edges)
        model = ExactFlow()
        result = ModelEvaluator(model, KOVASZNAY_SCENARIO, grid, batch_size=17)()
        self.assertTrue(model.training)
        self.assertIsNone(model.scale.grad)
        np.testing.assert_allclose(
            result['f'], kovasznay.analytical(*grid.coordinates.T), atol=1e-6
        )
        plotter = get_plot_func(KOVASZNAY_SCENARIO)
        plotter(
            '.',
            epoch=0,
            loss=0.0,
            analytical=kovasznay.analytical,
            loss_history=[(0, 0.0)],
            **result,
        )
        figure = plotter.figure()
        for trace, values in zip(figure.data[:3], result['f']):
            np.testing.assert_allclose(
                np.asarray(trace.z).ravel(),
                values[np.lexsort((grid.coordinates[:, 0], grid.coordinates[:, 1]))],
            )
        references = [
            trace for trace in figure.data if trace.name.endswith(' reference')
        ]
        errors = [
            trace for trace in figure.data if trace.name.endswith(' signed error')
        ]
        self.assertEqual(len(references), 3)
        self.assertEqual(len(errors), 3)
        for trace, values in zip(references, kovasznay.analytical(*grid.coordinates.T)):
            np.testing.assert_allclose(
                np.asarray(trace.z).ravel(),
                values[np.lexsort((grid.coordinates[:, 0], grid.coordinates[:, 1]))],
                atol=1e-6,
            )
        for trace in errors:
            np.testing.assert_allclose(trace.z, 0, atol=1e-6)

    def test_evaluation_reports_each_component_without_removing_pressure_offset(self):
        class ShiftedPressure(ExactFlow):
            def forward(self, xy):
                return super().forward(xy) + xy.new_tensor([0.0, 0.0, 0.25])

        metrics = evaluate(
            ShiftedPressure().to(device), KOVASZNAY_SCENARIO, kovasznay.analytical
        )
        self.assertLess(metrics['u_max_error'], 1e-6)
        self.assertLess(metrics['v_max_error'], 1e-6)
        self.assertAlmostEqual(metrics['p_rmse'], 0.25, places=6)
        self.assertAlmostEqual(metrics['p_max_error'], 0.25, places=6)
        self.assertAlmostEqual(metrics['rmse'], 0.25 / np.sqrt(3), places=6)


if __name__ == '__main__':
    unittest.main()
