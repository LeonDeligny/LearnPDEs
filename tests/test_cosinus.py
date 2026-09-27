"""Check the derivative hierarchy and independent extrapolation diagnostics."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from examples.compare_cosinus import comparison_figure, run, write_results
from examples.train_pinn import build_problem, evaluate
from learnpdes import COSINUS_SCENARIO, cosinus, device
from learnpdes.model.loss import Loss
from learnpdes.utils.interactive import InteractivePlot
from learnpdes.utils.loadscenarios import load_cosinus, load_exponential
from learnpdes.utils.visualization import visualization_grid


class TestCosinus(unittest.TestCase):
    @staticmethod
    def objective(forward, order):
        x, masks, *_ = load_cosinus(16)
        return Loss(COSINUS_SCENARIO, x, 1, forward, masks, cosinus_order=order)

    def test_training_and_evaluation_domains_are_separate(self):
        x, masks, *_ = load_cosinus(64)
        original = x.clone()
        np.testing.assert_allclose([x.min(), x.max()], [-np.pi, np.pi])
        self.assertEqual(masks['zero'].sum(), 1)
        grid = visualization_grid(COSINUS_SCENARIO, x, cosinus.EVALUATION_POINTS)
        np.testing.assert_allclose(
            grid.coordinates[[0, -1], 0], [-3 * np.pi, 3 * np.pi]
        )
        self.assertEqual(len(grid.coordinates), 601)
        torch.testing.assert_close(x, original)
        exponential, *_ = load_exponential(64)
        self.assertEqual((exponential.min().item(), exponential.max().item()), (-3, 3))

    def test_exact_cosine_satisfies_all_residuals_and_anchors(self):
        for order in (2, 4, 6, 8, 10, 12):
            with self.subTest(order=order):
                objective = self.objective(torch.cos, order)
                self.assertLess(objective.cosinus_loss()[0].item(), 1e-12)

    def test_cumulative_polynomial_residuals_and_anchor_signs(self):
        for order in (2, 4, 6):
            with self.subTest(order=order):
                # f(0)=1 and f'(0)=0; fourth derivative deliberately differs from 1.
                a = torch.nn.Parameter(torch.tensor(0.7, device=device))
                objective = self.objective(
                    lambda x: 1 - x**2 / 2 + a * x**4 / 24, order
                )
                x = objective.x
                expected = 3 * ((a - 1) * x**2 / 2 + a * x**4 / 24).square().mean()
                if order >= 4:
                    expected = (
                        expected
                        + 3 * (a - 1 + a * x**2 / 2).square().mean()
                        + (a - 1) ** 2
                    )
                if order >= 6:
                    expected = expected + 3 * a**2 + 1  # f^(6)=0; its target is -1.
                actual = objective.cosinus_loss()[0]
                torch.testing.assert_close(actual, expected)
                actual.backward(retain_graph=True)
                self.assertTrue(torch.isfinite(a.grad))
                self.assertNotEqual(a.grad.item(), 0)

    def test_invalid_orders_are_rejected(self):
        for order in (0, -2, 3, 5, 2.0, True):
            with (
                self.subTest(order=order),
                self.assertRaisesRegex(ValueError, 'even integer'),
            ):
                self.objective(torch.cos, order)

    def test_outer_error_is_not_hidden_by_inner_error(self):
        x = (np.arange(-4, 5) * np.pi).astype(np.float32)
        masks = cosinus.region_masks(x)
        np.testing.assert_array_equal(
            masks['inside'],
            [False, False, False, True, True, True, False, False, False],
        )
        np.testing.assert_array_equal(
            masks['outside'],
            [False, False, True, False, False, False, True, False, False],
        )
        np.testing.assert_array_equal(
            masks['far_outside'],
            [False, True, False, False, False, False, False, True, False],
        )
        np.testing.assert_array_equal(
            masks['outside_3pi'], masks['outside'] | masks['far_outside']
        )
        prediction = np.cos(x) + np.array([100, 3, 2, 0, 0, 0, 2, 3, 100])
        metrics = cosinus.region_mse(x, prediction)
        self.assertEqual(
            metrics,
            {
                'inside': 0.0,
                'outside': 4.0,
                'full': 1.6,
                'outside_3pi': 6.5,
                'far_outside': 9.0,
                'full_3pi': 26 / 7,
            },
        )

    def test_old_recordings_do_not_claim_far_domain_metrics(self):
        x = np.linspace(-2 * np.pi, 2 * np.pi, 9)
        self.assertEqual(
            set(cosinus.region_mse(x, np.cos(x))), {'inside', 'outside', 'full'}
        )

    def test_evaluation_does_not_change_training_state_or_gradients(self):
        model, objective, analytical = build_problem(
            COSINUS_SCENARIO, 8, cosinus_order=6
        )
        before = objective.cosinus_loss()[0].detach().clone()
        coordinates = objective.x.detach().clone()
        for training in (True, False):
            model.train(training)
            metrics = evaluate(model, COSINUS_SCENARIO, analytical)
            self.assertEqual(model.training, training)
            self.assertTrue(
                all(parameter.grad is None for parameter in model.parameters())
            )
            self.assertTrue(
                all(f'{region}_mse' in metrics for region in cosinus.REGION_LABELS)
            )
        torch.testing.assert_close(objective.cosinus_loss()[0], before)
        torch.testing.assert_close(objective.x, coordinates)

    def test_plots_show_numeric_pi_and_separate_evaluation_curves(self):
        x = np.linspace(-3 * np.pi, 3 * np.pi, 13)
        plotter = InteractivePlot(COSINUS_SCENARIO, cosinus_order=6)
        for step, offset in ((0, 1), (1, 0)):
            plotter(
                '.',
                epoch=step,
                inputs=x[:, None],
                f=np.cos(x) + offset,
                loss=offset,
                analytical=np.cos,
                loss_history=[(0, 1), (1, 0)],
            )
        fig = plotter.figure()
        self.assertIn('3.141593', fig.layout.xaxis.title.text)
        self.assertIn('π<br>3.142', fig.layout.xaxis.ticktext)
        self.assertIn('3π<br>9.425', fig.layout.xaxis.ticktext)
        mse_traces = [trace for trace in fig.data if trace.name.endswith(' MSE')]
        self.assertEqual(len(mse_traces), 6)
        for trace in mse_traces:
            self.assertEqual(fig.layout[f'yaxis{trace.yaxis[1:]}'].type, 'log')
        self.assertTrue(
            any(shape.x0 == -np.pi and shape.x1 == np.pi for shape in fig.layout.shapes)
        )
        self.assertEqual(plotter.checkpoints[-1]['evaluation_mse']['outside'], 0)
        np.testing.assert_allclose(fig.frames[-1].data[0].y, np.cos(x), atol=1e-6)
        for trace in mse_traces:
            self.assertAlmostEqual(trace.y[0], 1)
            self.assertGreater(
                trace.y[1], 0
            )  # Only the logarithmic display clamps zero.

    def test_matched_runs_train_and_export_reproducible_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            runs = [run(order, 7, 2, 8, directory, max_frames=3) for order in (2, 4, 6)]
            self.assertEqual(len({item['history'][0]['full_mse'] for item in runs}), 1)
            for result in runs:
                final = result['history'][-1]
                self.assertEqual(final['step'], 2)
                self.assertNotEqual(final['full_mse'], result['history'][0]['full_mse'])
                for region, mse in cosinus.region_mse(
                    result['coordinates'], result['final_prediction']
                ).items():
                    self.assertAlmostEqual(final[f'{region}_mse'], mse)
                self.assertTrue((Path(directory) / result['html']).is_file())
            write_results(runs, directory, points=8)
            payload = json.loads((Path(directory) / 'results.json').read_text())
            self.assertEqual(len(payload['runs']), 3)
            self.assertEqual(
                len((Path(directory) / 'summary.csv').read_text().splitlines()), 4
            )
            self.assertIn(
                'comparison-data', (Path(directory) / 'comparison.html').read_text()
            )
            self.assertEqual(comparison_figure(runs).layout.yaxis3.type, 'log')


if __name__ == '__main__':
    unittest.main()
