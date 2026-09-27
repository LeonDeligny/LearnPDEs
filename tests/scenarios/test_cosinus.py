"""Check the derivative hierarchy and independent extrapolation diagnostics."""

from __future__ import annotations

import unittest
from typing import cast

import numpy as np
import torch

from learnpdes import COSINUS_SCENARIO, device
from learnpdes.scenarios import cosinus
from learnpdes.scenarios.cosinus import Objective, load_cosinus
from learnpdes.scenarios.exponential import load_exponential
from learnpdes.scenarios.registry import get_scenario
from learnpdes.types import TensorFunction


class TestCosinus(unittest.TestCase):
    @staticmethod
    def objective(forward: TensorFunction, order: int) -> Objective:
        x, masks, *_ = load_cosinus(16)
        return Objective(COSINUS_SCENARIO, x, 1, forward, masks, cosinus_order=order)

    def test_training_and_evaluation_domains_are_separate(self) -> None:
        x, masks, *_ = load_cosinus(64)
        original = x.clone()
        np.testing.assert_allclose([x.min(), x.max()], [-np.pi, np.pi])
        self.assertEqual(masks['zero'].sum(), 1)
        grid = get_scenario(COSINUS_SCENARIO).grid(x, cosinus.EVALUATION_POINTS)
        np.testing.assert_allclose(
            grid.coordinates[[0, -1], 0], [-3 * np.pi, 3 * np.pi]
        )
        self.assertEqual(len(grid.coordinates), 601)
        torch.testing.assert_close(x, original)
        exponential, *_ = load_exponential(64)
        self.assertEqual((exponential.min().item(), exponential.max().item()), (-3, 3))

    def test_exact_cosine_satisfies_all_residuals_and_anchors(self) -> None:
        for order in (2, 4, 6, 8, 10, 12):
            with self.subTest(order=order):
                objective = self.objective(torch.cos, cast(int, order))
                self.assertLess(objective.cosinus_loss()[0].item(), 1e-12)

    def test_cumulative_polynomial_residuals_and_anchor_signs(self) -> None:
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
                assert a.grad is not None
                self.assertTrue(torch.isfinite(a.grad))
                self.assertNotEqual(a.grad.item(), 0)

    def test_invalid_orders_are_rejected(self) -> None:
        for order in (0, -2, 3, 5, 2.0, True):
            with (
                self.subTest(order=order),
                self.assertRaisesRegex(ValueError, 'even integer'),
            ):
                self.objective(torch.cos, cast(int, order))

    def test_outer_error_is_not_hidden_by_inner_error(self) -> None:
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


if __name__ == '__main__':
    unittest.main()
