"""Kovasznay equations and boundary conditions against its exact solution."""

import unittest
from unittest.mock import patch

import torch

from learnpdes.model.fluid import FluidObjective
from learnpdes.physics.navier_stokes import navier_stokes
from learnpdes.scenarios import kovasznay
from learnpdes.scenarios.kovasznay import KovasznayProblem
from learnpdes.scenarios.registry import get_scenario
from learnpdes.training import build_problem


class TestKovasznay(unittest.TestCase):
    def test_equations_against_exact_kovasznay_and_polynomial(self) -> None:
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

    def test_exact_kovasznay_conditions_and_no_interior_reference_in_loss(self) -> None:
        class Exact(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))

            def forward(self, xy: torch.Tensor) -> torch.Tensor:
                return self.scale * torch.stack(
                    kovasznay.analytical(*xy.unbind(1)), dim=1
                )

        objective = FluidObjective(KovasznayProblem(), Exact(), 7)
        value, *_ = objective.loss()
        self.assertLess(value.item(), 1e-25)
        evaluator = get_scenario('kovasznay').fluid_evaluation
        assert evaluator is not None
        metrics = evaluator.evaluate(objective.model, objective.problem, count=100)
        self.assertLess(metrics['relative_l2'], 1e-14)
        model, objective, _ = build_problem('kovasznay', 5)
        with patch(
            'learnpdes.scenarios.kovasznay.analytical',
            side_effect=AssertionError('Interior reference leaked into training'),
        ):
            objective.loss()[0].backward()


if __name__ == '__main__':
    unittest.main()
