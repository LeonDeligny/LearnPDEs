"""Separate exact-reference scenario checks for curved viscous walls."""

import unittest
from unittest.mock import patch

import numpy as np
import torch

from learnpdes.model.fluid import FluidObjective
from learnpdes.scenarios import circular_couette
from learnpdes.scenarios.circular_couette import EVALUATION, CircularCouetteProblem
from learnpdes.scenarios.registry import get_scenario
from learnpdes.training import build_problem


class TestCircularCouette(unittest.TestCase):
    def test_independent_formula_equations_walls_and_pressure_gauge(self) -> None:
        problem = CircularCouetteProblem()
        samples = problem.sample(100, 64, generator=torch.Generator().manual_seed(11))

        class Exact(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))

            def forward(self, xy: torch.Tensor) -> torch.Tensor:
                return self.scale * torch.stack(
                    circular_couette.analytical(*xy.unbind(1)), 1
                )

        model = Exact()
        xy = samples.interior.requires_grad_()
        for value in problem.residuals(xy, model(xy)).values():
            self.assertLess(value.abs().max().item(), 1e-13)
        for value in problem.boundary_residuals(model, samples.boundary).values():
            self.assertLess(value.abs().max().item(), 1e-14)
        u, v, p = circular_couette.analytical(
            torch.tensor([1.5, 2.0], dtype=torch.float64),
            torch.zeros(2, dtype=torch.float64),
        )
        self.assertEqual(u[0], 0)
        self.assertAlmostEqual(v[0].item(), 5 / 9, places=14)
        self.assertAlmostEqual(p[1].item(), 0, places=14)
        metrics = EVALUATION.evaluate(model, problem, count=128)
        self.assertLess(metrics['relative_l2'], 1e-14)

    def test_training_uses_boundary_data_only_and_enforces_walls(self) -> None:
        model, objective, analytical = build_problem('circular-couette', 5)
        assert isinstance(objective, FluidObjective)
        model.cpu().double()
        with patch(
            'learnpdes.scenarios.circular_couette.analytical',
            side_effect=AssertionError('Exact interior leaked into training'),
        ):
            objective.loss()[0].backward()
        self.assertIsNotNone(analytical)
        for name, residual in objective.problem.boundary_residuals(
            model, objective.samples.boundary
        ).items():
            if name != 'pressure':
                self.assertLess(residual.abs().max().item(), 2e-15)
        self.assertTrue(objective.problem.contains(objective.samples.interior).all())
        grid = get_scenario('circular-couette').grid(objective.input_space, 12)
        self.assertTrue(np.isfinite(grid.coordinates).all())
        radii = np.linalg.norm(grid.coordinates, axis=1)
        self.assertGreaterEqual(radii.min(), 1 - 1e-14)
        self.assertLessEqual(radii.max(), 2 + 1e-14)


if __name__ == '__main__':
    unittest.main()
