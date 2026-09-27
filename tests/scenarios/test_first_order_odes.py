"""Verify the A1–A3 equations, initial conditions, and analytical references."""

from __future__ import annotations

import unittest
from collections.abc import Callable
from types import ModuleType
from unittest.mock import patch

import numpy as np
import torch

from learnpdes.model.objectives import CollocationObjective
from learnpdes.scenarios import exponential, forced_linear, logistic
from learnpdes.scenarios.registry import get_scenario
from learnpdes.training import build_problem, evaluate
from learnpdes.types import CollocationData, TensorFunction


def forced_solution(t: torch.Tensor) -> torch.Tensor:
    return torch.exp(-t / 5) * torch.sin(t)


CASES = (
    (exponential, exponential.load_exponential, torch.exp, (-3, 3), 1.0),
    (forced_linear, forced_linear.load_forced_linear, forced_solution, (0, 2), 0.0),
    (logistic, logistic.load_logistic, torch.sigmoid, (0, 2), 0.5),
)


class ExactModel(torch.nn.Module):
    def __init__(self, solution: TensorFunction) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.solution = solution

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.scale * self.solution(t)


class TestFirstOrderODEs(unittest.TestCase):
    @staticmethod
    def objective(
        module: ModuleType,
        loader: Callable[[int], CollocationData],
        forward: TensorFunction,
    ) -> CollocationObjective:
        coordinates, masks, *_ = loader(64)
        # Check the actual loss in float64 on CPU, including on MPS hosts.
        with (
            patch('learnpdes.model.objectives.device', torch.device('cpu')),
            patch.object(module.Objective, 'device', torch.device('cpu')),
        ):
            return module.Objective(
                module.SCENARIO.name, coordinates.double(), 1, forward, masks
            )

    def test_domains_initial_points_and_reference_values(self) -> None:
        expected = (
            (
                exponential.load_exponential,
                [-1, 0, 1],
                [0.367879441171, 1, 2.718281828459],
            ),
            (
                forced_linear.load_forced_linear,
                [0, 1, 2],
                [0, 0.688938173085, 0.609520293010],
            ),
            (logistic.load_logistic, [0, 1, 2], [0.5, 0.731058578630, 0.880797077978]),
        )
        for loader, coordinates, values in expected:
            with self.subTest(loader=loader.__name__):
                _, _, outputs, analytical, *_ = loader(64)
                self.assertEqual(outputs, 1)
                assert analytical is not None
                np.testing.assert_allclose(
                    analytical(np.array(coordinates, dtype=np.float64)),
                    values,
                    rtol=0,
                    atol=5e-13,
                )
        for module, loader, _, bounds, initial in CASES:
            for count in (3, 64):
                with self.subTest(scenario=module.SCENARIO.name, points=count):
                    coordinates, masks, _, analytical, *_ = loader(count)
                    self.assertEqual(
                        (coordinates.min().item(), coordinates.max().item()), bounds
                    )
                    self.assertTrue(masks['zero'].any())
                    self.assertTrue(torch.all(coordinates[masks['zero']] == 0))
                    assert analytical is not None
                    self.assertEqual(float(np.asarray(analytical(0.0))), initial)
                    grid = module.SCENARIO.grid(coordinates, 21)
                    np.testing.assert_array_equal(grid.coordinates[[0, -1], 0], bounds)

    def test_exact_solutions_satisfy_actual_objectives_in_float64(self) -> None:
        for module, loader, forward, _, _ in CASES:
            with self.subTest(scenario=module.SCENARIO.name):
                objective = self.objective(module, loader, forward)
                loss, *_ = objective.get_loss(module.SCENARIO.name)()
                self.assertEqual(loss.dtype, torch.float64)
                self.assertLess(loss.item(), 1e-25)

    def test_initial_condition_selects_the_right_solution(self) -> None:
        # Each candidate still solves its ODE, so only the initial value can
        # distinguish it from the intended solution.
        alternatives = (
            (exponential, exponential.load_exponential, lambda t: 2 * torch.exp(t), 1),
            (
                forced_linear,
                forced_linear.load_forced_linear,
                lambda t: torch.exp(-t / 5) * (torch.sin(t) + 1),
                1,
            ),
            (logistic, logistic.load_logistic, lambda t: t * 0, 0.25),
            (logistic, logistic.load_logistic, lambda t: t * 0 + 1, 0.25),
        )
        for module, loader, forward, expected in alternatives:
            with self.subTest(scenario=module.SCENARIO.name, forward=forward):
                loss, *_ = self.objective(module, loader, forward).loss()
                self.assertAlmostEqual(loss.item(), expected, places=12)

    def test_forcing_and_nonlinearity_are_required(self) -> None:
        # These candidates meet the initial value but violate the equation.
        for module, loader, forward in (
            (exponential, exponential.load_exponential, lambda t: t * 0 + 1),
            (forced_linear, forced_linear.load_forced_linear, lambda t: t * 0),
            (logistic, logistic.load_logistic, lambda t: torch.exp(t) / 2),
        ):
            with self.subTest(scenario=module.SCENARIO.name):
                loss, *_ = self.objective(module, loader, forward).loss()
                self.assertGreater(loss.item(), 0.1)

    def test_logistic_reference_is_bounded_and_increasing(self) -> None:
        values = logistic.analytical(np.linspace(0, 2, 1025))
        self.assertTrue(np.all((values > 0) & (values < 1)))
        self.assertTrue(np.all(np.diff(values) > 0))

    def test_registry_build_and_independent_evaluation(self) -> None:
        for module, _, solution, _, _ in CASES:
            with self.subTest(scenario=module.SCENARIO.name):
                name = module.SCENARIO.name
                self.assertIs(get_scenario(name), module.SCENARIO)
                self.assertEqual(get_scenario(name).reference_type, 'analytical')
                _, objective, analytical = build_problem(name, points=8)
                self.assertTrue(torch.isfinite(objective.loss()[0]))
                metrics = evaluate(ExactModel(solution), name, analytical)
                self.assertLess(metrics['rmse'], 3e-6)
                self.assertLess(metrics['relative_l2'], 1e-6)


if __name__ == '__main__':
    unittest.main()
