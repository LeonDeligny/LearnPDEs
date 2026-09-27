"""Opt-in quantitative training regressions (several minutes on a CPU).

LEARNPDES_FLUID_ACCURACY=1 uv run python -m unittest tests.test_fluid_convergence -v
"""

import os
import tempfile
import unittest

import torch

from learnpdes.fluid import build_fluid_problem
from learnpdes.fluid_evaluation import evaluate_fluid
from learnpdes.model.trainer import Trainer


@unittest.skipUnless(
    os.environ.get('LEARNPDES_FLUID_ACCURACY') == '1',
    'Set LEARNPDES_FLUID_ACCURACY=1 for full training checks',
)
class TestFluidConvergence(unittest.TestCase):
    def train_case(self, scenario, points, adam, lbfgs):
        torch.manual_seed(0)
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            model, objective, _ = build_fluid_problem(scenario, points)
            with tempfile.TemporaryDirectory() as folder:
                trainer = Trainer(
                    model.parameters,
                    objective.loss,
                    {'learning_rate': 0.001, 'epochs': adam, 'lbfgs_steps': lbfgs},
                    {
                        'plot_func': lambda *args, **kwargs: None,
                        'output_dir': folder,
                        'max_frames': 6,
                    },
                    objective=objective,
                )
                trainer.train()
            # A new seed and more unseen points than the CLI's default evaluator.
            return evaluate_fluid(model, objective.problem, count=4096, seed=2027)
        finally:
            torch.set_num_threads(previous_threads)

    def test_kovasznay_matches_exact_interior_solution(self):
        metrics = self.train_case('kovasznay', 31, 1500, 1500)
        for name in ('u', 'v', 'p'):
            self.assertLess(metrics[f'{name}_relative_l2'], 0.01, metrics)
        for name in ('continuity', 'momentum_u', 'momentum_v'):
            self.assertLess(metrics[f'{name}_rms'], 0.02, metrics)

    def test_cylinder_satisfies_physics_diagnostics_without_reference_data(self):
        metrics = self.train_case('cylinder', 45, 3000, 3000)
        # These checks do not establish force, pressure, or full-field accuracy.
        # Published DFG numerical reference values are prohibited simulation data.
        self.assertLess(metrics['outlet_flow_relative_error'], 0.01, metrics)
        for name in ('continuity', 'momentum_u', 'momentum_v'):
            self.assertLess(metrics[f'{name}_rms'], 0.03, metrics)


if __name__ == '__main__':
    unittest.main()
