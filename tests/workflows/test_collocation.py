"""Training workflows export the collocation points used by each optimizer."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from learnpdes.model.fluid import FluidObjective
from learnpdes.model.objectives import CollocationObjective
from learnpdes.model.pinn import PINN
from learnpdes.model.trainer import Trainer
from learnpdes.scenarios.registry import get_scenario
from learnpdes.training import build_problem
from learnpdes.utils.artifacts import TrainingRun
from learnpdes.utils.collocation import load_final_points
from learnpdes.utils.interactive import InteractivePlot
from learnpdes.visualization.fields import ModelEvaluator


class TestCollocation(unittest.TestCase):
    def train(
        self, scenario: str, folder: str | Path, *, adam: int = 2, lbfgs: int = 0
    ) -> tuple[
        PINN, CollocationObjective | FluidObjective, InteractivePlot, TrainingRun
    ]:
        torch.manual_seed(0)
        model, objective, analytical = build_problem(scenario, 5)
        plotter = InteractivePlot(scenario)
        grid = get_scenario(scenario).grid(objective.input_space, 9)
        run = TrainingRun(folder, scenario)
        trainer = Trainer(
            model.parameters,
            objective.get_loss(scenario),
            {
                'learning_rate': 0.001,
                'epochs': adam,
                'lbfgs_steps': lbfgs,
                'resample_every': 1,
            },
            {
                'plot_func': plotter,
                'evaluate': ModelEvaluator(model, scenario, grid),
                **run.plot_options(gif=False),
                'max_frames': 20,
            },
            analytical=analytical,
            run=run,
            model=model,
            objective=objective if isinstance(objective, FluidObjective) else None,
            collocation=objective.collocation_points,
        )
        trainer.train()
        return model, objective, plotter, run

    def test_fluid_coordinates_follow_adam_resampling_then_freeze_for_lbfgs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as folder:
            model, objective, plotter, run = self.train('cylinder', folder, lbfgs=2)
            recording = json.loads((run.directory / 'collocation.json').read_text())
            self.assertIn('collocation.json', run.manifest['artifacts'])
            self.assertEqual(
                [c['step'] for c in recording['checkpoints']], list(range(5))
            )
            indices = [c['point_set'] for c in recording['checkpoints']]
            self.assertNotEqual(indices[0], indices[1])
            self.assertNotEqual(indices[1], indices[2])
            self.assertEqual(indices[2:], [indices[2]] * 3)
            groups = load_final_points(run.directory)
            assert groups is not None
            assert isinstance(objective, FluidObjective)
            actual = (
                objective.samples.interior.to(next(model.parameters()))
                .detach()
                .cpu()
                .numpy()
            )
            np.testing.assert_array_equal(groups['interior']['coordinates'], actual)
            self.assertEqual(len(groups['interior']['coordinates']), 25)
            self.assertEqual(len(groups['mass_flux']['coordinates']), 12 * 64)
            self.assertIn('cylinder', groups)
            self.assertEqual(groups, plotter.checkpoints[-1]['collocation'])
            assert plotter.coordinates is not None
            self.assertNotEqual(len(plotter.coordinates), 25)

    def test_fixed_grid_keeps_boundary_roles_and_deduplicates_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            _, objective, plotter, run = self.train('laplace', folder)
            recording = json.loads((run.directory / 'collocation.json').read_text())
            self.assertEqual(len(recording['point_sets']), 1)
            groups = recording['point_sets'][0]
            np.testing.assert_array_equal(
                groups['equation']['coordinates'],
                objective.inputs.detach().cpu().numpy(),
            )
            self.assertEqual(len(groups['equation']['coordinates']), 25)
            self.assertEqual(len(groups['top']['coordinates']), 5)
            assert plotter.coordinates is not None
            self.assertEqual(len(plotter.coordinates), 81)


if __name__ == '__main__':
    unittest.main()
