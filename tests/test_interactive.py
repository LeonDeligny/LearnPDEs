"""Check exported numerical data and integration with real optimizer updates."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from learnpdes import COSINUS_SCENARIO, LAPLACE_SCENARIO
from learnpdes.model.trainer import Trainer
from learnpdes.utils.interactive import InteractivePlot


class TestInteractive(unittest.TestCase):
    def test_laplace_orientation_and_scales_cover_every_checkpoint(self):
        xy = np.array([[1, 2], [0, 0], [1, 0], [0, 1], [0, 2], [1, 1]])

        def reference(x, y):
            return x + 10 * y

        values = reference(*xy.T)
        plotter = InteractivePlot(LAPLACE_SCENARIO)
        for step, offset in [(0, -0.1), (10, 5.0)]:
            plotter(
                '.',
                epoch=step,
                inputs=xy,
                f=values + offset,
                loss=1.0,
                analytical=reference,
                loss_history=[(0, 1.0), (step, 1.0)],
            )
        fig = plotter.figure()
        np.testing.assert_array_equal(fig.data[1].z, [[0, 1], [10, 11], [20, 21]])
        np.testing.assert_allclose(
            fig.frames[-1].data[0].z, [[5, 6], [15, 16], [25, 26]]
        )
        np.testing.assert_array_equal(fig.frames[-1].data[1].z, np.full((3, 2), 5.0))
        self.assertEqual(
            (fig.layout.coloraxis.cmin, fig.layout.coloraxis.cmax),
            (fig.layout.coloraxis3.cmin, fig.layout.coloraxis3.cmax),
        )
        self.assertLessEqual(fig.layout.coloraxis.cmin, -0.1)
        self.assertGreaterEqual(fig.layout.coloraxis.cmax, 26)
        self.assertEqual(fig.layout.coloraxis2.cmin, -fig.layout.coloraxis2.cmax)
        self.assertGreaterEqual(fig.layout.coloraxis2.cmax, 5)

    def test_final_frame_matches_training_and_exports_without_encoder(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        plotter = InteractivePlot(COSINUS_SCENARIO)

        def objective():
            value = parameter * 1.0
            return value.square().sum(), torch.zeros((1, 1)), value, None

        with tempfile.TemporaryDirectory() as folder:
            with patch('learnpdes.model.trainer.create_gif') as encode:
                trainer = Trainer(
                    lambda: iter([parameter]),
                    objective,
                    {'learning_rate': 0.1, 'epochs': 3},
                    {'plot_func': plotter, 'output_dir': folder, 'gif_path': None},
                    analytical=np.cos,
                )
                trainer.train()
            encode.assert_not_called()
            fig = plotter.figure()
            self.assertEqual([frame.name for frame in fig.frames], ['0', '1', '2', '3'])
            np.testing.assert_allclose(
                fig.frames[-1].data[0].y, parameter.detach().numpy()
            )
            self.assertAlmostEqual(fig.frames[-1].data[2].y[0], parameter.item() ** 2)
            path = Path(folder) / 'training.html'
            plotter.write_html(path)
            self.assertIn('src="plotly.min.js"', path.read_text())
            self.assertTrue((path.parent / 'plotly.min.js').is_file())
            self.assertFalse(list(Path(folder).glob('*.png')))

    def test_rejects_changed_or_incomplete_grids(self):
        plotter = InteractivePlot(LAPLACE_SCENARIO)
        frame = dict(
            epoch=0,
            inputs=np.array([[0, 0], [1, 0], [1, 1]]),
            f=np.zeros(3),
            loss=1.0,
            analytical=lambda x, y: x * 0,
            loss_history=[(0, 1.0)],
        )
        plotter('.', **frame)
        with self.assertRaisesRegex(ValueError, 'complete rectangular'):
            plotter.figure()
        with self.assertRaisesRegex(ValueError, 'same visualization grid'):
            plotter('.', **(frame | {'epoch': 1, 'inputs': frame['inputs'] + 1}))

    def test_zero_updates_exports_the_untrained_model(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        plotter = InteractivePlot(COSINUS_SCENARIO)
        with tempfile.TemporaryDirectory() as folder:
            trainer = Trainer(
                lambda: iter([parameter]),
                lambda: (
                    parameter.square().sum(),
                    torch.zeros((1, 1)),
                    parameter,
                    None,
                ),
                {'learning_rate': 0.1, 'epochs': 0},
                {'plot_func': plotter, 'output_dir': folder},
                analytical=np.cos,
            )
            trainer.train()
            self.assertTrue(trainer.html_path.is_file())
        self.assertEqual(parameter.item(), 1.0)
        self.assertEqual([frame['step'] for frame in plotter.checkpoints], [0])


if __name__ == '__main__':
    unittest.main()
