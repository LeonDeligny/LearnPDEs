"""Exercise the CLI through real training, checkpoint capture, and GIF encoding."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from examples import generate_animations, train_pinn
from learnpdes.utils.interactive import InteractivePlot
from learnpdes.utils.visualization import ModelEvaluator, visualization_grid


def render_test_frames(plotter, folder):
    """Replace only browser rendering; keep the real checkpoint/encoding pipeline."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(plotter.checkpoints):
        Image.new('RGB', (20, 20), (index * 40, 0, 0)).save(
            folder / f'epoch_{frame["step"]}.png'
        )


class TestTrainingExport(unittest.TestCase):
    def test_default_training_gif_contains_initial_and_final_updates(self):
        model, objective, analytical = train_pinn.build_problem('poiseuille', 3)
        plotter = InteractivePlot('poiseuille')
        with (
            tempfile.TemporaryDirectory() as directory,
            contextlib.chdir(directory),
            contextlib.redirect_stdout(io.StringIO()) as output,
            patch('learnpdes.utils.plot.require_gif_export'),
            patch('learnpdes.model.trainer.require_gif_export'),
            patch('learnpdes.training.require_gif_export'),
            patch(
                'learnpdes.training.build_problem',
                return_value=(model, objective, analytical),
            ),
            patch('learnpdes.training.get_plot_func', return_value=plotter),
            patch.object(InteractivePlot, 'write_frames', render_test_frames),
            patch(
                'sys.argv',
                [
                    'train_pinn',
                    'poiseuille',
                    '--epochs',
                    '3',
                    '--points',
                    '3',
                    '--resolution',
                    '5',
                    '--max-frames',
                    '2',
                ],
            ),
        ):
            train_pinn.main()
            run_dir = next(
                Path('assets/runs/poiseuille').glob('*/training.html')
            ).parent
            self.assertTrue((run_dir / 'training.html').is_file())
            self.assertTrue((run_dir / 'plotly.min.js').is_file())
            self.assertEqual(
                sorted(p.name for p in (run_dir / 'frames').glob('*.png')),
                ['epoch_0.png', 'epoch_3.png'],
            )
            with Image.open(run_dir / 'training.gif') as gif:
                self.assertEqual(gif.n_frames, 2)
                self.assertEqual(gif.info['loop'], 0)
                gif.seek(1)
                self.assertEqual(gif.info['duration'], 2000)
            self.assertIn(str(run_dir / 'training.gif'), output.getvalue())
            self.assertIn('Final RMSE:', output.getvalue())
        self.assertEqual([frame['step'] for frame in plotter.checkpoints], [0, 3])
        expected = ModelEvaluator(
            model,
            'poiseuille',
            visualization_grid('poiseuille', objective.input_space, 5),
        )()
        np.testing.assert_allclose(plotter.checkpoints[-1]['prediction'], expected['f'])
        final_loss, *_ = objective.get_loss('poiseuille')()
        self.assertAlmostEqual(plotter.checkpoints[-1]['loss'], final_loss.item())

    def test_no_gif_exports_html_without_static_dependencies(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            contextlib.redirect_stdout(io.StringIO()),
            patch('learnpdes.utils.plot.require_gif_export') as dependencies,
            patch.object(InteractivePlot, 'write_frames') as render,
            patch(
                'sys.argv',
                [
                    'train_pinn',
                    'poiseuille',
                    '--epochs',
                    '1',
                    '--points',
                    '3',
                    '--resolution',
                    '3',
                    '--no-gif',
                    '--output-dir',
                    directory,
                ],
            ),
        ):
            train_pinn.main()
            dependencies.assert_not_called()
            render.assert_not_called()
            run_dir = next(
                Path(directory).glob('runs/poiseuille/*/training.html')
            ).parent
            self.assertTrue((run_dir / 'training.html').is_file())
            self.assertFalse((run_dir / 'training.gif').exists())

    def test_missing_exporter_fails_before_training(self):
        with (
            contextlib.redirect_stderr(io.StringIO()) as error,
            patch('learnpdes.utils.plot.shutil.which', return_value=None),
            patch('learnpdes.training.build_problem') as build,
            patch('sys.argv', ['train_pinn', 'poiseuille']),
        ):
            with self.assertRaises(SystemExit) as stopped:
                train_pinn.main()
            self.assertEqual(stopped.exception.code, 2)
            build.assert_not_called()
            self.assertIn('Install FFmpeg', error.getvalue())
            self.assertIn('--no-gif', error.getvalue())

    def test_animation_generator_supports_poiseuille(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            contextlib.chdir(directory),
            contextlib.redirect_stdout(io.StringIO()),
            patch('learnpdes.utils.plot.require_gif_export'),
            patch('learnpdes.model.trainer.require_gif_export'),
            patch('learnpdes.training.require_gif_export'),
            patch.object(InteractivePlot, 'write_frames', render_test_frames),
            patch(
                'sys.argv',
                [
                    'generate_animations',
                    '--scenario',
                    'poiseuille',
                    '--epochs',
                    '2',
                    '--max-frames',
                    '2',
                ],
            ),
        ):
            generate_animations.main()
            run_dir = next(Path('assets/runs/poiseuille').glob('*/run.json')).parent
            result = json.loads((run_dir / 'run.json').read_text())
            self.assertEqual(result['export']['checkpoint_steps'], [0, 2])
            self.assertEqual(result['status'], 'completed')
            self.assertIn('rmse', result['validation'])
            self.assertTrue((run_dir / 'training.html').is_file())
            with Image.open(run_dir / 'training.gif') as gif:
                self.assertEqual(gif.n_frames, 2)


if __name__ == '__main__':
    unittest.main()
