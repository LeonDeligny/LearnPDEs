"""Training entry points produce a GIF with the initial and final states."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from examples.train_pinn import main as train_cli
from learnpdes import KOVASZNAY_SCENARIO
from learnpdes.main import main


def render_test_frames(plotter, folder):
    """Exercise the real GIF encoder without requiring Chrome in unit tests."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for frame in plotter.checkpoints:
        Image.new('RGB', (20, 20), (frame['step'] * 50, 0, 0)).save(
            folder / f'epoch_{frame["step"]}.png'
        )


class TestTrainingExports(unittest.TestCase):
    def setUp(self):
        # Browser rendering is replaced below; CI only needs the real encoder.
        for target in (
            'learnpdes.model.trainer.require_gif_export',
            'learnpdes.training.require_gif_export',
            'learnpdes.utils.plot.require_gif_export',
        ):
            dependency_check = patch(target)
            dependency_check.start()
            self.addCleanup(dependency_check.stop)

    def test_main_saves_gif_by_default_with_initial_and_final_states(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with patch(
                'learnpdes.utils.interactive.InteractivePlot.write_frames',
                autospec=True,
                side_effect=render_test_frames,
            ) as render:
                output = main(
                    KOVASZNAY_SCENARIO,
                    epochs=3,
                    visualization_resolution=5,
                    output_dir=folder,
                    max_frames=3,
                )
            render.assert_called_once()
            plotter, frame_dir = render.call_args.args
            folder = output.parent
            self.assertEqual(frame_dir, folder / 'frames')
            self.assertEqual(
                [frame['step'] for frame in plotter.checkpoints], [0, 1, 3]
            )
            with Image.open(folder / 'training.gif') as gif:
                self.assertEqual(gif.n_frames, 3)
                self.assertEqual(gif.info['loop'], 0)
                gif.seek(2)
                self.assertEqual(gif.info['duration'], 2000)
                self.assertEqual(gif.convert('RGB').getpixel((0, 0)), (150, 0, 0))
            self.assertTrue((folder / 'training.html').is_file())
            self.assertEqual(len(list(frame_dir.glob('epoch_*.png'))), 3)

    def test_cli_saves_gif_by_default_and_allows_explicit_opt_out(self):
        for gif_enabled in (True, False):
            with (
                self.subTest(gif_enabled=gif_enabled),
                tempfile.TemporaryDirectory() as directory,
            ):
                folder = Path(directory)
                command = [
                    'train_pinn',
                    'kovasznay',
                    '--epochs',
                    '2',
                    '--points',
                    '5',
                    '--resolution',
                    '5',
                    '--max-frames',
                    '3',
                    '--output-dir',
                    directory,
                ]
                if not gif_enabled:
                    command.append('--no-gif')
                with (
                    patch('sys.argv', command),
                    patch(
                        'learnpdes.utils.interactive.InteractivePlot.write_frames',
                        autospec=True,
                        side_effect=render_test_frames,
                    ) as render,
                ):
                    train_cli()
                folder = next(folder.glob('runs/kovasznay/*/training.html')).parent
                self.assertEqual(render.call_count, int(gif_enabled))
                self.assertEqual((folder / 'training.gif').is_file(), gif_enabled)
                self.assertTrue((folder / 'training.html').is_file())
                if gif_enabled:
                    with Image.open(folder / 'training.gif') as gif:
                        self.assertEqual(gif.n_frames, 3)
                        gif.seek(2)
                        self.assertEqual(gif.info['duration'], 2000)


if __name__ == '__main__':
    unittest.main()
