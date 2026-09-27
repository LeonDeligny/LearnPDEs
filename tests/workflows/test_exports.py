"""Training entry points produce a GIF with the initial and final states."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageSequence

from learnpdes import KOVASZNAY_SCENARIO
from learnpdes.cli import main as train_cli
from learnpdes.config import RunConfig
from learnpdes.training import train
from learnpdes.utils.interactive import InteractivePlot


def render_test_frames(plotter: InteractivePlot, folder: str | Path) -> None:
    """Exercise the real GIF encoder without requiring Chrome in workflow tests."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for frame in plotter.checkpoints:
        Image.new('RGB', (20, 20), (frame['step'] * 50, 0, 0)).save(
            folder / f'epoch_{frame["step"]}.png'
        )


class TestTrainingExports(unittest.TestCase):
    def setUp(self) -> None:
        # Browser rendering is replaced below; CI only needs the real encoder.
        for target in (
            'learnpdes.model.trainer.require_gif_export',
            'learnpdes.training.require_gif_export',
            'learnpdes.utils.plot.require_gif_export',
        ):
            dependency_check = patch(target)
            dependency_check.start()
            self.addCleanup(dependency_check.stop)

    def test_python_training_exports_gif_with_initial_and_final_states(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with patch(
                'learnpdes.utils.interactive.InteractivePlot.write_frames',
                autospec=True,
                side_effect=render_test_frames,
            ) as render:
                trainer = train(
                    RunConfig(
                        KOVASZNAY_SCENARIO,
                        epochs=3,
                        points=3,
                        hidden_dim=8,
                        hidden_layers=1,
                        resolution=5,
                        output_dir=folder,
                        max_frames=3,
                    )
                )
                output = trainer.html_path
            render.assert_called_once()
            plotter, frame_dir = render.call_args.args
            assert output is not None
            folder = output.parent
            self.assertEqual(frame_dir, folder / 'frames')
            self.assertEqual(
                [frame['step'] for frame in plotter.checkpoints], [0, 1, 3]
            )
            with Image.open(folder / 'training.gif') as gif:
                self.assertEqual(len(list(ImageSequence.Iterator(gif))), 3)
            self.assertTrue((folder / 'training.html').is_file())
            self.assertEqual(len(list(frame_dir.glob('epoch_*.png'))), 3)

    def test_cli_saves_gif_by_default_and_allows_explicit_opt_out(self) -> None:
        for gif_enabled in (True, False):
            with (
                self.subTest(gif_enabled=gif_enabled),
                tempfile.TemporaryDirectory() as directory,
            ):
                folder = Path(directory)
                command = [
                    'learnpdes',
                    'train',
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
                        self.assertEqual(len(list(ImageSequence.Iterator(gif))), 3)


if __name__ == '__main__':
    unittest.main()
