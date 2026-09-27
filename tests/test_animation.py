"""Decode animations to verify timing, ordering, and final-state capture."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch
from PIL import Image

from learnpdes.model.trainer import Trainer, checkpoint_steps
from learnpdes.utils.plot import create_gif


class TestAnimation(unittest.TestCase):
    def test_playback_metadata_and_numeric_order(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for step, color in [(100, 'blue'), (2, 'red'), (10, 'green')]:
                Image.new('RGB', (20, 20), color).save(folder / f'epoch_{step}.png')
            output = folder / 'training.gif'
            create_gif(output, folder)
            with Image.open(output) as gif:
                self.assertEqual(gif.info['loop'], 0)
                self.assertEqual(gif.n_frames, 3)
                for index, color in enumerate([(255, 0, 0), (0, 128, 0), (0, 0, 255)]):
                    gif.seek(index)
                    self.assertEqual(gif.info['duration'], 2000 if index == 2 else 100)
                    self.assertEqual(gif.convert('RGB').getpixel((0, 0)), color)

    def test_single_frame_and_empty_input(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'animation.gif'
            with self.assertRaises(ValueError):
                create_gif(output, folder)
            Image.new('RGB', (20, 20), 'red').save(Path(folder) / 'epoch_1.png')
            create_gif(output, folder)
            with Image.open(output) as gif:
                self.assertEqual(gif.info['duration'], 2000)
                self.assertEqual(gif.info['loop'], 0)
            with self.assertRaises(ValueError):
                create_gif(output, folder, duration_ms=0.1)

    def test_schedule_is_early_heavy_and_includes_final(self):
        steps = sorted(checkpoint_steps(1057))
        self.assertEqual((steps[0], steps[-1]), (0, 1057))
        self.assertLessEqual(len(steps), 80)
        self.assertGreater(
            sum(step < 100 for step in steps), sum(step > 957 for step in steps)
        )
        self.assertEqual(checkpoint_steps(1), {0, 1})
        self.assertEqual(checkpoint_steps(0), {0})
        self.assertEqual(checkpoint_steps(100, max_frames=2), {0, 100})

    def test_final_frame_and_loss_match_completed_updates(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        frames = []

        def objective():
            value = parameter * 1.0
            return value.square().sum(), torch.zeros((1, 1)), value, None

        def plot(folder, **frame):
            frames.append(frame)

        with (
            tempfile.TemporaryDirectory() as folder,
            patch('learnpdes.model.trainer.create_gif', Mock()) as encode,
        ):
            trainer = Trainer(
                lambda: iter([parameter]),
                objective,
                {'learning_rate': 0.1, 'epochs': 3},
                {
                    'plot_func': plot,
                    'output_dir': folder,
                    'gif_path': Path(folder) / 'training.gif',
                },
            )
            trainer.train()
            encode.assert_called_once()
        self.assertEqual([frame['epoch'] for frame in frames], [0, 1, 2, 3])
        np.testing.assert_allclose(frames[-1]['f'], parameter.detach().numpy())
        self.assertAlmostEqual(frames[-1]['loss'], parameter.item() ** 2)

    def test_missing_encoder_preserves_frames_and_existing_output(self):
        with tempfile.TemporaryDirectory() as folder:
            frame = Path(folder) / 'epoch_1.png'
            output = Path(folder) / 'animation.gif'
            Image.new('RGB', (20, 20), 'red').save(frame)
            output.write_bytes(b'previous animation')
            with patch('learnpdes.utils.plot.shutil.which', return_value=None):
                with self.assertRaisesRegex(RuntimeError, 'Install FFmpeg'):
                    create_gif(output, folder)
            self.assertTrue(frame.exists())
            self.assertEqual(output.read_bytes(), b'previous animation')


if __name__ == '__main__':
    unittest.main()
