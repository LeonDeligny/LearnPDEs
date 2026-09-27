"""GIF paths stay literal and failures preserve existing artifacts."""

import contextlib
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

from PIL import Image

from learnpdes.utils import plot


class TestGIFPaths(unittest.TestCase):
    def test_relative_paths_with_metacharacters_are_literal_local_paths(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            contextlib.chdir(directory),
        ):
            folder = Path('-frames: with spaces; $(touch INJECTED)')
            folder.mkdir()
            for step, color in ((10, 'blue'), (2, 'red')):
                Image.new('RGB', (20, 20), color).save(folder / f'epoch_{step}.png')
            output = folder / 'output; $(touch INJECTED).gif'
            plot.create_gif(output, folder, duration_ms=120, final_hold_ms=2400)
            with Image.open(output) as gif:
                self.assertEqual(cast(Any, gif).n_frames, 2)
                self.assertEqual(gif.info['duration'], 120)
                self.assertEqual(gif.convert('RGB').getpixel((0, 0)), (255, 0, 0))
                gif.seek(1)
                self.assertEqual(gif.info['duration'], 2400)
                self.assertEqual(gif.convert('RGB').getpixel((0, 0)), (0, 0, 255))
            self.assertFalse(Path('INJECTED').exists())
            self.assertEqual(len(list(folder.iterdir())), 3)

    def test_encoder_failure_preserves_frames_and_previous_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            frame = folder / 'epoch_1.png'
            Image.new('RGB', (20, 20), 'red').save(frame)
            output = folder / 'training.gif'
            output.write_bytes(b'previous animation')
            with patch(
                'learnpdes.utils.plot.subprocess.run',
                side_effect=plot.subprocess.CalledProcessError(
                    1, 'ffmpeg', stderr='encoding failed'
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, 'PNG frames retained'):
                    plot.create_gif(output, folder)
            self.assertEqual(output.read_bytes(), b'previous animation')
            self.assertEqual(set(folder.iterdir()), {frame, output})


if __name__ == '__main__':
    unittest.main()
