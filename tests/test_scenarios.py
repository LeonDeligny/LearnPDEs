"""Every registered scenario completes through the public batch command."""

import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from learnpdes.cli import main
from learnpdes.scenarios import SCENARIOS


class TestScenarios(unittest.TestCase):
    def test_all_scenarios_export_html_without_static_encoder(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            contextlib.chdir(folder),
            contextlib.redirect_stdout(io.StringIO()),
            patch('learnpdes.model.trainer.create_gif') as encode,
            patch('learnpdes.utils.plot.require_gif_export') as dependencies,
        ):
            main(
                [
                    'train',
                    'all',
                    '--epochs',
                    '1',
                    '--points',
                    '3',
                    '--resolution',
                    '3',
                    '--max-frames',
                    '2',
                    '--no-gif',
                    '--output-dir',
                    folder,
                ]
            )
            encode.assert_not_called()
            dependencies.assert_not_called()
            manifests = list(Path(folder).glob('runs/*/*/run.json'))
            self.assertEqual(len(manifests), len(SCENARIOS))
            self.assertEqual({p.parent.parent.name for p in manifests}, set(SCENARIOS))
            for path in manifests:
                with self.subTest(scenario=path.parent.parent.name):
                    manifest = json.loads(path.read_text())
                    self.assertEqual(manifest['status'], 'completed')
                    self.assertEqual(manifest['completed_steps'], 1)
                    self.assertTrue(math.isfinite(manifest['final_loss']))
                    self.assertEqual(manifest['export']['checkpoint_steps'], [0, 1])
                    output = path.parent / 'training.html'
                    self.assertIn('Plotly.newPlot', output.read_text())
                    self.assertTrue((path.parent / 'plotly.min.js').is_file())
                    self.assertTrue((path.parent / 'loss.csv').is_file())
                    self.assertTrue((path.parent / 'model.pt').is_file())
                    self.assertFalse((path.parent / 'training.gif').exists())


if __name__ == '__main__':
    unittest.main()
