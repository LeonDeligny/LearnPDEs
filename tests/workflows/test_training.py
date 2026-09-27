"""Run scenario training and the commands that consume completed runs."""

import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from learnpdes.cli import main
from learnpdes.config import RunConfig
from learnpdes.scenarios.registry import SCENARIOS
from learnpdes.training import train


class TestTraining(unittest.TestCase):
    def test_all_scenarios_export_html_without_static_encoder(self) -> None:
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

    def test_saved_fluid_run_can_be_verified_refined_plotted_and_published(
        self,
    ) -> None:
        with (
            tempfile.TemporaryDirectory() as folder,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            root = Path(folder)
            trainer = train(
                RunConfig(
                    'cylinder',
                    epochs=1,
                    points=3,
                    hidden_dim=8,
                    hidden_layers=1,
                    resolution=3,
                    max_frames=2,
                    save_gif=False,
                    output_dir=root,
                )
            )
            assert trainer.run is not None
            source = trainer.run.directory
            main(['verify', str(source), '--output-dir', str(root / 'verification')])
            index = json.loads((root / 'verification/index.json').read_text())
            report = json.loads(Path(index['scenarios'][0]['report']).read_text())
            self.assertEqual(report['run_id'], trainer.run.manifest['run_id'])
            self.assertEqual(report['reference_type'], 'none')

            main(
                [
                    'refine',
                    str(source),
                    '--points',
                    '4',
                    '--lbfgs-steps',
                    '1',
                    '--output-dir',
                    str(root / 'refined'),
                ]
            )
            refined = next((root / 'refined').glob('runs/cylinder/*/run.json'))
            manifest = json.loads(refined.read_text())
            self.assertEqual(manifest['status'], 'completed')
            self.assertEqual(manifest['completed_steps'], 1)
            self.assertTrue((refined.parent / 'verification.json').is_file())

            main(['plot', str(source), '--resolution', '9'])
            diagnostics = json.loads((source / 'plots/diagnostics.json').read_text())
            for figure in diagnostics['figures']:
                self.assertTrue((source / 'plots' / figure).is_file())

            main(['publish', str(source), '--output-dir', str(root / 'published')])
            published = root / 'published/examples/cylinder'
            self.assertTrue((published / 'training.html').is_file())
            self.assertTrue((published / 'plotly.min.js').is_file())
            self.assertEqual(
                json.loads((published / 'plots/diagnostics.json').read_text()),
                diagnostics,
            )
            for figure in diagnostics['figures']:
                self.assertEqual(
                    (published / 'plots' / figure).read_bytes(),
                    (source / 'plots' / figure).read_bytes(),
                )
            self.assertIn(
                'plots/diagnostics.json',
                json.loads((published / 'run.json').read_text())['artifacts'],
            )

    def test_overrides_reach_training_and_are_saved(self) -> None:
        with (
            tempfile.TemporaryDirectory() as folder,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            main(
                [
                    'kovasznay',
                    '--epochs',
                    '1',
                    '--lbfgs-steps',
                    '1',
                    '--points',
                    '3',
                    '--learning-rate',
                    '0.002',
                    '--hidden-dim',
                    '8',
                    '--hidden-layers',
                    '2',
                    '--seed',
                    '17',
                    '--resample-every',
                    '2',
                    '--resolution',
                    '3',
                    '--max-frames',
                    '2',
                    '--no-gif',
                    '--output-dir',
                    folder,
                ]
            )
            path = next(Path(folder).glob('runs/kovasznay/*/run.json'))
            manifest = json.loads(path.read_text())
            self.assertEqual(manifest['status'], 'completed')
            self.assertEqual(manifest['completed_steps'], 2)
            self.assertEqual(manifest['training']['learning_rate'], 0.002)
            self.assertEqual(manifest['training']['resample_every'], 2)
            self.assertEqual(manifest['settings']['model']['hidden_dim'], 8)
            self.assertEqual(manifest['settings']['model']['num_hidden_layers'], 2)
            self.assertEqual(manifest['settings']['seed'], 17)
            self.assertEqual(manifest['settings']['config']['points'], 3)
            self.assertTrue((path.parent / 'model.pt').is_file())


if __name__ == '__main__':
    unittest.main()
