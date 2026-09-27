"""Run isolation, failure recovery, portable exports, and intentional publication."""

from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from learnpdes import COSINUS_SCENARIO
from learnpdes.model.trainer import Trainer
from learnpdes.utils.artifacts import TrainingRun, atomic_json, publish_run
from learnpdes.utils.interactive import InteractivePlot


class TestArtifacts(unittest.TestCase):
    def train(self, root, *, fail_export=False):
        model = torch.nn.Linear(1, 1, bias=False)
        run = TrainingRun(root, COSINUS_SCENARIO, settings={'seed': 0})
        plotter = InteractivePlot(COSINUS_SCENARIO)

        def loss():
            value = model(torch.ones(1, 1))
            return value.square().mean(), torch.zeros(1, 1), value, None

        trainer = Trainer(
            model.parameters,
            loss,
            {'learning_rate': 0.1, 'epochs': 2},
            {'plot_func': plotter, **run.plot_options(gif=fail_export)},
            analytical=np.cos,
            run=run,
            model=model,
            validation=lambda: {'rmse': 0.25},
        )
        if fail_export:

            def fail(folder):
                folder.mkdir()
                (folder / 'epoch_0.png').write_bytes(b'partial rendering')
                raise RuntimeError('Renderer failed')

            with (
                patch('learnpdes.model.trainer.require_gif_export'),
                patch.object(plotter, 'write_frames', side_effect=fail),
            ):
                with self.assertRaisesRegex(RuntimeError, 'Renderer failed'):
                    trainer.train()
        else:
            trainer.train()
        return run, model, trainer

    def test_simultaneous_runs_are_isolated_and_slug_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as folder:
            with ThreadPoolExecutor(max_workers=4) as pool:
                runs = list(
                    pool.map(
                        lambda _: TrainingRun(folder, '../potential flow'), range(8)
                    )
                )
            self.assertEqual(len({run.directory for run in runs}), 8)
            for run in runs:
                self.assertEqual(
                    run.directory.parent, Path(folder) / 'runs/potential_flow'
                )
                self.assertEqual(
                    json.loads((run.directory / 'run.json').read_text())['status'],
                    'created',
                )
            with self.assertRaises(ValueError):
                TrainingRun(folder, '..')

    def test_completed_run_contains_reloadable_weights_history_and_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            run, model, trainer = self.train(folder)
            metadata = json.loads((run.directory / 'run.json').read_text())
            self.assertEqual(metadata['status'], 'completed')
            self.assertEqual(metadata['completed_steps'], 2)
            self.assertEqual(metadata['validation'], {'rmse': 0.25})
            self.assertEqual(
                set(metadata['artifacts']),
                {'training.html', 'plotly.min.js', 'loss.csv', 'model.pt'},
            )
            with (run.directory / 'loss.csv').open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([int(row['step']) for row in rows], [0, 1, 2])
            checkpoint = torch.load(
                run.directory / 'model.pt', map_location='cpu', weights_only=True
            )
            self.assertEqual(checkpoint['completed_steps'], 2)
            restored = torch.nn.Linear(1, 1, bias=False)
            restored.load_state_dict(checkpoint['model_state_dict'])
            torch.testing.assert_close(restored.weight, model.weight)
            self.assertEqual(
                json.loads((run.directory.parent / 'latest.json').read_text())[
                    'run_id'
                ],
                run.id,
            )
            with self.assertRaisesRegex(ValueError, 'new TrainingRun'):
                trainer.train()

    def test_export_failure_preserves_training_and_previous_success(self):
        with tempfile.TemporaryDirectory() as folder:
            good, *_ = self.train(folder)
            failed, _, _ = self.train(folder, fail_export=True)
            metadata = json.loads((failed.directory / 'run.json').read_text())
            self.assertEqual(metadata['status'], 'failed')
            self.assertEqual(metadata['completed_steps'], 2)
            self.assertEqual(metadata['error']['message'], 'Renderer failed')
            for name in ('model.pt', 'loss.csv', 'training.html', 'frames/epoch_0.png'):
                self.assertTrue((failed.directory / name).is_file())
            self.assertEqual(
                json.loads((good.directory.parent / 'latest.json').read_text())[
                    'run_id'
                ],
                good.id,
            )
            with self.assertRaisesRegex(ValueError, 'completed'):
                publish_run(failed.directory, folder)

    def test_publish_requires_intent_and_rejects_corruption_without_losing_example(
        self,
    ):
        with tempfile.TemporaryDirectory() as folder:
            first, *_ = self.train(folder)
            target = publish_run(first.directory, folder)
            before = (target / 'training.html').read_bytes()
            second, *_ = self.train(folder)
            with self.assertRaises(FileExistsError):
                publish_run(second.directory, folder)
            with patch(
                'learnpdes.utils.artifacts.shutil.copy2',
                side_effect=OSError('disk full'),
            ):
                with self.assertRaises(OSError):
                    publish_run(second.directory, folder, replace=True)
            self.assertEqual((target / 'training.html').read_bytes(), before)
            (second.directory / 'training.html').write_text('corrupted')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                publish_run(second.directory, folder, replace=True)
            self.assertEqual((target / 'training.html').read_bytes(), before)
            publish_run(first.directory, folder, replace=True)
            self.assertFalse((target / 'model.pt').exists())
            self.assertTrue((target / 'plotly.min.js').is_file())
            (target / 'training.gif').write_bytes(b'existing published GIF')
            with self.assertRaisesRegex(ValueError, 'includes a GIF'):
                publish_run(first.directory, folder, replace=True)
            self.assertEqual(
                (target / 'training.gif').read_bytes(), b'existing published GIF'
            )
            self.assertEqual((target / 'training.html').read_bytes(), before)

    def test_invalid_json_never_replaces_previous_record(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'run.json'
            atomic_json(path, {'value': 1})
            with self.assertRaises(ValueError):
                atomic_json(path, {'value': float('nan')})
            self.assertEqual(json.loads(path.read_text()), {'value': 1})
            self.assertEqual(list(Path(folder).iterdir()), [path])

    def test_comparison_failure_records_status_without_publishing_latest(self):
        from examples import compare_cosinus

        with (
            tempfile.TemporaryDirectory() as folder,
            patch(
                'examples.compare_cosinus.run', side_effect=RuntimeError('run failed')
            ),
            patch('sys.argv', ['compare_cosinus', '--output-dir', folder]),
        ):
            with self.assertRaisesRegex(RuntimeError, 'run failed'):
                compare_cosinus.main()
            root = Path(folder) / 'comparisons/cosinus_derivatives'
            metadata = json.loads(next(root.glob('*/run.json')).read_text())
            self.assertEqual(metadata['status'], 'failed')
            self.assertEqual(metadata['completed_runs'], 0)
            self.assertEqual(metadata['error']['message'], 'run failed')
            self.assertFalse((root / 'latest.json').exists())


if __name__ == '__main__':
    unittest.main()
