"""Published diagnostic plots must remain consistent and safe to replace."""

import json
import tempfile
import unittest
from pathlib import Path

from learnpdes.utils.artifacts import TrainingRun, atomic_json, publish_run


class TestPublication(unittest.TestCase):
    def make_run(self, root: str | Path, *, png: bool = True) -> TrainingRun:
        run = TrainingRun(root, 'cylinder')
        for name in ('training.html', 'plotly.min.js', 'model.pt'):
            (run.directory / name).write_bytes(name.encode())
        run.finish()
        plots = run.directory / 'plots'
        plots.mkdir()
        (plots / 'flow-fields.html').write_text('<html>fields</html>')
        (plots / 'plotly.min.js').write_text('// plot runtime')
        if png:
            (plots / 'flow-fields.png').write_bytes(b'plot-image')
        atomic_json(
            plots / 'diagnostics.json',
            {
                'run_id': run.id,
                'checkpoint_sha256': run.manifest['artifacts']['model.pt']['sha256'],
                'figures': ['flow-fields.html'],
            },
        )
        return run

    def test_mismatched_plots_cannot_replace_a_published_result(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = self.make_run(root)
            target = publish_run(first.directory, root)
            original = (target / 'run.json').read_bytes()
            second = self.make_run(root)
            diagnostics = second.directory / 'plots/diagnostics.json'
            correct = json.loads(diagnostics.read_text())
            for field in ('run_id', 'checkpoint_sha256'):
                with self.subTest(field=field):
                    atomic_json(diagnostics, {**correct, field: 'wrong-checkpoint'})
                    with self.assertRaisesRegex(
                        ValueError, 'selected run and checkpoint'
                    ):
                        publish_run(second.directory, root, replace=True)
                    self.assertEqual((target / 'run.json').read_bytes(), original)
                    self.assertEqual(
                        (target / 'plots/flow-fields.png').read_bytes(), b'plot-image'
                    )

    def test_replacement_preserves_existing_plot_links(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = self.make_run(root)
            target = publish_run(first.directory, root)
            original = (target / 'run.json').read_bytes()
            without_png = self.make_run(root, png=False)
            with self.assertRaisesRegex(ValueError, 'plots/flow-fields.png'):
                publish_run(without_png.directory, root, replace=True)
            self.assertEqual((target / 'run.json').read_bytes(), original)

            replacement = self.make_run(root)
            publish_run(replacement.directory, root, replace=True)
            published = json.loads((target / 'run.json').read_text())
            self.assertEqual(published['run_id'], replacement.id)
            self.assertIn('plots/flow-fields.png', published['artifacts'])
            self.assertFalse((target / 'model.pt').exists())


if __name__ == '__main__':
    unittest.main()
