"""Run and export a matched comparison of cosine derivative orders."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from learnpdes.cli import app
from learnpdes.commands import compare_cosinus
from learnpdes.commands.compare_cosinus import run, write_results
from learnpdes.scenarios import cosinus


class TestComparison(unittest.TestCase):
    def test_matched_runs_train_and_export_reproducible_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runs = [run(order, 7, 2, 8, directory, max_frames=3) for order in (2, 4, 6)]
            self.assertEqual(len({item['history'][0]['full_mse'] for item in runs}), 1)
            for result in runs:
                final = result['history'][-1]
                self.assertEqual(final['step'], 2)
                self.assertNotEqual(final['full_mse'], result['history'][0]['full_mse'])
                for region, mse in cosinus.region_mse(
                    result['coordinates'], result['final_prediction']
                ).items():
                    self.assertAlmostEqual(final[f'{region}_mse'], mse)
                self.assertTrue((Path(directory) / result['html']).is_file())
            write_results(runs, directory, points=8)
            payload = json.loads((Path(directory) / 'results.json').read_text())
            self.assertEqual(len(payload['runs']), 3)
            self.assertEqual(
                len((Path(directory) / 'summary.csv').read_text().splitlines()), 4
            )
            self.assertIn(
                'comparison-data', (Path(directory) / 'comparison.html').read_text()
            )

    def test_comparison_multiple_values_and_viewer_rebuild(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result = runner.invoke(
                app,
                [
                    'compare-cosinus',
                    '--orders',
                    '2',
                    '--orders',
                    '4',
                    '--seeds',
                    '0',
                    '--seeds',
                    '1',
                    '--epochs',
                    '1',
                    '--points',
                    '3',
                    '--max-frames',
                    '2',
                    '--output-dir',
                    folder,
                ],
            )
            self.assertEqual(result.exit_code, 0, result.output)
            results = next(root.glob('comparisons/cosinus_derivatives/*/results.json'))
            runs = json.loads(results.read_text())['runs']
            self.assertEqual(
                [(run['order'], run['seed']) for run in runs],
                [(2, 0), (4, 0), (2, 1), (4, 1)],
            )
            manifest = json.loads((results.parent / 'run.json').read_text())
            self.assertEqual(manifest['status'], 'completed')
            with patch.object(compare_cosinus, 'run') as train:
                rebuilt = runner.invoke(
                    app,
                    [
                        'compare-cosinus',
                        '--results',
                        str(results),
                        '--output-html',
                        str(root / 'view.html'),
                    ],
                )
                self.assertEqual(rebuilt.exit_code, 0, rebuilt.output)
                train.assert_not_called()
            self.assertIn('plotly', (root / 'view.html').read_text())


if __name__ == '__main__':
    unittest.main()
