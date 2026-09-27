"""Verify saved curves are synchronized at actual recorded optimizer steps."""

import copy
import json
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np

from learnpdes.utils.cosinus_comparison import comparison_data, write_comparison


def recorded_runs():
    x = np.linspace(-3 * np.pi, 3 * np.pi, 13)
    return [
        {
            'order': order,
            'seed': seed,
            'coordinates': x.tolist(),
            'history': [
                {
                    'step': step,
                    'prediction': (
                        np.cos(x) + order / 100 + seed + step / 1000
                    ).tolist(),
                }
                for step in (0, 5, 10)
            ],
        }
        for seed in (0, 1)
        for order in (2, 4, 6, 8, 10, 12)
    ]


class TestCosinusComparison(unittest.TestCase):
    def test_every_order_and_seed_keeps_its_actual_checkpoint_curve(self):
        runs = recorded_runs()
        payload = comparison_data(runs)
        self.assertEqual(payload['orders'], [2, 4, 6, 8, 10, 12])
        self.assertEqual(payload['seeds'], [0, 1])
        self.assertEqual(payload['steps'], [0, 5, 10])
        self.assertEqual(payload['evaluation_pi'], 3)
        self.assertEqual(len(payload['regions']), 6)
        self.assertIn('far_outside', payload['regions'])
        for original, result in zip(runs, payload['runs']):
            for frame, rendered in zip(original['history'], result['frames']):
                self.assertEqual(rendered['prediction'], frame['prediction'])
                expected = (
                    original['order'] / 100 + original['seed'] + frame['step'] / 1000
                ) ** 2
                for mse in rendered['mse'].values():
                    self.assertAlmostEqual(mse, expected)

    def test_unequal_recording_frequencies_use_shared_steps_without_interpolation(self):
        runs = recorded_runs()[:2]
        del runs[0]['history'][1]
        payload = comparison_data(runs)
        self.assertEqual(payload['steps'], [0, 10])
        self.assertEqual(
            payload['runs'][1]['frames'][1]['prediction'],
            runs[1]['history'][2]['prediction'],
        )

    def test_legacy_viewer_keeps_original_metrics_without_fabricating_far_samples(self):
        runs = recorded_runs()[:1]
        x = np.linspace(-2 * np.pi, 2 * np.pi, 9)
        runs[0]['coordinates'] = x.tolist()
        for frame in runs[0]['history']:
            frame['prediction'] = np.cos(x).tolist()
        payload = comparison_data(runs)
        self.assertEqual(payload['evaluation_pi'], 2)
        self.assertEqual(set(payload['regions']), {'inside', 'outside', 'full'})

    def test_rejects_unusable_or_misaligned_recordings(self):
        cases = []
        runs = recorded_runs()[:2]
        runs[1]['coordinates'][0] += 0.1
        cases.append((runs, 'same evaluation'))
        runs = recorded_runs()[:2]
        del runs[0]['history'][0]['prediction']
        cases.append((runs, 'Checkpoint predictions are missing'))
        runs = recorded_runs()[:2]
        runs[0]['history'][0]['prediction'][0] = float('nan')
        cases.append((runs, 'must be finite'))
        runs = recorded_runs()[:2]
        runs[0]['history'][0]['prediction'].pop()
        cases.append((runs, 'match the grid'))
        runs = recorded_runs()[:2]
        runs[0]['history'][1]['step'] = 0
        cases.append((runs, 'strictly increasing'))
        runs = recorded_runs()[:2]
        runs[1] = copy.deepcopy(runs[0])
        cases.append((runs, 'unique run'))
        for runs, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                comparison_data(runs)

    def test_export_is_offline_and_preserves_small_and_zero_errors(self):
        runs = recorded_runs()
        runs[0]['history'][0]['prediction'] = np.cos(runs[0]['coordinates']).tolist()
        with tempfile.TemporaryDirectory() as folder:
            path = write_comparison(runs, Path(folder) / 'comparison.html')
            html = path.read_text()
            payload = json.loads(
                re.search(
                    r'<script id="comparison-data" type="application/json">(.*?)</script>',
                    html,
                ).group(1)
            )
            self.assertEqual(payload['runs'][0]['frames'][0]['mse']['full'], 0)
            self.assertIn('Show all orders', html)
            self.assertIn('id="step"', html)
            self.assertNotIn('__COMPARISON_DATA__', html)
            self.assertTrue((path.parent / 'plotly.min.js').is_file())


if __name__ == '__main__':
    unittest.main()
