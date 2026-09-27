"""Check field orientation, comparable scales, and publication frame layout."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from learnpdes.utils.plot import (
    error_metrics,
    save_2d_plot,
    save_airfoil_plot,
    save_plot,
)


class TestPlots(unittest.TestCase):
    def test_error_metrics(self):
        metrics = error_metrics(np.array([2.0, 0.0]), np.array([1.0, 1.0]))
        self.assertAlmostEqual(metrics['relative_l2'], 1)
        self.assertAlmostEqual(metrics['max_abs'], 1)
        self.assertEqual(error_metrics(np.zeros(2), np.zeros(2))['relative_l2'], 0)
        self.assertTrue(np.isinf(error_metrics(np.ones(2), np.zeros(2))['relative_l2']))

    def test_rectangular_field_orientation_and_shared_scale(self):
        xy = np.array([[1, 2], [0, 0], [1, 0], [0, 1], [0, 2], [1, 1]])
        values = xy[:, 0] + 10 * xy[:, 1]
        with patch('learnpdes.utils.plot.save_figure') as save:
            save_2d_plot('.', 10, xy, values, 2.31e-6, None, lambda x, y: x + 10 * y)
        fig = save.call_args.args[0]
        try:
            predicted, reference, error = [ax.collections[0] for ax in fig.axes[:3]]
            np.testing.assert_array_equal(
                predicted.get_array(), [[0, 1], [10, 11], [20, 21]]
            )
            self.assertIs(predicted.norm, reference.norm)
            np.testing.assert_array_equal(error.get_array(), np.zeros((3, 2)))
            self.assertIn('2.31e-06', fig._suptitle.get_text())
            self.assertEqual(
                [ax.get_title() for ax in fig.axes[:3]],
                ['Prediction', 'Reference', 'Signed error'],
            )
        finally:
            plt.close(fig)

    def test_frame_dimensions_and_zero_flow(self):
        with tempfile.TemporaryDirectory() as folder:
            x = np.linspace(-3, 3, 30)
            save_plot(
                folder, 1, x, np.cos(x), 1e-6, None, np.cos, [(0, 1), (1, 1e-6)], 10
            )
            xy = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
            save_airfoil_plot(folder, 2, xy, (np.zeros(4),) * 3, 0, None)
            for step in (1, 2):
                with Image.open(Path(folder) / f'epoch_{step}.png') as frame:
                    self.assertEqual(frame.size, (1600, 900))


if __name__ == '__main__':
    unittest.main()
