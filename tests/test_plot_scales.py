"""Verify comparable numerical ranges in exported Plotly figures."""

import unittest

import numpy as np

from learnpdes import LAPLACE_SCENARIO, POTENTIAL_FLOW_SCENARIO
from learnpdes.utils.plot import get_plot_func


class TestPlotScales(unittest.TestCase):
    def setUp(self):
        self.points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

    def capture(self, plotter, values, step=0):
        plotter(
            '.',
            epoch=step,
            inputs=self.points,
            f=values,
            loss=1.0,
            analytical=lambda x, y: x + y,
            loss_history=[(0, 1.0), (step, 1.0)],
        )

    def test_signed_error_is_centered_for_both_signs_and_exact_solution(self):
        for offset in (-0.1, 0, 0.1):
            with self.subTest(offset=offset):
                plotter = get_plot_func(LAPLACE_SCENARIO)
                self.capture(plotter, self.points.sum(axis=1) + offset)
                scale = plotter.figure().layout.coloraxis2
                self.assertEqual(scale.cmin, -scale.cmax)
                self.assertGreater(scale.cmax, 0)
                self.assertGreaterEqual(scale.cmax + 1e-12, abs(offset))

    def test_flow_scales_cover_all_frames_and_accept_explicit_limits(self):
        plotter = get_plot_func(POTENTIAL_FLOW_SCENARIO, color_limits={'p': (0, 10)})
        self.capture(plotter, (np.zeros(4),) * 3)
        self.capture(plotter, (np.full(4, -5.0), np.full(4, 3.0), np.ones(4)), step=10)
        fig = plotter.figure()
        for trace, bound in zip(fig.data[:2], [5, 3]):
            self.assertEqual(trace.zmin, -trace.zmax)
            self.assertGreaterEqual(trace.zmax, bound)
            self.assertFalse(trace.zauto)
        self.assertEqual((fig.data[2].zmin, fig.data[2].zmax), (0, 10))
        # Frame deltas update values only, so the original ranges stay fixed.
        for frame in fig.frames:
            for trace in frame.data[:3]:
                self.assertIsNone(trace.zmin)
                self.assertIsNone(trace.zmax)

    def test_plotters_have_independent_ranges(self):
        first, second = [get_plot_func(LAPLACE_SCENARIO) for _ in range(2)]
        self.capture(first, np.zeros(4))
        self.capture(second, np.full(4, 100.0))
        self.assertEqual(first.figure().layout.coloraxis.cmax, 2)
        self.assertEqual(second.figure().layout.coloraxis.cmax, 100)


if __name__ == '__main__':
    unittest.main()
