"""Check that rendered animation frames keep comparable color mappings."""

import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from learnpdes import LAPLACE_SCENARIO, POTENTIAL_FLOW_SCENARIO
from learnpdes.utils.plot import get_plot_func
from learnpdes.utils.utility import laplace_function


class TestPlotColors(unittest.TestCase):
    def setUp(self):
        self.points = np.array([(x, y) for x in (0, 0.5, 1) for y in (0, 0.5, 1)])
        self.reference = laplace_function(*self.points.T)
        self.addCleanup(plt.close, 'all')

    def render(self, plotter, values, epoch=0, analytical=None, **kwargs):
        with patch('learnpdes.utils.plot.save_figure') as save:
            plotter(
                '.',
                epoch,
                self.points,
                values,
                1.0,
                None,
                analytical=analytical,
                **kwargs,
            )
        fig = save.call_args.args[0]
        fig.canvas.draw()
        return [ax.collections[0] for ax in fig.axes[:3] if ax.collections]

    def test_laplace_scale_is_shared_and_frozen_across_frames(self):
        plotter = get_plot_func(LAPLACE_SCENARIO)
        first = self.render(
            plotter, 2 * self.reference - 0.25, analytical=laplace_function
        )
        limits = [artist.get_clim() for artist in first]
        colors = [artist.to_rgba([0, 0.25, 1]) for artist in first]
        self.assertIs(first[0].norm, first[1].norm)
        self.assertLessEqual(limits[0][0], -0.25)
        self.assertGreaterEqual(limits[0][1], 1.75)
        np.testing.assert_array_equal(colors[0], colors[1])

        later = self.render(
            plotter, self.reference + 0.01, epoch=100, analytical=laplace_function
        )
        for artist, expected_limits, expected_colors in zip(later, limits, colors):
            self.assertEqual(artist.get_clim(), expected_limits)
            np.testing.assert_array_equal(artist.to_rgba([0, 0.25, 1]), expected_colors)
            self.assertEqual(artist.colorbar.extend, 'both')
        self.assertEqual(later[2].norm(0), 0.5)
        self.assertLess(abs(later[2].norm(0.01) - 0.5), 0.02)

    def test_signed_errors_are_centered_for_each_sign_and_exact_solution(self):
        for offset in (-0.1, 0, 0.1):
            with self.subTest(offset=offset):
                error = self.render(
                    get_plot_func(LAPLACE_SCENARIO),
                    self.reference + offset,
                    analytical=laplace_function,
                )[2]
                self.assertEqual(error.norm(0), 0.5)
                self.assertEqual(error.norm.vmin, -error.norm.vmax)
                self.assertGreater(error.norm.vmax, 0)
                self.assertEqual(error.cmap.name, 'RdBu_r')
                if offset == 0:
                    np.testing.assert_array_equal(error.norm(error.get_array()), 0.5)

    def test_flow_scales_and_colorbars_do_not_follow_frame_extrema(self):
        plotter = get_plot_func(POTENTIAL_FLOW_SCENARIO)
        first = self.render(plotter, (-2 - self.points[:, 0], np.zeros(9), np.zeros(9)))
        limits = [artist.get_clim() for artist in first]
        colors = [artist.to_rgba([-0.5, 0, 0.5]) for artist in first]
        ticks = [artist.colorbar.get_ticks().copy() for artist in first]
        later = self.render(
            plotter,
            (20 * self.points[:, 0], -10 * self.points[:, 1], np.full(9, 5.0)),
            epoch=100,
        )
        for index, artist in enumerate(later):
            self.assertEqual(artist.get_clim(), limits[index])
            np.testing.assert_array_equal(artist.to_rgba([-0.5, 0, 0.5]), colors[index])
            np.testing.assert_array_equal(artist.colorbar.get_ticks(), ticks[index])
            self.assertEqual(artist.colorbar.extend, 'both')
        self.assertEqual(first[0].norm(0), 0.5)
        self.assertEqual(first[1].norm(0), 0.5)

    def test_new_plotters_have_independent_limits_without_a_reference(self):
        first_plotter = get_plot_func(LAPLACE_SCENARIO)
        first = self.render(first_plotter, np.zeros(9))[0]
        original_limits = first.get_clim()
        other = self.render(get_plot_func(LAPLACE_SCENARIO), np.full(9, 100.0))[0]
        self.assertNotEqual(original_limits, other.get_clim())
        later = self.render(first_plotter, np.full(9, 10.0), epoch=100)[0]
        self.assertEqual(later.get_clim(), original_limits)

    def test_caller_can_choose_animation_limits(self):
        norms = {'solution': Normalize(-5, 5), 'error': Normalize(-2, 2)}
        artists = self.render(
            get_plot_func(LAPLACE_SCENARIO),
            self.reference,
            analytical=laplace_function,
            color_norms=norms,
        )
        self.assertIs(artists[0].norm, norms['solution'])
        self.assertIs(artists[1].norm, norms['solution'])
        self.assertIs(artists[2].norm, norms['error'])


if __name__ == '__main__':
    unittest.main()
