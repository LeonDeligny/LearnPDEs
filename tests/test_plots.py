"""Check numerical metrics and Plotly mesh data."""

import unittest

from learnpdes.scenarios import DEFAULT_MESH

import numpy as np
import plotly.graph_objects as go
import torch

from learnpdes import POTENTIAL_FLOW_SCENARIO
from learnpdes.utils.plot import error_metrics, get_plot_func, plot_mesh, plot_xy
from learnpdes.utils.visualization import airfoil_grid


class TestPlots(unittest.TestCase):
    def test_error_metrics(self):
        metrics = error_metrics(np.array([2.0, 0.0]), np.array([1.0, 1.0]))
        self.assertAlmostEqual(metrics['relative_l2'], 1)
        self.assertAlmostEqual(metrics['max_abs'], 1)
        self.assertEqual(error_metrics(np.zeros(2), np.zeros(2))['relative_l2'], 0)
        self.assertTrue(np.isinf(error_metrics(np.ones(2), np.zeros(2))['relative_l2']))

    def test_plotly_uses_supplied_fluid_cells_without_filling_airfoil(self):
        grid = airfoil_grid(DEFAULT_MESH, subdivisions=0)
        plotter = get_plot_func(POTENTIAL_FLOW_SCENARIO)
        plotter(
            '.',
            epoch=0,
            inputs=grid.coordinates,
            f=(np.zeros(len(grid.coordinates)),) * 3,
            loss=0.0,
            analytical=None,
            loss_history=[(0, 0.0)],
            triangles=grid.triangles,
            boundary_edges=grid.boundary_edges,
        )
        fig = plotter.figure()
        for trace in fig.data[:3]:
            values = np.asarray(trace.z)
            np.testing.assert_array_equal(values[np.isfinite(values)], 0)
            # A sample inside the solid airfoil must have no field value.
            ix = np.argmin(np.abs(np.asarray(trace.x) - 0.5))
            iy = np.argmin(np.abs(np.asarray(trace.y) - 0.08))
            self.assertTrue(np.isnan(values[iy, ix]))
        self.assertTrue(np.isfinite(fig.layout.yaxis4.range).all())

    def test_mesh_inspection_returns_plotly_figures(self):
        xy = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        nodes = plot_xy(xy, show=False)
        mesh = plot_mesh(xy, {'airfoil': torch.ones(4, dtype=torch.bool)}, show=False)
        self.assertIsInstance(nodes, go.Figure)
        self.assertIsInstance(mesh, go.Figure)
        np.testing.assert_array_equal(nodes.data[0].x, xy[:, 0].numpy())
        self.assertEqual(len(mesh.data), 2)


if __name__ == '__main__':
    unittest.main()
