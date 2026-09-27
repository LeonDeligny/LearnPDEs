"""Verify interpolation accuracy and preservation of missing fluid cells."""

import unittest

import numpy as np

from learnpdes.utils.mesh_raster import MeshRaster
from learnpdes.utils.visualization import rectangular_triangles


class TestMeshRaster(unittest.TestCase):
    def test_affine_field_and_hole_use_only_supplied_cells(self):
        x, y = np.meshgrid(np.arange(5), np.arange(5))
        xy = np.column_stack((x.ravel(), y.ravel()))
        cells = rectangular_triangles(xy)
        centers = xy[cells].mean(axis=1)
        cells = cells[~np.all((centers > 1) & (centers < 3), axis=1)]
        raster = MeshRaster(xy, cells, resolution=41)
        actual = raster.sample(2 * xy[:, 0] - 3 * xy[:, 1] + 7)
        xx, yy = np.meshgrid(raster.x, raster.y)
        hole = (xx > 1) & (xx < 3) & (yy > 1) & (yy < 3)
        self.assertTrue(np.isnan(actual[hole]).all())
        np.testing.assert_allclose(
            actual[~hole], (2 * xx - 3 * yy + 7)[~hole], atol=1e-6
        )

    def test_complete_grid_preserves_original_samples_in_any_node_order(self):
        xy = np.array([[1.0, 2.0], [0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
        raster = MeshRaster(xy, rectangular_triangles(xy))
        np.testing.assert_array_equal(raster.sample([17, 9, 3, 5]), [[9, 3], [5, 17]])


if __name__ == '__main__':
    unittest.main()
