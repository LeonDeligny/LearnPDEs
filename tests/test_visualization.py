"""Checks for independent evaluation and geometry-preserving plotting samples."""

import unittest

import numpy as np
import torch

from learnpdes import (
    LAPLACE_SCENARIO,
    POTENTIAL_FLOW_SCENARIO,
    SOLENOIDAL_FLOW_SCENARIO,
)
from learnpdes.utils.visualization import (
    ModelEvaluator,
    VisualizationGrid,
    airfoil_grid,
    visualization_grid,
)


class Quadratic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, xy):
        return self.scale * (xy[:, :1] ** 2 + 3 * xy[:, 1:2] ** 2)


class TestVisualization(unittest.TestCase):
    def test_dense_grid_does_not_modify_training_grid(self):
        training = torch.cartesian_prod(
            torch.linspace(0, 1, 10), torch.linspace(0, 1, 10)
        )
        original = training.clone()
        grid = visualization_grid(LAPLACE_SCENARIO, training)
        self.assertEqual(grid.coordinates.shape, (90_000, 2))
        torch.testing.assert_close(training, original)
        model = Quadratic()
        result = ModelEvaluator(model, LAPLACE_SCENARIO, grid)()
        np.testing.assert_allclose(
            result['f'].ravel(),
            grid.coordinates[:, 0] ** 2 + 3 * grid.coordinates[:, 1] ** 2,
            atol=1e-6,
        )
        self.assertTrue(model.training)
        self.assertIsNone(model.scale.grad)

    def test_flow_uses_derivatives_without_parameter_gradients(self):
        model = Quadratic()
        model.eval()
        xy = np.array([[0.2, 0.4], [0.5, 0.8], [0.9, 0.1]])
        for scenario in (POTENTIAL_FLOW_SCENARIO, SOLENOIDAL_FLOW_SCENARIO):
            with self.subTest(scenario=scenario):
                result = ModelEvaluator(
                    model, scenario, VisualizationGrid(xy), density=2, batch_size=2
                )()
                u, v, p = result['f']
                if scenario == POTENTIAL_FLOW_SCENARIO:
                    np.testing.assert_allclose(u, 2 * xy[:, 0])
                    np.testing.assert_allclose(v, 6 * xy[:, 1])
                    np.testing.assert_allclose(p, u**2 + v**2)
                else:
                    np.testing.assert_allclose(u, 6 * xy[:, 1])
                    np.testing.assert_allclose(v, -2 * xy[:, 0])
                    np.testing.assert_array_equal(p, 0)
                self.assertFalse(model.training)
                self.assertIsNone(model.scale.grad)

    def test_refinement_preserves_fluid_area_and_solid_hole(self):
        mesh_path = 'meshes/mesh_airfoil_ch10sm.su2'
        coarse = airfoil_grid(mesh_path, subdivisions=0)
        fine = airfoil_grid(mesh_path)
        self.assertGreater(len(fine.coordinates), len(coarse.coordinates))
        np.testing.assert_array_equal(fine.boundary_edges, coarse.boundary_edges)

        def area(grid):
            vertices = grid.coordinates[grid.triangulation.triangles]
            a, b = vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]
            return np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum() / 2

        self.assertAlmostEqual(area(coarse), area(fine))
        # The middle of the airfoil must remain outside the fluid triangulation.
        center = coarse.boundary_edges.mean(axis=(0, 1))
        self.assertEqual(fine.triangulation.get_trifinder()(*center), -1)


if __name__ == '__main__':
    unittest.main()
