"""Sample the supplied fluid triangles for flat Plotly heatmaps.

Interpolation uses only existing cells, so holes and concave boundaries are
never filled by a new triangulation. The weights are reused for every frame.
"""

import numpy as np

from learnpdes.utils.visualization import rectangular_triangles


class MeshRaster:
    def __init__(self, xy, triangles, resolution=401):
        xy = np.asarray(xy, dtype=float)
        triangles = np.asarray(triangles)
        x, y = np.unique(xy[:, 0]), np.unique(xy[:, 1])
        self.order = None
        if len(x) * len(y) == len(xy) and np.array_equal(
            triangles, rectangular_triangles(xy)
        ):
            self.x, self.y = x, y
            self.order = np.lexsort((xy[:, 0], xy[:, 1]))
            return

        span = np.ptp(xy, axis=0)
        counts = np.maximum(
            2, np.rint((resolution - 1) * span / span.max()).astype(int) + 1
        )
        self.x = np.linspace(xy[:, 0].min(), xy[:, 0].max(), counts[0])
        self.y = np.linspace(xy[:, 1].min(), xy[:, 1].max(), counts[1])
        self.vertices = np.zeros((*counts[::-1], 3), dtype=int)
        self.weights = np.full((*counts[::-1], 3), np.nan)
        for cell in triangles:
            a, b, c = xy[cell]
            lower, upper = xy[cell].min(axis=0), xy[cell].max(axis=0)
            ix0, ix1 = (
                np.searchsorted(self.x, lower[0]),
                np.searchsorted(self.x, upper[0], side='right'),
            )
            iy0, iy1 = (
                np.searchsorted(self.y, lower[1]),
                np.searchsorted(self.y, upper[1], side='right'),
            )
            if ix0 == ix1 or iy0 == iy1:
                continue
            determinant = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
            if determinant == 0:
                continue
            xx, yy = np.meshgrid(self.x[ix0:ix1], self.y[iy0:iy1])
            wa = (
                (b[1] - c[1]) * (xx - c[0]) + (c[0] - b[0]) * (yy - c[1])
            ) / determinant
            wb = (
                (c[1] - a[1]) * (xx - c[0]) + (a[0] - c[0]) * (yy - c[1])
            ) / determinant
            weights = np.stack((wa, wb, 1 - wa - wb), axis=-1)
            inside = np.all(weights >= -1e-10, axis=-1)
            self.vertices[iy0:iy1, ix0:ix1][inside] = cell
            self.weights[iy0:iy1, ix0:ix1][inside] = weights[inside]

    def sample(self, values):
        values = np.asarray(values)
        if self.order is not None:
            return (
                values[self.order].reshape(len(self.y), len(self.x)).astype(np.float32)
            )
        return np.sum(values[self.vertices] * self.weights, axis=-1).astype(np.float32)
