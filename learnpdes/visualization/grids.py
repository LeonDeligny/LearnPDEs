"""Plotting geometry, kept independent of training samples and scenario dispatch."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
from torch import Tensor

from learnpdes.types import Array


@dataclass
class VisualizationGrid:
    coordinates: np.ndarray
    triangles: np.ndarray | None = None
    boundary_edges: np.ndarray | None = None


def _cell_triangles(elements: list[str]) -> list[tuple[int, ...]]:
    triangles: list[tuple[int, ...]] = []
    for element in elements:
        values = [int(value) for value in element.split()]
        if values[0] == 5:
            triangles.append(tuple(values[1:4]))
        elif values[0] == 9:
            a, b, c, d = values[1:5]
            triangles.extend([(a, b, c), (a, c, d)])
        else:
            raise ValueError(f'Unsupported SU2 cell type: {values[0]}')
    return triangles


def _read_airfoil_mesh(mesh_path: str | Path) -> tuple[Array, Array, Array]:
    lines = Path(mesh_path).read_text().splitlines()
    triangles, edges, coordinates = [], [], []
    for index, line in enumerate(lines):
        if line.startswith('NELEM='):
            count = int(line.split('=')[1])
            triangles.extend(_cell_triangles(lines[index + 1 : index + 1 + count]))
        elif line.startswith('NPOIN='):
            count = int(line.split('=')[1].split()[0])
            coordinates = [
                [float(value) for value in point.split()[:2]]
                for point in lines[index + 1 : index + 1 + count]
            ]
        elif line.startswith('MARKER_TAG=') and line.split('=')[1].strip() == 'airfoil':
            count = int(lines[index + 1].split('=')[1])
            edges = [
                [int(value) for value in edge.split()[1:3]]
                for edge in lines[index + 2 : index + 2 + count]
            ]
    if not all((coordinates, triangles, edges)):
        raise ValueError(
            'Expected SU2 fluid cells, coordinates, and an airfoil marker.'
        )
    return np.asarray(coordinates), np.asarray(triangles), np.asarray(edges)


def airfoil_grid(mesh_path: str | Path, subdivisions: int = 1) -> VisualizationGrid:
    """Refine the SU2 fluid cells, retaining the hole and boundary connectivity.

    Refining the existing mesh preserves its concentration of samples near the
    airfoil. Re-triangulating only the points would fill the solid interior.
    """
    if subdivisions < 0:
        raise ValueError('Mesh subdivisions cannot be negative.')
    xy, triangles, edges = _read_airfoil_mesh(mesh_path)
    boundary_edges = xy[edges]
    for _ in range(subdivisions):
        # Give adjacent cells the same midpoint, preserving connectivity and
        # refining only existing fluid cells rather than filling the solid hole.
        cell_edges = triangles[:, [[0, 1], [1, 2], [2, 0]]]
        unique_edges, inverse = np.unique(
            np.sort(cell_edges.reshape(-1, 2), axis=1), axis=0, return_inverse=True
        )
        midpoints = inverse.reshape(-1, 3) + len(xy)
        xy = np.vstack((xy, xy[unique_edges.astype(np.intp)].mean(axis=1)))
        a, b, c = triangles.T
        ab, bc, ca = midpoints.T
        triangles = np.vstack(
            (
                np.column_stack((a, ab, ca)),
                np.column_stack((ab, b, bc)),
                np.column_stack((ca, bc, c)),
                np.column_stack((ab, bc, ca)),
            )
        )
    return VisualizationGrid(xy, triangles, boundary_edges)


def rectangular_triangles(coordinates: np.ndarray) -> np.ndarray:
    """Connect a complete rectangular grid without triangulating across holes."""
    xy = np.asarray(coordinates)
    x, y = np.unique(xy[:, 0]), np.unique(xy[:, 1])
    if (
        len(x) < 2
        or len(y) < 2
        or len(x) * len(y) != len(xy)
        or len(np.unique(xy, axis=0)) != len(xy)
    ):
        raise ValueError('Nonrectangular flow grids require explicit fluid triangles.')
    nodes = np.lexsort((xy[:, 0], xy[:, 1])).reshape(len(y), len(x))
    a, b = nodes[:-1, :-1].ravel(), nodes[:-1, 1:].ravel()
    c, d = nodes[1:, 1:].ravel(), nodes[1:, :-1].ravel()
    return np.vstack((np.column_stack((a, b, c)), np.column_stack((a, c, d))))


def line_grid(
    input_space: Tensor,
    resolution: int,
    *,
    bounds: tuple[float, float] | None = None,
    **kwargs: object,
) -> VisualizationGrid:
    values = input_space.detach().cpu().numpy()
    low, high = bounds if bounds is not None else (values.min(), values.max())
    return VisualizationGrid(np.linspace(low, high, resolution)[:, None])


def rectangle_grid(
    input_space: Tensor, resolution: int, **kwargs: object
) -> VisualizationGrid:
    bounds = input_space.detach().cpu().numpy()
    x, y = np.meshgrid(
        np.linspace(bounds[:, 0].min(), bounds[:, 0].max(), resolution),
        np.linspace(bounds[:, 1].min(), bounds[:, 1].max(), resolution),
        indexing='ij',
    )
    return VisualizationGrid(np.column_stack((x.ravel(), y.ravel())))


def ring_grid(
    center: ArrayLike,
    inner: float,
    outer: float | Callable[[Array], Array],
    resolution: int,
    *,
    outer_boundary: bool = False,
) -> VisualizationGrid:
    count = max(64, 4 * resolution)
    angles = np.arange(count) * (2 * np.pi / count)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    if callable(outer):
        outer_radii = np.asarray(outer(directions), dtype=float)
    else:
        outer_radii = np.full(count, float(outer))
    fraction = np.linspace(0, 1, max(3, resolution)) ** 1.5
    radii = inner + fraction[:, None] * (outer_radii - inner)
    coordinates = (np.asarray(center) + radii[:, :, None] * directions).reshape(-1, 2)
    a = np.arange((len(fraction) - 1) * count)
    b = a // count * count + (a + 1) % count
    triangles = np.vstack(
        (np.column_stack((a, b, b + count)), np.column_stack((a, b + count, a + count)))
    )
    edges = coordinates[
        np.column_stack((np.arange(count), (np.arange(count) + 1) % count))
    ]
    if outer_boundary:
        ring = np.arange(count) + (len(fraction) - 1) * count
        edges = np.concatenate(
            (edges, coordinates[np.column_stack((ring, np.roll(ring, -1)))])
        )
    return VisualizationGrid(coordinates, triangles, edges)
