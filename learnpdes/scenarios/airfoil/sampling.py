"""Load geometry and collocation coordinates from the bundled or supplied mesh."""

import re
from collections.abc import Callable
from pathlib import Path

import torch
from torch import Tensor

from learnpdes.model.encodings import identity
from learnpdes.resources import DEFAULT_MESH
from learnpdes.utils.utility import analyze_xy, get_marker_masks


def load_2d_mesh(
    num_inputs: int,
    plot: bool = False,
    augmented_grid: int = True,
    filepath: str | Path = DEFAULT_MESH,
) -> tuple[
    Tensor,
    dict[str, Tensor],
    int,
    None,
    Callable[[Tensor], Tensor],
    Callable[[Tensor], Tensor],
    Callable[[Tensor], Tensor],
]:
    """Loads node coordinates from a SU2 mesh file.

    Returns as a tensor of shape [N, 2].
    """
    # Define constants
    output_dim = 1  # potential (u = dphi_dx, v = dphi_dy)
    input_homeo, output_homeo, encoding = identity, identity, identity

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find the line with "NPOIN"
    for i, line in enumerate(lines):
        if 'NPOIN' in line:
            num_points = int(re.findall(r'\d+', line)[0])
            start_idx = i + 1
            break
    else:
        raise ValueError('NPOIN not found in SU2 file.')

    # Read the next num_points lines for coordinates
    coords = []
    for j in range(start_idx, start_idx + num_points):
        parts = lines[j].strip().split()
        x, y = float(parts[0]), float(parts[1])
        coords.append([x, y])

    xy = torch.tensor(coords, dtype=torch.float32)

    if augmented_grid:
        # Compute min/max for each axis
        xy_min = xy.min(dim=0).values
        xy_max = xy.max(dim=0).values

        # Option 1: Full bounding box
        x0, x1 = 1.0, xy_max[0].item()
        y0, y1 = xy_min[1].item(), xy_max[1].item()

        # Create the grid points
        xg = torch.linspace(x0, x1, num_inputs)
        yg = torch.linspace(y0, y1, num_inputs)
        grid_points = torch.cartesian_prod(xg, yg)

        # Concatenate to the existing mesh
        xy = torch.cat([xy, grid_points], dim=0)

    analyze_xy(xy)
    num_points = xy.shape[0]
    mesh_masks = get_marker_masks(str(filepath), num_points)

    print(f'Total number of vertices: {xy.shape[0]}')
    if plot:
        from learnpdes.utils.plot import plot_mesh

        plot_mesh(xy, mesh_masks)

    return (
        xy,
        mesh_masks,
        output_dim,
        None,
        input_homeo,
        output_homeo,
        encoding,
    )
