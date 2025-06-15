'''
Utility functions.
'''

# ======= Imports =======

import torch

from torch import Tensor
from numpy import ndarray
from torch.nn import Module
from typing import Union

from numpy import (
    pi,
    sin,
    sinh,
)

# ======= Class =======


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x

# ======= Functions =======


def laplace_function(x: ndarray, y: ndarray) -> ndarray:
    '''
    Laplace function.
    '''
    return sin(pi * x) * sinh(pi * y) / sinh(pi)


def analyze_xy(xy: Tensor) -> None:
    x = xy[:, 0]
    y = xy[:, 1]

    x_min, x_max = x.min().item(), x.max().item()
    y_min, y_max = y.min().item(), y.max().item()

    x_min_count = (x == x_min).sum().item()
    x_max_count = (x == x_max).sum().item()
    y_min_count = (y == y_min).sum().item()
    y_max_count = (y == y_max).sum().item()

    print(f"x min: {x_min} (count: {x_min_count})")
    print(f"x max: {x_max} (count: {x_max_count})")
    print(f"y min: {y_min} (count: {y_min_count})")
    print(f"y max: {y_max} (count: {y_max_count})")


def get_marker_masks(
    filepath: str,
    num_points: int
) -> dict[str, Tensor]:
    """
    Returns a dictionary
    mapping each MARKER_TAG to a boolean mask over the node array.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    marker_masks: dict[str, Tensor] = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("MARKER_TAG="):
            tag = line.split("=")[1].strip()
            # Find number of elements for this marker
            while not lines[i].startswith("MARKER_ELEMS="):
                i += 1
            num_elems = int(lines[i].split("=")[1].strip())
            indices = []
            for j in range(i + 1, i + 1 + num_elems):
                parts = lines[j].strip().split()
                # For line elements, last two numbers are node indices
                idxs = [int(x) for x in parts[-2:]]
                indices.extend(idxs)
            indices = list(set(indices))
            mask = torch.zeros(num_points, dtype=torch.bool)
            mask[indices] = True
            marker_masks[tag] = mask
            i = i + num_elems
        i += 1

    for tag, mask in marker_masks.items():
        print(f"Marker: {tag}, Number of elements: {mask.sum().item()}")

    return marker_masks


def order_airfoil_points(xy: Tensor) -> Tensor:
    # xy: [N, 2] airfoil points (unordered)
    ordered = [0]
    used = set(ordered)
    for _ in range(1, len(xy)):
        last = ordered[-1]
        dists = torch.norm(xy - xy[last], dim=1)
        dists[list(used)] = float('inf')
        next_idx = torch.argmin(dists).item()
        ordered.append(next_idx)
        used.add(next_idx)
    return xy[ordered]


def compute_normals(
    xy: Tensor,
    airfoil_mask: Tensor
) -> tuple[Tensor, Tensor]:
    """
    Compute outward normals for the airfoil boundary.
    """
    # Extract airfoil boundary points
    airfoil_pts = xy[airfoil_mask]  # shape [M, 2]

    # Compute tangents using finite differences (circular for closed airfoil)
    tangents = airfoil_pts.roll(-1, 0) - airfoil_pts.roll(1, 0)  # shape [M, 2]
    tangents = tangents / (torch.norm(tangents, dim=1, keepdim=True) + 1e-12)

    # Rotate tangent by 90 degrees to get normal (outward direction: (dy, -dx))
    normals = torch.stack(
        [tangents[:, 1], -tangents[:, 0]],
        dim=1,
    )  # shape [M, 2]
    normals = normals / (torch.norm(normals, dim=1, keepdim=True) + 1e-12)

    # Place normals back into full mesh shape (zeros elsewhere)
    normals_x = torch.zeros(xy.shape[0], dtype=xy.dtype, device=xy.device)
    normals_y = torch.zeros(xy.shape[0], dtype=xy.dtype, device=xy.device)
    normals_x[airfoil_mask] = normals[:, 0]
    normals_y[airfoil_mask] = normals[:, 1]

    return -normals_x, -normals_y


def detach_to_numpy(
    f: Union[tuple[Tensor, ...], Tensor]
) -> Union[tuple[ndarray, ...], ndarray]:
    """
    Essentially sends to cpu, detach, and converts to ndarray.
    Applies .cpu().detach().numpy() on object.
    """
    if isinstance(f, tuple):
        return tuple(fi.cpu().detach().numpy() for fi in f)
    else:
        return f.cpu().detach().numpy()
