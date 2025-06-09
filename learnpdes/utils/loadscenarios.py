'''
Utility functions.
'''

# ======= Imports =======

import re
import torch
import numpy as np

from torch import linspace
from learnpdes.utils.plot import plot_mesh
from learnpdes.model.encodings import identity
from learnpdes.utils.utility import (
    analyze_xy,
    laplace_function,
    get_marker_masks,
)

from torch import Tensor
from typing import (
    Union,
    Callable,
)

from learnpdes import (
    EXPONENTIAL_SCENARIO,
    COSINUS_SCENARIO,
    LAPLACE_SCENARIO,
    POTENTIAL_FLOW_SCENARIO,
    WIND_TUNNEL_SCENARIO,
)

# ======= Functions =======


def load_real_space(
    num_inputs: int
) -> tuple[Tensor, dict[str, Tensor]]:
    '''
    Load real segment around 0, ensuring correct order.
    '''
    real_space = torch.cat([
        linspace(-3, 3, num_inputs),
        torch.tensor([0.0]),
    ])
    sorted_space, _ = torch.sort(real_space)
    mask_zero = sorted_space == 0
    mesh_masks = {"zero": mask_zero}
    return sorted_space, mesh_masks


def load_exponential(
    num_inputs: int
) -> tuple[
    Tensor,
    dict[str, Tensor],
    int, Callable,
    Callable, Callable, Callable
]:
    '''
    Load configuration space (around 0) for exponential PDE
    '''
    # Define constants
    output_dim = 1
    input_homeo = identity
    output_homeo = identity
    encoding = identity
    analytical = np.exp

    # Load space
    x, mesh_masks = load_real_space(num_inputs)

    return (
        x, mesh_masks,
        output_dim, analytical,
        input_homeo, output_homeo, encoding,
    )


def load_cosinus(
    num_inputs: int
) -> tuple[
    Tensor,
    dict[str, Tensor],
    int, Callable,
    Callable, Callable, Callable
]:
    '''
    Load configuration space (around 0) for exponential PDE
    '''
    # Define constants
    output_dim = 1
    input_homeo = identity
    output_homeo = identity
    encoding = identity
    analytical = np.cos

    # Load space
    x, mesh_masks = load_real_space(num_inputs)

    return (
        x, mesh_masks,
        output_dim, analytical,
        input_homeo, output_homeo, encoding,
    )


def load_laplace(
    num_inputs: int
) -> tuple[
    Tensor, dict[str, Tensor],
    int, Callable,
    Callable, Callable, Callable
]:
    '''
    Load a square [0, 1] x [0, 1] as input space for laplace PDE.
    '''
    # Define constants
    output_dim = 1
    input_homeo = identity
    output_homeo = identity
    encoding = identity
    analytical = laplace_function

    # Load space
    xy = torch.cartesian_prod(
        torch.linspace(0, 1, num_inputs),
        torch.linspace(0, 1, num_inputs),
    )

    # Create masks for each boundary
    x = xy[:, 0]
    y = xy[:, 1]
    mesh_masks = {
        "inlet": x == 0,
        "outlet": x == 1,
        "bottom": y == 0,
        "top": y == 1,
    }

    return (
        xy, mesh_masks,
        output_dim, analytical,
        input_homeo, output_homeo, encoding,
    )


def load_wind_tunnel(
    num_inputs: int
) -> tuple[
    Tensor, dict[str, Tensor],
    int, Callable,
    Callable, Callable, Callable
]:
    '''
    Load a square [0, 4] x [0, 1] as input space.
    '''
    # Define constants
    output_dim = 1
    input_homeo = identity
    output_homeo = identity
    encoding = identity

    # Load space
    xy = torch.cartesian_prod(
        torch.linspace(0, 4, num_inputs),
        torch.linspace(0, 1, num_inputs),
    )

    # Create masks for each boundary
    x = xy[:, 0]
    y = xy[:, 1]
    mesh_masks = {
        "inlet": x == 0,
        "outlet": x == 4,
        "wall": ((y == 0) | (y == 1)),
    }

    return (
        xy, mesh_masks,
        output_dim, None,
        input_homeo, output_homeo, encoding,
    )


def load_potential_flow(
    num_inputs: int,
    plot: bool = False,
    augmented_grid: int = True,
) -> tuple[
    Tensor, dict[str, Tensor],
    int, None,
    Callable, Callable, Callable
]:
    '''
    Loads node coordinates from a SU2 mesh file.
    Returns as a tensor of shape [N, 2].
    '''
    # Define constants
    output_dim = 1  # potential (u = dphi_dx, v = dphi_dy)
    input_homeo, output_homeo, encoding = identity, identity, identity

    filepath = "./meshes/mesh_airfoil_ch10sm.su2"
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find the line with "NPOIN"
    for i, line in enumerate(lines):
        if "NPOIN" in line:
            num_points = int(re.findall(r'\d+', line)[0])
            start_idx = i + 1
            break
    else:
        raise ValueError("NPOIN not found in SU2 file.")

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
    mesh_masks = get_marker_masks(filepath, num_points)

    print(f"Total number of vertices: {xy.shape[0]}")
    if plot:
        plot_mesh(xy, mesh_masks)

    return (
        xy, mesh_masks,
        output_dim, None,
        input_homeo, output_homeo, encoding,
    )


def load_scenario(
    scenario: str,
    num_inputs: int = 100,
) -> tuple[
    Tensor, dict[str, Tensor],
    int, Union[Callable, None],
    Callable, Callable, Callable,
]:
    '''
    Function that loads the according scenario with:
        - space data
        - mesh masks (for boundaries)
        - output dimension of the PINN
        - analytical solution (= None) if applicable
        - input homeomorphism
        - output homeomorphism
        - encoding
    '''
    print(f'Loading scenario: {scenario}')

    if scenario == EXPONENTIAL_SCENARIO:
        return load_exponential(num_inputs)
    elif scenario == COSINUS_SCENARIO:
        return load_cosinus(num_inputs)
    elif scenario == LAPLACE_SCENARIO:
        return load_laplace(num_inputs)
    elif scenario == POTENTIAL_FLOW_SCENARIO:
        return load_potential_flow(num_inputs)
    elif scenario == WIND_TUNNEL_SCENARIO:
        return load_wind_tunnel(num_inputs)
    else:
        raise ValueError(
            f'{scenario=} is not a valid scenario. '
            'Please look at the README.md '
            'for valid scenarios identifiers.'
        )
