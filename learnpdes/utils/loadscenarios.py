"""
Utility functions.
"""

# ======= Imports =======

import re
import torch
from learnpdes.scenarios import DEFAULT_MESH

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
    cosinus,
    kovasznay,
    KOVASZNAY_SCENARIO,
    poiseuille,
    EXPONENTIAL_SCENARIO,
    COSINUS_SCENARIO,
    CYLINDER_SCENARIO,
    LAPLACE_SCENARIO,
    POISEUILLE_SCENARIO,
    POTENTIAL_FLOW_SCENARIO,
    SOLENOIDAL_FLOW_SCENARIO,
)

# ======= Functions =======


def load_real_space(
    num_inputs: int, bounds: tuple[float, float] = (-3, 3)
) -> tuple[Tensor, dict[str, Tensor]]:
    """
    Load real segment around 0, ensuring correct order.
    """
    real_space = torch.cat(
        [
            linspace(*bounds, num_inputs),
            torch.tensor([0.0]),
        ]
    )
    sorted_space, _ = torch.sort(real_space)
    mask_zero = sorted_space == 0
    mesh_masks = {'zero': mask_zero}
    return sorted_space, mesh_masks


def load_exponential(
    num_inputs: int,
) -> tuple[Tensor, dict[str, Tensor], int, Callable, Callable, Callable, Callable]:
    """
    Load configuration space (around 0) for exponential PDE
    """
    # Define constants
    output_dim = 1
    input_homeo = identity
    output_homeo = identity
    encoding = identity
    analytical = np.exp

    # Load space
    x, mesh_masks = load_real_space(num_inputs)

    return (
        x,
        mesh_masks,
        output_dim,
        analytical,
        input_homeo,
        output_homeo,
        encoding,
    )


def load_cosinus(
    num_inputs: int,
) -> tuple[Tensor, dict[str, Tensor], int, Callable, Callable, Callable, Callable]:
    """
    Load cosine collocation points on [-pi, pi], including the initial point.
    """
    # Define constants
    output_dim = 1
    input_homeo = identity
    output_homeo = identity
    encoding = identity
    analytical = np.cos

    # Load space
    x, mesh_masks = load_real_space(num_inputs, cosinus.TRAINING_BOUNDS)

    return (
        x,
        mesh_masks,
        output_dim,
        analytical,
        input_homeo,
        output_homeo,
        encoding,
    )


def load_laplace(
    num_inputs: int,
) -> tuple[Tensor, dict[str, Tensor], int, Callable, Callable, Callable, Callable]:
    """
    Load a square [0, 1] x [0, 1] as input space for laplace PDE.
    """
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
        'inlet': x == 0,
        'outlet': x == 1,
        'bottom': y == 0,
        'top': y == 1,
    }

    return (
        xy,
        mesh_masks,
        output_dim,
        analytical,
        input_homeo,
        output_homeo,
        encoding,
    )


def load_poiseuille(
    num_inputs: int,
) -> tuple[Tensor, dict[str, Tensor], int, Callable, Callable, Callable, Callable]:
    """Load a flat channel with primitive outputs (u, v, p)."""
    if num_inputs < 3:
        raise ValueError('Poiseuille flow requires at least 3 points per axis.')
    xy = torch.cartesian_prod(
        torch.linspace(0, poiseuille.LENGTH, num_inputs),
        torch.linspace(0, poiseuille.HEIGHT, num_inputs),
    )
    x, y = xy.unbind(dim=1)
    mesh_masks = {
        'inlet': x == 0,
        'outlet': x == poiseuille.LENGTH,
        'wall': (y == 0) | (y == poiseuille.HEIGHT),
    }
    return xy, mesh_masks, 3, poiseuille.analytical, identity, identity, identity


def load_kovasznay(
    num_inputs: int,
) -> tuple[Tensor, dict[str, Tensor], int, Callable, Callable, Callable, Callable]:
    """Sample an obstacle-free rectangle for the Re=40 velocity-pressure problem."""
    if num_inputs < 3:
        raise ValueError('Kovasznay flow requires at least 3 points per axis.')
    xy = torch.cartesian_prod(
        torch.linspace(*kovasznay.X_BOUNDS, num_inputs),
        torch.linspace(*kovasznay.Y_BOUNDS, num_inputs),
    )
    x, y = xy.unbind(dim=1)
    mesh_masks = {
        'inlet': x == kovasznay.X_BOUNDS[0],
        'outlet': x == kovasznay.X_BOUNDS[1],
        'bottom': y == kovasznay.Y_BOUNDS[0],
        'top': y == kovasznay.Y_BOUNDS[1],
    }
    return xy, mesh_masks, 3, kovasznay.analytical, identity, identity, identity


def load_wind_tunnel(
    num_inputs: int,
) -> tuple[Tensor, dict[str, Tensor], int, Callable, Callable, Callable, Callable]:
    """
    Load a square [0, 4] x [0, 1] as input space.
    """
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
        'inlet': x == 0,
        'outlet': x == 4,
        'wall': ((y == 0) | (y == 1)),
    }

    return (
        xy,
        mesh_masks,
        output_dim,
        None,
        input_homeo,
        output_homeo,
        encoding,
    )


def load_2d_mesh(
    num_inputs: int,
    plot: bool = False,
    augmented_grid: int = True,
    filepath=DEFAULT_MESH,
) -> tuple[Tensor, dict[str, Tensor], int, None, Callable, Callable, Callable]:
    """
    Loads node coordinates from a SU2 mesh file.
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
    mesh_masks = get_marker_masks(filepath, num_points)

    print(f'Total number of vertices: {xy.shape[0]}')
    if plot:
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


def load_scenario(
    scenario: str,
    num_inputs: int = 100,
) -> tuple[
    Tensor,
    dict[str, Tensor],
    int,
    Union[Callable, None],
    Callable,
    Callable,
    Callable,
]:
    """
    Function that loads the according scenario with:
        - space data
        - mesh masks (for boundaries)
        - output dimension of the PINN
        - analytical solution (= None) if applicable
        - input homeomorphism
        - output homeomorphism
        - encoding
    """
    print(f'Loading scenario: {scenario}')

    if scenario == EXPONENTIAL_SCENARIO:
        return load_exponential(num_inputs)
    elif scenario == COSINUS_SCENARIO:
        return load_cosinus(num_inputs)
    elif scenario == LAPLACE_SCENARIO:
        return load_laplace(num_inputs)
    elif scenario == KOVASZNAY_SCENARIO:
        return load_kovasznay(num_inputs)
    elif scenario == CYLINDER_SCENARIO:
        from learnpdes.fluid import CylinderProblem, sample_fluid

        if num_inputs < 3:
            raise ValueError('Cylinder flow requires at least 3 points per axis.')
        samples = sample_fluid(
            CylinderProblem(),
            num_inputs**2,
            4 * num_inputs,
            generator=torch.Generator().manual_seed(torch.initial_seed()),
        )
        groups = [samples.interior, *samples.boundary.values()]
        xy = torch.cat(groups).float()
        masks, offset = {}, len(samples.interior)
        for name, points in samples.boundary.items():
            masks[name] = torch.zeros(len(xy), dtype=torch.bool)
            masks[name][offset : offset + len(points)] = True
            offset += len(points)
        return xy, masks, 3, None, identity, identity, identity
    elif scenario == POISEUILLE_SCENARIO:
        return load_poiseuille(num_inputs)
    elif scenario in [POTENTIAL_FLOW_SCENARIO, SOLENOIDAL_FLOW_SCENARIO]:
        return load_2d_mesh(num_inputs, plot=False)
    else:
        raise ValueError(
            f'{scenario=} is not a valid scenario. '
            'Please look at the README.md '
            'for valid scenarios identifiers.'
        )
