'''
Utility functions.
'''

# ======= Imports =======

import re
import torch
import numpy as np

from torch import linspace
from learnpdes.model.encodings import identity
from learnpdes.utils.utility import (
    analyze_xy,
    laplace_function,
    get_marker_masks,
)

from torch import Tensor
from typing import (
    Dict,
    Tuple,
    Union,
    Callable,
)

from learnpdes import (
    EXPONENTIAL_SCENARIO,
    COSINUS_SCENARIO,
    LAPLACE_SCENARIO,
    POTENTIAL_FLOW_SCENARIO,
)

# ======= Functions =======


def load_real_space(
    num_inputs: int
) -> Tuple[Tensor, Dict[str, Tensor]]:
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
) -> Tuple[
    Tensor,
    Dict[str, Tensor],
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
) -> Tuple[
    Tensor,
    Dict[str, Tensor],
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
) -> Tuple[
    Tensor, Dict[str, Tensor],
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


def load_potential_flow() -> Tuple[
    Tensor, Dict[str, Tensor],
    int, None,
    Callable, Callable, Callable
]:
    '''
    Loads node coordinates from a SU2 mesh file.
    Returns as a tensor of shape [N, 2].
    '''
    # Define constants
    output_dim = 3
    input_homeo = identity
    output_homeo = identity
    encoding = identity
    analytical = None

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
    analyze_xy(xy)
    num_points = xy.shape[0]
    mesh_masks = get_marker_masks(filepath, num_points)

    print(f"Total number of vertices: {num_points}")

    return (
        xy, mesh_masks,
        output_dim, analytical,
        input_homeo, output_homeo, encoding,
    )


def load_scenario(
    scenario: str,
    num_inputs: int = 100,
) -> Tuple[
    Tensor, Dict[str, Tensor],
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
        return load_potential_flow()
    else:
        raise ValueError(
            f'{scenario=} is not a valid scenario. '
            'Please look at the README.md '
            'for valid scenarios identifiers.'
        )
