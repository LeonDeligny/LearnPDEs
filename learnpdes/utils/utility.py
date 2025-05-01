'''
Utility functions.
'''

# ======= Imports =======

import gmsh
import torch
import numpy as np

from torch import (
    # load,
    linspace,
)
# from learnpdes.utils.decorators import validate
from learnpdes.model.encodings import (
    identity,
    # fourier,
)
from learnpdes.utils.meshing import (
    generate_mesh,
    gmsh_to_tensor,
    tag_surfaces_to_meshes,
    set_variable_mesh_sizes,
    add_physical_fuild_marker,
)

from torch import Tensor
from numpy import ndarray
from torch.nn import Module
from typing import (
    Tuple,
    Union,
    Literal,
    Callable,
)

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


def load_real_space(num_inputs: int) -> Tensor:
    '''
    Load real segment around 0, ensuring correct order.
    '''
    real_space = torch.cat([
        linspace(-3, 3, num_inputs),
        torch.tensor([0.0]),
    ])
    sorted_space, _ = torch.sort(real_space)
    return sorted_space


def load_exponential(
    num_inputs: int
) -> Tuple[Tensor, Callable, Callable, Callable, Callable]:
    '''
    Load configuration space (around 0) for exponential PDE
    '''
    x = load_real_space(num_inputs)
    input_homeo = identity
    output_homeo = identity
    encoding = identity
    analytical = np.exp

    return x, analytical, input_homeo, output_homeo, encoding


def load_cosinus(
    num_inputs: int
) -> Tuple[Tensor, Callable, Callable, Callable, Callable]:
    '''
    Load configuration space (around 0) for exponential PDE
    '''
    x = load_real_space(num_inputs)
    input_homeo = identity
    output_homeo = identity
    encoding = identity
    analytical = np.cos

    return x, analytical, input_homeo, output_homeo, encoding


def load_laplace(
    num_inputs: int
) -> Tuple[Tensor, Callable, Callable, Callable, Callable]:
    # 2D Space [0, 1] x [0, 1]
    xy = torch.cartesian_prod(
        torch.linspace(0, 1, num_inputs),
        torch.linspace(0, 1, num_inputs),
    )

    input_homeo = identity
    output_homeo = identity
    encoding = identity
    analytical = laplace_function

    return xy, analytical, input_homeo, output_homeo, encoding


def load_potential_flow(
) -> Tuple[Tensor, None, Callable, Callable, Callable]:
    '''
    Source:
        https://jsdokken.com/dolfinx-tutorial/chapter2/ns_code2.html
    '''
    gmsh.initialize()

    # Define constants
    L = 1.0
    H = 1.0
    CENTER_X = 0.2
    CENTER_Y = 0.5
    RADIUS = 0.05
    GDIM = 2
    rectangle = gmsh.model.occ.addRectangle(0, 0, 0, L, H, tag=1)
    obstacle = gmsh.model.occ.addDisk(CENTER_X, CENTER_Y, 0, RADIUS, RADIUS)

    # Do not mesh interior of the disk
    gmsh.model.occ.cut([(GDIM, rectangle)], [(GDIM, obstacle)])
    gmsh.model.occ.synchronize()

    volumes = add_physical_fuild_marker(dim=GDIM)
    tag_surfaces_to_meshes(volumes, length=L, height=H)
    set_variable_mesh_sizes(obstacle, radius=RADIUS, height=H)
    generate_mesh(gdim=GDIM)
    xy = gmsh_to_tensor()

    input_homeo = identity
    output_homeo = identity
    encoding = identity
    analytical = None

    import matplotlib.pyplot as plt
    plot = False
    if plot:
        plt.figure(figsize=(6, 6))
        plt.scatter(xy[:, 0], xy[:, 1], s=2, alpha=0.7)
        plt.gca().set_aspect('equal')
        plt.title("Gmsh-generated mesh nodes")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    return xy, analytical, input_homeo, output_homeo, encoding


def load_scenario(
    scenario: Literal[
        'exponential',
        'cosinus',
        'laplace',
        'potential flow'
    ],
    num_inputs: int = 100,
) -> Tuple[Tensor, Union[Callable, None], Callable, Callable, Callable]:

    print(f'Loading scenario: {scenario}')

    if scenario == 'exponential':
        return load_exponential(num_inputs)
    elif scenario == 'cosinus':
        return load_cosinus(num_inputs)
    elif scenario == 'laplace':
        return load_laplace(num_inputs)
    elif scenario == 'potential flow':
        return load_potential_flow()


def detach_to_numpy(
    f: Union[Tuple[Tensor, ...], Tensor]
) -> Union[Tuple[ndarray, ...], ndarray]:
    """
    Essentially sends to cpu, detach, and converts to ndarray.
    Applies .cpu().detach().numpy() on object.
    """
    if isinstance(f, tuple):
        return tuple(fi.cpu().detach().numpy() for fi in f)
    else:
        return f.cpu().detach().numpy()
