'''
Utility functions.
'''

# ======= Imports =======

import torch
import numpy as np

from torch import linspace
from pydantic import validate_call
from learnpdes.model.encoding import (
    identity,
    # complex_projection,
    # real_projection,
    # fourier,
)

from torch import Tensor
from numpy import ndarray
from torch.nn import Module
from typing import (
    Tuple,
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


@validate_call
def load_scenario(
    scenario: Literal['exponential', 'cosinus', 'laplace'],
    num_inputs: int = 100,
) -> Tuple[Tensor, Callable, Callable, Callable, Callable]:

    print(f'Loading scenario: {scenario}')

    if scenario == 'exponential':
        return load_exponential(num_inputs)
    elif scenario == 'cosinus':
        return load_cosinus(num_inputs)
    elif scenario == 'laplace':
        return load_laplace(num_inputs)
