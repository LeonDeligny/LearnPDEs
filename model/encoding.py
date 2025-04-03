'''
Encodings for PINNs (pre-layers).

Examples of encodings:
    1. Polynomial encoding, E(x) = (x, x^2, x^3,..., x^n).-
    2. Fourier encoding, E(x) = (cos(<f, x>), sin(<f, x>)).
    3. TODO: Add more encodings and sources.

'''

# ======= Imports =======

import math
import torch

from torch import Tensor
from torch.nn import Parameter

from utils.utility import num_inputs
from torch import (
    cat,
    cos,
    sin,
)

from numpy import pi

# ======= Functions =======


def polynomial(x: Tensor, input_dim: int) -> Tensor:
    '''encoding(x) = (x, x^2, x^3, ..., x^input_dim)'''

    return cat([x.view(-1, 1)**i for i in range(1, input_dim + 1)], dim=1)


def fourier(x: Tensor, dim: int = 10, scale: float = 1.0) -> Tensor:
    '''
    encoding(x) = (cos(2 pi <f, x>), sin(2 pi <f, x>))
    f is a learnable parameter.
    In 1D, <f, x> = f * x is in the range [0, 1].
    '''

    kernel = scale * Parameter(torch.randn(num_inputs(x), dim // 2), requires_grad=True)
    x_proj = pi * x @ kernel

    return cat([cos(x_proj), sin(x_proj)], dim=1)

# ======= Main =======


if __name__ == '__main__':
    print('Nothing to execute.')
