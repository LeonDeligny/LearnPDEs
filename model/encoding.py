'''
Encodings for PINNs (pre-layers).

Examples of encodings:
    1. Polynomial encoding, E(x) = (x, x^2, x^3,..., x^n).-
    2. Fourier encoding, E(x) = (cos(<f, x>), sin(<f, x>)).
    3. TODO: Add more encodings with respective sources.

'''

# ======= Imports =======

import torch

from torch import Tensor
from torch.nn import Parameter

from torch import (
    cat,
    cos,
    sin,
)

from numpy import pi
from __init__ import device

# ======= Functions =======


def identity(x: Tensor) -> Tensor:
    return x


def complex_projection(x: Tensor) -> Tensor:
    '''encoding(x) = (x, ix)'''
    return cat([x, 1j * x], dim=1)


def real_projection(x: Tensor) -> Tensor:
    return x.real + x.imag


def polynomial(x: Tensor, dim: int) -> Tensor:
    '''encoding(x) = (x, x^2, x^3, ..., x^input_dim)'''
    return cat([x.view(-1, 1)**i for i in range(1, dim + 1)], dim=1)


def fourier(x: Tensor, dim: int = 10, scale: float = 1.0) -> Tensor:
    '''
    encoding(x) = (cos(2 pi <f, x>), sin(2 pi <f, x>))
    f is a learnable parameter, stands for frequency.
    In 1D, <f, x> = f * x is in the range [0, 1].
    '''

    f = torch.randn(x.numel(), dim // 2).to(device)
    kernel = scale * Parameter(f, requires_grad=True)
    x_proj = pi * x @ kernel

    return cat([cos(x_proj), sin(x_proj)], dim=1)

# ======= Main =======


if __name__ == '__main__':
    print('Nothing to execute.')
