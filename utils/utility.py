'''
Utility functions.
'''

# ======= Imports =======

import torch

from torch import Tensor

from __init__ import pi_tensor

# ======= Functions =======


def laplace_function(x: Tensor, y: Tensor) -> Tensor:
    '''
    Laplace function.
    '''
    pi: Tensor = pi_tensor
    return torch.sin(pi * x) * torch.sinh(pi * y) / torch.sinh(pi)


# ======= Main =======


if __name__ == '__main__':
    print('Nothing to execute.')
