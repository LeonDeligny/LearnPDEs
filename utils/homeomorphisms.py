'''
Homeomorphic transformations.
'''

#  ======= Imports =======


import torch

from torch import Tensor

# ======= Functions =======


def input_homeo(x: Tensor) -> Tensor:
    '''
    Apply a homeomorphic transformation:
        homeo: R -> ]0, 1[
        homeo(x) = (1 + x / sqrt(1 + x^2) ) / 2
        homeo(-infty) = 0
        homeo(infty) = 1
    '''
    return (1 + x / torch.sqrt(1 + x**2)) / 2.0


def output_homeo(x: Tensor) -> Tensor:
    '''
    Apply a homeomorphic transformation:
        homeo: ]-1, 1[ -> ]0, infty[
        homeo(-1) = 0
        homeo(1) = infty
    '''
    raise NotImplementedError('Output homeomorphism not implemented.')

# ======= Main =======


if __name__ == '__main__':
    print('Nothing to execute.')
