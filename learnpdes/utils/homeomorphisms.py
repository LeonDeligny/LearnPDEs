'''
Homeomorphic transformations.
'''

#  ======= Imports =======

from torch import sqrt

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
    return (1 + x / sqrt(1 + x**2)) / 2.0
