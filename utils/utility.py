'''
Utility functions.
'''

from torch import Tensor

# ======= Functions =======


def num_inputs(x: Tensor) -> int:
    return x.shape[1] if len(x.shape) > 1 else 1

# ======= Main =======


if __name__ == '__main__':
    print('Nothing to execute.')
