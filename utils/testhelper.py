'''
Helper class with default values for testing.
'''

#  ======= Imports =======

import torch

from torch import Tensor
from typing import Callable
from functools import partial
from unittest import TestCase

# ======= Class =======


class TestHelper(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.default_input = torch.tensor([
            [-10.0], [-1.0], [0.0], [1.0], [10.0]
        ])

    def assert_equal_function(
        self: 'TestHelper',
        f: Callable[[Tensor], Tensor],
        input: Tensor,
        expected: Tensor,
    ) -> None:
        actual = f(input)
        assert torch.allclose(
            actual,
            expected
        ), f'Expected {expected}, but got {actual}'
        if isinstance(f, partial):
            f_name = f.func.__name__
            constant_args = f.keywords
            print(
                f'Partial function test {f_name} '
                f'passed with constant argument(s): {constant_args}'
            )
        else:
            f_name = f.__name__
            print(f'Function test {f_name} passed.')

# ======= Main =======


if __name__ == '__main__':
    print('Nothing to execute.')
