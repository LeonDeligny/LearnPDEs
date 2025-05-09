'''
Helper class with default values for testing.
'''

#  ======= Imports =======

from torch import (
    tensor,
    allclose,
)

from torch import Tensor
from typing import Callable
from functools import partial
from unittest import TestCase

from learnpdes import device

# ======= Class =======


class TestHelper(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.default_input = tensor([
            [-10.0], [-1.0], [0.0], [1.0], [10.0]
        ]).to(device)
        cls.device = 'cpu'

    def assert_equal_function(
        self: 'TestHelper',
        f: Callable[[Tensor], Tensor],
        inputs: Tensor,
        expected: Tensor,
    ) -> None:
        actual = f(inputs).to(self.device)
        if actual.requires_grad:
            actual = actual.detach()
        expected = expected.to(self.device)
        self.assertTrue(
            allclose(actual, expected, atol=1e-4),
            f'Expected {expected}, but got {actual}'
        )
        if isinstance(f, partial):
            print(
                f'Partial function test {f.func.__name__} '
                f'passed with constant argument(s): {f.keywords}'
            )
        else:
            print(f'Function test {f.__name__} passed.')
