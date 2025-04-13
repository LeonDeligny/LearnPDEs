'''
Test encodings.
'''

# ======= Imports =======

import torch
import unittest

from utils.decorators import time
from model.encoding import (
    identity,
    polynomial,
)

from functools import partial
from utils.testhelper import TestHelper

# ======= Functions =======


class TestModel(TestHelper):

    @time
    def test_encodings(self):
        self.assert_equal_function(
            f=partial(polynomial, dim=1),
            input=self.default_input,
            expected=self.default_input,
        )

        self.assert_equal_function(
            f=partial(polynomial, dim=2),
            input=self.default_input,
            expected=torch.tensor(
                [
                    [-10.0, 100.0],
                    [-1.0, 1.0],
                    [0.0, 0.0],
                    [1.0, 1.0],
                    [10.0, 100.0],
                ]
            )
        )

        self.assert_equal_function(
            f=identity,
            input=self.default_input,
            expected=self.default_input,
        )


# ======= Main =======

if __name__ == '__main__':
    unittest.main(verbosity=0)
