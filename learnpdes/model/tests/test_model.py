'''
Test encodings.
'''

# ======= Imports =======

import torch

from learnpdes.utils.decorators import time
from learnpdes.model.encoding import (
    identity,
    polynomial,
)

from unittest import main
from functools import partial
from learnpdes.utils.testhelper import TestHelper

# ======= Functions =======


class TestModel(TestHelper):

    @time
    def test_encodings(self):
        self.assert_equal_function(
            f=partial(polynomial, dim=1),
            inputs=self.default_input,
            expected=self.default_input,
        )

        self.assert_equal_function(
            f=partial(polynomial, dim=2),
            inputs=self.default_input,
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
            inputs=self.default_input,
            expected=self.default_input,
        )


# ======= Main =======

if __name__ == '__main__':
    main(verbosity=0)
