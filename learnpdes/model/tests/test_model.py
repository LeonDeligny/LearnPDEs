'''
Test model module.
'''

# ======= Imports =======

from torch import tensor
from learnpdes.utils.decorators import time
from learnpdes.model.encodings import (
    fourier,
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
            expected=tensor([
                [-10.0, 100.0],
                [-1.0, 1.0],
                [0.0, 0.0],
                [1.0, 1.0],
                [10.0, 100.0],
            ])
        )

        self.assert_equal_function(
            f=identity,
            inputs=self.default_input,
            expected=self.default_input,
        )

        self.assert_equal_function(
            f=partial(fourier, dim=2, scale=1.0),
            inputs=self.default_input,
            expected=tensor([
                [
                    -0.2791, -0.9788, 0.7861, 0.5472, -0.8841,
                    0.9603, 0.2050, -0.6181, 0.8370, 0.4673
                ],
                [
                    0.1284, 0.6044, 0.8464, -0.2133, -0.9650,
                    0.9917, 0.7967, 0.5326, -0.9770, -0.2624
                ],
                [
                    1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000
                ],
                [
                    0.1284, 0.6044, 0.8464, -0.2133, -0.9650,
                    -0.9917, -0.7967, -0.5326, 0.9770, 0.2624
                ],
                [
                    -0.2791, -0.9788, 0.7861, 0.5472, -0.8841,
                    -0.9603, -0.2050, 0.6181, -0.8370, -0.4673
                ]
            ]),
        )


# ======= Main =======

if __name__ == '__main__':
    main(verbosity=0)
