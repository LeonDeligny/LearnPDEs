'''
Test utils module.
'''

# ======= Imports =======

from torch import tensor
from learnpdes.utils.decorators import time
from learnpdes.utils.homeomorphisms import input_homeo

from unittest import main
from learnpdes.utils.utility import Identity
from learnpdes.utils.testhelper import TestHelper

# ======= Functions =======


class TestUtils(TestHelper):

    @time
    def test_identity(self):
        id_class = Identity()
        self.assert_equal_function(
            f=id_class.forward,
            inputs=self.default_input,
            expected=self.default_input,
        )

    @time
    def test_input_homeomorphism(self):
        self.assert_equal_function(
            f=input_homeo,
            inputs=tensor([0.0]),
            expected=tensor([0.5]),
        )


# ======= Main =======

if __name__ == '__main__':
    main(verbosity=0)  # pragma: no cover
