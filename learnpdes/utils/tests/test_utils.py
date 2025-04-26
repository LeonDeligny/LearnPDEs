'''
Test utils module.
'''

# ======= Imports =======

from learnpdes.utils.decorators import time

from unittest import main
from learnpdes.utils.utility import Identity
from learnpdes.utils.testhelper import TestHelper

# ======= Functions =======


class TestUtils(TestHelper):

    @time
    def test_identity(self):
        self.assert_equal_function(
            f=Identity().forward,
            inputs=self.default_input,
            expected=self.default_input,
        )


# ======= Main =======
# pragma: no cover
if __name__ == '__main__':
    main(verbosity=0)
