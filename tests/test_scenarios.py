'''
Testing the integration of the learnpdes module.

Test out the following scenarios:
    1. exponential
    2. cosinus
    3. laplace
'''

# ======= Imports =======

from learnpdes.main import main as pinn

from unittest import (
    main,
    TestCase,
)

# ======= Tests =======


class TestScenarios(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.epochs: int = 10

    def test_first_scenario(self) -> None:
        pinn('exponential', self.epochs)

    def test_second_scenario(self) -> None:
        pinn('cosinus', self.epochs)

    def test_third_scenario(self) -> None:
        pinn('laplace', self.epochs)


# ======= Main =======


if __name__ == '__main__':
    main(verbosity=0)
