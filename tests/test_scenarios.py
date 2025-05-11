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

from learnpdes import (
    EXPONENTIAL_SCENARIO,
    COSINUS_SCENARIO,
    LAPLACE_SCENARIO,
    POTENTIAL_FLOW_SCENARIO,
)

# ======= Tests =======


class TestScenarios(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.epochs: int = 10

    def test_first_scenario(self) -> None:
        pinn(EXPONENTIAL_SCENARIO, self.epochs, pre_epochs=0)

    def test_second_scenario(self) -> None:
        pinn(COSINUS_SCENARIO, self.epochs, pre_epochs=0)

    def test_third_scenario(self) -> None:
        pinn(LAPLACE_SCENARIO, self.epochs, pre_epochs=0)

    def test_fourth_scenario(self) -> None:
        pinn(POTENTIAL_FLOW_SCENARIO, self.epochs, pre_epochs=0)


# ======= Main =======

if __name__ == '__main__':
    main(verbosity=0)  # pragma: no cover
