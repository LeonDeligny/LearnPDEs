"""Constants and variables for the project."""

# ======= Imports =======

from typing import Final

from torch import (
    Tensor,
    manual_seed,
    pi,
    tensor,
)
from torch import device as TorchDevice
from torch.backends.mps import is_available

# ======= Constants =======

pi_tensor: Tensor = tensor(pi)

# ======= Variables =======

# Detects if Metal Performance Shaders (MPS)
# is available on your system.
device_type: str = 'mps' if is_available() else 'cpu'
# device_type: str = 'cpu'
device: Final[TorchDevice] = TorchDevice(device_type)

# Fixing seed
manual_seed(0)

# ======= Scenarios =======

EXPONENTIAL_SCENARIO: Final[str] = 'exponential'
FORCED_LINEAR_SCENARIO: Final[str] = 'forced-linear'
LOGISTIC_SCENARIO: Final[str] = 'logistic'
COSINUS_SCENARIO: Final[str] = 'cosinus'
LAPLACE_SCENARIO: Final[str] = 'laplace'
KOVASZNAY_SCENARIO: Final[str] = 'kovasznay'
CYLINDER_SCENARIO: Final[str] = 'cylinder'
CIRCULAR_COUETTE_SCENARIO: Final[str] = 'circular-couette'
POISEUILLE_SCENARIO: Final[str] = 'poiseuille'
POTENTIAL_FLOW_SCENARIO: Final[str] = 'potential flow'
SOLENOIDAL_FLOW_SCENARIO: Final[str] = 'solenoidal flow'
