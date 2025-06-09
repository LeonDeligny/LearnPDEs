'''
Constants and variables for the project.
'''

# ======= Imports =======

from torch.backends.mps import is_available
from torch import (
    tensor,
    manual_seed,
)

from torch import Tensor
from torch import device as TorchDevice

from torch import pi

# ======= Constants =======

pi_tensor: Tensor = tensor(pi)

# ======= Variables =======

# Detects if Metal Performance Shaders (MPS)
# is available on your system.
device_type: str = (
    'mps'
    if is_available()
    else 'cpu'
)
# device_type: str = 'cpu'
device = TorchDevice(device_type)
print(f'Using {device_type=}.')

# Fixing seed
manual_seed(0)

# ======= Scenarios =======

EXPONENTIAL_SCENARIO = 'exponential'
COSINUS_SCENARIO = 'cosinus'
LAPLACE_SCENARIO = 'laplace'
POTENTIAL_FLOW_SCENARIO = 'potential flow'
