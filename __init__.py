'''
Constants and variables for the project.
'''

# ======= Imports =======

from torch.backends.mps import is_available

from torch import device as TorchDevice

# ======= Variables =======

# Detects if Metal Performance Shaders (MPS)
# is available on your system.
device_type: str = (
    "mps"
    if is_available()
    else "cpu"
)
device = TorchDevice(device_type)

print(f"Using device: {device_type}.")

# ======= Main =======


if __name__ == '__main__':
    print('Nothing to execute.')
