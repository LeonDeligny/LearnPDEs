'''
Constants and variables for the project.
'''

# ======= Imports =======

import torch

# ======= Variables =======

# Detects if Metal Performance Shaders (MPS)
# is available on your system.
device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using device: {device}.")

# ======= Main =======


if __name__ == '__main__':
    print('Nothing to execute.')
