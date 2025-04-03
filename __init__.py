'''
Constants and variables for the project.
'''

# ======= Imports =======

import torch

# ======= Variables =======

# Detects if MPS is available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ======= Main =======


if __name__ == '__main__':
    print('Nothing to execute.')
