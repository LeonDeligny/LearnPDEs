"""Dimensionless Kovasznay benchmark: steady incompressible Navier–Stokes.

The network predicts (u, v, p) directly, with unit density, no body force,
and kinematic viscosity 1 / Re. The rectangle contains no solid obstacle.
"""

import math

import numpy as np
import torch

REYNOLDS = 40.0
VISCOSITY = 1.0 / REYNOLDS
X_BOUNDS = (-0.5, 1.0)
Y_BOUNDS = (-0.5, 1.5)
DECAY_RATE = REYNOLDS / 2 - math.sqrt(REYNOLDS**2 / 4 + 4 * math.pi**2)


def analytical(x, y):
    """Return exact (u, v, p) arrays or tensors, retaining Torch autograd.

    Pressure uses the gauge p(0, y) = 0. Training fixes this same additive
    constant by prescribing the exact pressure at the bottom-right corner.
    """
    backend = torch if isinstance(x, torch.Tensor) else np
    x, y = (
        backend.broadcast_tensors(x, y)
        if backend is torch
        else np.broadcast_arrays(x, y)
    )
    decay = backend.exp(DECAY_RATE * x)
    phase = 2 * math.pi * y
    u = 1 - decay * backend.cos(phase)
    v = DECAY_RATE / (2 * math.pi) * decay * backend.sin(phase)
    p = (1 - decay**2) / 2
    return u, v, p
