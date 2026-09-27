"""Plane Poiseuille benchmark in nondimensional model units.

Stationary plates bound 0 <= y <= H; pressure decreases along 0 <= x <= L.
The reference is for evaluation only, never a velocity target during training.
"""

import numpy as np

LENGTH = 4.0
HEIGHT = 1.0
DENSITY = 1.0
VISCOSITY = 0.1  # Dynamic viscosity mu, not kinematic viscosity nu = mu / rho.
PRESSURE_DROP = 3.2
OUTLET_PRESSURE = 0.0
INLET_PRESSURE = OUTLET_PRESSURE + PRESSURE_DROP


def analytical(x, y) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (u, v, p): u = G y(H-y)/(2 mu), p = p_out + G(L-x)."""
    x, y = np.broadcast_arrays(x, y)
    forcing = PRESSURE_DROP / LENGTH  # G = -dp/dx > 0.
    u = forcing * y * (HEIGHT - y) / (2 * VISCOSITY)
    v = np.zeros_like(u)
    p = OUTLET_PRESSURE + forcing * (LENGTH - x)
    return u, v, p
