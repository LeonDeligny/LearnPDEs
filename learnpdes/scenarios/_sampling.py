"""Basic coordinate sampling shared by scenario definitions."""

import torch
from torch import Tensor, linspace


def load_real_space(
    num_inputs: int, bounds: tuple[float, float] = (-3, 3)
) -> tuple[Tensor, dict[str, Tensor]]:
    """Load real segment around 0, ensuring correct order."""
    real_space = torch.cat(
        [
            linspace(*bounds, num_inputs),
            torch.tensor([0.0]),
        ]
    )
    sorted_space, _ = torch.sort(real_space)
    mask_zero = sorted_space == 0
    mesh_masks = {'zero': mask_zero}
    return sorted_space, mesh_masks
