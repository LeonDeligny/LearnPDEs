"""Cosine training domain and evaluation-only extrapolation diagnostics."""

import numpy as np

TRAINING_BOUNDS = (-np.pi, np.pi)
EVALUATION_BOUNDS = (-3 * np.pi, 3 * np.pi)
EVALUATION_POINTS = 601
REGION_LABELS = {
    'inside': 'Inside [−π, π]',
    'outside': 'Outside: π < |x| ≤ 2π',
    'outside_3pi': 'Outside to 3π: π < |x| ≤ 3π',
    'far_outside': 'Far outside: 2π < |x| ≤ 3π',
    'full': 'Full [−2π, 2π]',
    'full_3pi': 'Full [−3π, 3π]',
}
REGION_TITLES = {
    'inside': 'Inside MSE',
    'outside': 'Outside to 2π MSE',
    'full': 'Full ±2π MSE',
    'outside_3pi': 'Outside to 3π MSE',
    'far_outside': 'Far band MSE',
    'full_3pi': 'Full ±3π MSE',
}


def validate_order(order: int) -> None:
    if isinstance(order, bool) or not isinstance(order, int) or order < 2 or order % 2:
        raise ValueError('Cosinus derivative order must be an even integer >= 2.')


def region_masks(coordinates) -> dict[str, np.ndarray]:
    """Separate the near and far bands; tolerate float32 interval endpoints."""
    x = np.asarray(coordinates).ravel()
    tolerance = 4 * np.finfo(np.float32).eps
    inside = np.abs(x) <= np.pi + tolerance
    full = np.abs(x) <= 2 * np.pi + tolerance
    extended = np.abs(x) <= 3 * np.pi + tolerance
    masks = {'inside': inside, 'outside': full & ~inside, 'full': full}
    # Old recordings only reach ±2π. Do not label their truncated interval ±3π.
    if (extended & ~full).any():
        masks.update(
            outside_3pi=extended & ~inside,
            far_outside=extended & ~full,
            full_3pi=extended,
        )
    return masks


def region_mse(coordinates, prediction) -> dict[str, float]:
    """Compare to cos(x), never supplying these targets to the optimizer."""
    x = np.asarray(coordinates).ravel()
    values = np.asarray(prediction).ravel()
    if values.shape != x.shape:
        raise ValueError('Cosinus predictions must match the evaluation coordinates.')
    error = (values - np.cos(x)) ** 2
    return {
        region: float(error[mask].mean())
        for region, mask in region_masks(x).items()
        if mask.any()
    }
