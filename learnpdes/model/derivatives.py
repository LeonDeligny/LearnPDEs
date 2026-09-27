"""Taylor-mode input derivatives for one-dimensional Linear/Tanh networks.

Propagating Taylor coefficients avoids repeatedly differentiating an expanding
reverse-mode graph. Parameters remain in PyTorch's graph for training.
See https://docs.jax.dev/en/latest/jax.experimental.jet.html for the general
Taylor-mode approach; this implementation uses PyTorch and the tanh recurrence.
"""

from math import factorial

import torch
from torch.nn import Linear, Sequential, Tanh
from torch.nn.functional import linear


def _tanh_coefficients(
    coefficients: torch.Tensor, degrees: torch.Tensor, order: int
) -> torch.Tensor:
    values = [coefficients[0].tanh()]
    complement = [1 - values[0].square()]
    derivative = coefficients[1:] * degrees
    for degree in range(1, order + 1):
        values.append(
            (derivative[:degree] * torch.stack(complement).flip(0)).sum(0) / degree
        )
        if degree < order:
            series = torch.stack(values)
            complement.append(-(series * series.flip(0)).sum(0))
    return torch.stack(values)


def tanh_derivatives(
    network: Sequential, x: torch.Tensor, order: int
) -> list[torch.Tensor]:
    """Return f, f', ..., f^(order), with gradients through network parameters.

    Internally a[n] = f^(n)/n!. For y(t)=tanh(a(t)), y'=a'*(1-y*y).
    Equating powers of t computes each next coefficient from previous ones;
    truncation is exact for derivatives through the requested order.
    """
    if order < 0 or x.ndim != 2 or x.shape[1] != 1:
        raise ValueError(
            'Taylor derivatives require a nonnegative order and inputs (N, 1).'
        )
    coefficients = torch.stack(
        [x]
        + ([torch.ones_like(x)] if order else [])
        + [torch.zeros_like(x) for _ in range(max(0, order - 1))]
    )
    degrees = torch.arange(1, order + 1, dtype=x.dtype, device=x.device)[:, None, None]
    for layer in network:
        if isinstance(layer, Linear):
            coefficients = linear(coefficients, layer.weight)
            if layer.bias is not None:
                coefficients = torch.cat(
                    (coefficients[:1] + layer.bias, coefficients[1:])
                )
        elif isinstance(layer, Tanh):
            coefficients = _tanh_coefficients(coefficients, degrees, order)
        else:
            raise ValueError('Taylor derivatives support only Linear and Tanh layers.')
    return [value * factorial(degree) for degree, value in enumerate(coefficients)]
