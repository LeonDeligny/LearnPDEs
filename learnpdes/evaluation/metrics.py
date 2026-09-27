"""Evaluation-only exact comparisons shared by scalar and vector scenarios."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from learnpdes.types import Analytical, Array


def compare_exact(
    model: torch.nn.Module,
    coordinates: torch.Tensor,
    analytical: Analytical | None,
    *,
    components: tuple[str, ...] = (),
    diagnostics: Callable[[Array, Array], dict[str, float]] | None = None,
) -> dict[str, float]:
    if analytical is None:
        return {}
    parameter = next(model.parameters())
    # Interior exact values are used only for evaluation. Kovasznay training
    # uses exact velocity boundary values and one pressure reference.
    exact = analytical(*(column.numpy() for column in coordinates.unbind(dim=1)))
    exact = (
        np.column_stack(exact)
        if isinstance(exact, tuple)
        else np.asarray(exact)[:, None]
    )
    exact = torch.as_tensor(exact, dtype=coordinates.dtype, device=parameter.device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        try:
            prediction = model(coordinates.to(parameter))
        finally:
            model.train(was_training)
        error = prediction - exact
        metrics = {
            'rmse': error.square().mean().sqrt().item(),
            'max_error': error.abs().max().item(),
            'relative_l2': (
                torch.linalg.vector_norm(error) / torch.linalg.vector_norm(exact)
            ).item(),
        }

        for index, name in enumerate(components):
            metrics[f'{name}_rmse'] = error[:, index].square().mean().sqrt().item()
            metrics[f'{name}_max_error'] = error[:, index].abs().max().item()
            if torch.count_nonzero(exact[:, index]):
                metrics[f'{name}_relative_l2'] = (
                    torch.linalg.vector_norm(error[:, index])
                    / torch.linalg.vector_norm(exact[:, index])
                ).item()
        if diagnostics is not None:
            metrics.update(diagnostics(coordinates.numpy(), prediction.cpu().numpy()))
        return metrics
