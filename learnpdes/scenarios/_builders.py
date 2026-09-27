"""Shared model assembly; the scenario supplies its data and objective class."""

from collections.abc import Callable

import torch
from torch import Tensor

from learnpdes import device
from learnpdes.config import RunConfig
from learnpdes.model.objectives import CollocationObjective
from learnpdes.model.pinn import PINN
from learnpdes.types import Analytical


def build_collocation(
    config: RunConfig,
    data: tuple[
        Tensor,
        dict[str, Tensor],
        int,
        Analytical | None,
        Callable[[Tensor], Tensor],
        Callable[[Tensor], Tensor],
        Callable[[Tensor], Tensor],
    ],
    objective_type: type[CollocationObjective],
    *,
    formulation: str | None = None,
) -> tuple[PINN, CollocationObjective, Analytical | None]:
    if config.points is None or config.hidden_dim is None:
        raise ValueError('Resolved configuration is missing model dimensions.')
    inputs, masks, outputs, analytical, input_homeo, output_homeo, encoding = data
    input_dim = 1 if inputs.ndim == 1 else inputs.shape[1]
    model = PINN(
        nn_params={
            'input_dim': input_dim,
            'hidden_dim': config.hidden_dim,
            'output_dim': outputs,
            'num_hidden_layers': config.hidden_layers,
            'activation': torch.nn.Tanh,
        },
        input_homeo=input_homeo,
        output_homeo=output_homeo,
        encoding=encoding,
    ).to(device)
    objective = objective_type(
        scenario=formulation or config.scenario,
        input_space=inputs,
        input_dim=input_dim,
        forward=model.forward,
        mesh_masks=masks,
        cosinus_order=config.cosinus_order,
        cosinus_derivatives=model.input_derivatives
        if config.cosinus_order >= 8
        else None,
    )
    return model, objective, analytical
