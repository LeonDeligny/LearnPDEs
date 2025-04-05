'''
This script is the entry point for training a
Physics-Informed Neural Network (PINN) model.

Author:
    Leon Deligny

Scenarios:
    1. Exponential ODE, Analytical solution: f = exp
        f' = f, f(0) = 1.

        We only assume that:
            - f > 0 on R
            - f is increasing on R
            - lim_x->-infty f(x) = 0
            - lim_x->infty f(x) = infty
            - f is C^infty on R

'''

# ======= Imports =======

import torch
import numpy as np

from typing import Dict
from torch import Tensor
from model.pinn import PINN
from typing import Callable
# from functools import partial

from utils.decorators import time
from torch import linspace
# from utils.homeomorphisms import input_homeo, output_homeo
from model.encoding import (
    identity,
    # fourier,
)
from __init__ import device

# ======= Main =======


@time
def main(
    input_space: Tensor = torch.cat([
        linspace(-3, 3, 10_000).view(-1, 1), 
        torch.tensor([0.0]).view(-1, 1),
    ]).unique(dim=0).sort(dim=0).values,
    nn_params: Dict = {
        'hidden_dim': 100,
        'num_hidden_layers': 4,
    },
    training_params: Dict = {
        'learning_rate': 0.001,
        'nb_epochs': 10_000,
    },
    loss_func_name: str = 'cosinus_loss',
    input_homeo: Callable[[Tensor], Tensor] = identity,
    output_homeo: Callable[[Tensor], Tensor] = identity,
    # partial(fourier, dim=10, scale=1.0),
    encoding: Callable[[Tensor], Tensor] = identity,
    analytical = np.cos,
) -> None:
    '''
    Description of workflow.
        1. Construct model.
        2. Train model.
        3. Evaluate model ?
    '''

    # 1. Construct model
    model = PINN(
        input_space=input_space,
        nn_params=nn_params,
        training_params=training_params,
        loss_func_name=loss_func_name,
        input_homeo=input_homeo,
        output_homeo=output_homeo,
        encoding=encoding,
        analytical=analytical,
    ).to(device)

    # 2. Train model
    model.train()

    # 3. Evaluate model
    # TODO: Evaluate model


if __name__ == '__main__':
    main()
