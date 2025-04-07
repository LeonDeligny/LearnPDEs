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
# import numpy as np

from torch import Tensor
from model.pinn import PINN
from model.loss import Loss
from model.trainer import Trainer
# from functools import partial

# from torch import linspace
from utils.decorators import time
from utils.utility import laplace_function
# from utils.homeomorphisms import input_homeo, output_homeo
from model.encoding import (
    identity,
    # fourier,
)
from __init__ import device

# ======= Main =======


@time
def main() -> None:
    '''
    Description of workflow.
        1. Construct model.
        2. Train model.
        3. Evaluate model ?
    '''

    # Set
    num_inputs: int = 800

    # 1D Space (mimicking Real space)
    # x = torch.cat([
    #         linspace(-3, 3, num_inputs),
    #         torch.tensor([0.0]),
    # ])

    # 2D Space [0, 1] x [0, 1]
    xy = torch.cartesian_prod(
        torch.linspace(0, 1, num_inputs),
        torch.linspace(0, 1, num_inputs),
    )

    # Define the spaces to use
    input_space: Tensor = xy
    input_dim: int = input_space.ndimension()

    # 1. Construct model
    pinn = PINN(
        nn_params={
            'input_dim': input_dim,
            'hidden_dim': 100,
            'num_hidden_layers': 4,
        },
        input_homeo=identity,
        output_homeo=identity,
        encoding=identity,
    ).to(device)

    # 2. Define loss function
    loss = Loss(
        input_space=input_space,
        forward=pinn.forward,
    )

    # 3. Train model
    trainer = Trainer(
        model_params=pinn.parameters,
        loss=loss.laplace_loss,
        training_params={
            'learning_rate': 0.001,
            'nb_epochs': 10_000,
        },
        analytical=laplace_function,
    )

    trainer.train()

    # 4. Evaluate model
    # TODO: Evaluate model
    # Look at convergence of the loss function
    # Look at extremas of the analytical solution to create
    # a 'test' set


if __name__ == '__main__':
    main()
