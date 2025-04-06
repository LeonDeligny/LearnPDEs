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

from model.pinn import PINN
from model.loss import Loss
from model.trainer import Trainer
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
def main() -> None:
    '''
    Description of workflow.
        1. Construct model.
        2. Train model.
        3. Evaluate model ?
    '''

    # 1. Construct model
    pinn = PINN(
        nn_params={
            'hidden_dim': 100,
            'num_hidden_layers': 4,
        },
        input_homeo=identity,
        output_homeo=identity,
        encoding=identity,
    ).to(device)

    # 2. Define loss function
    loss = Loss(
        x=torch.cat([
            linspace(-3, 3, 10_000).view(-1, 1),
            torch.tensor([0.0]).view(-1, 1),
        ]).unique(dim=0).sort(dim=0).values,
        forward=pinn.forward,
    )

    # 3. Train model
    trainer = Trainer(
        model_params=pinn.parameters,
        loss=loss.exponential_loss,
        training_params={
            'learning_rate': 0.001,
            'nb_epochs': 10_000,
        },
        analytical=np.exp,
    )

    trainer.train()

    # 4. Evaluate model
    # TODO: Evaluate model
    # Look at convergence of the loss function
    # Look at extremas of the analytical solution to create
    # a "test" set


if __name__ == '__main__':
    main()
