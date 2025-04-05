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

from typing import Dict
from torch import Tensor, profiler
from model.model import PINN

from torch import linspace

from __init__ import device

# ======= Main =======


def main(
    input_space: Tensor = linspace(-4, 4, 10_000).view(-1, 1),  # [n, 1]
    nn_params: Dict = {
        'hidden_dim': 400,
        'num_hidden_layers': 6,
    },
    training_params: Dict = {
        'learning_rate': 0.001,
        'nb_epochs': 10_000,
    },
    loss_func_name: str = 'exponential_loss',
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
    ).to(device)

    # 2. Train model
    with profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.MPS,
        ],
        on_trace_ready=profiler.tensorboard_trace_handler('./log'),
        record_shapes=True,
        with_stack=True
    ) as prof:
        model.train()

    print(
        prof.key_averages().table(
            sort_by="cuda_time_total"
            if device == "mps"
            else "cpu_time_total")
    )

    # 3. Evaluate model
    # TODO: Evaluate model


if __name__ == '__main__':
    main()
