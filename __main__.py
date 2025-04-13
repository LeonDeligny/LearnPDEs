'''
This script is the entry point for training a
Physics-Informed Neural Network (PINN) model.

Author:
    Leon Deligny

'''

# ======= Imports =======

from model.pinn import PINN
from model.loss import Loss
from model.trainer import Trainer

from utils.decorators import time
from utils.utility import load_scenario

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

    # Scenario (for now) can be either:
    # 'exponential'
    # 'cosinus'
    # 'laplace'
    scenario = 'exponential'

    (
        input_space, analytical,
        input_homeo, output_homeo, encoding
    ) = load_scenario(scenario, num_inputs=100)

    # Dimension of input space
    input_dim = input_space.ndimension()

    # 1. Construct model
    pinn = PINN(
        nn_params={
            'input_dim': input_dim,
            'hidden_dim': 200,
            'num_hidden_layers': 4,
        },
        input_homeo=input_homeo,
        output_homeo=output_homeo,
        encoding=encoding,
    ).to(device)

    # 2. Define loss function
    loss = Loss(
        input_space=input_space,
        forward=pinn.forward,
    )

    # 3. Train model
    trainer = Trainer(
        model_params=pinn.parameters,
        loss=loss.get_loss(scenario),
        training_params={
            'learning_rate': 0.001,
            'nb_epochs': 10_000,
        },
        dim_plot=input_dim,
        analytical=analytical,
    )

    trainer.train()

    # 4. Evaluate model
    # TODO: Evaluate model
    # Look at convergence of the loss function
    # Look at extremas of the analytical solution to create
    # a 'test' set


if __name__ == '__main__':
    main()
