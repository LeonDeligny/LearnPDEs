'''
This script is the entry point for training a
Physics-Informed Neural Network (PINN) model.
'''

# ======= Imports =======

from learnpdes.utils.decorators import time
from learnpdes.utils.loadscenarios import load_scenario

from torch.nn import Tanh
from learnpdes.model.pinn import PINN
from learnpdes.model.loss import Loss
from learnpdes.model.trainer import Trainer

from learnpdes import device

# ======= Main =======


@time
def main(
    scenario: str,
    epochs: int = 10_000,
) -> None:
    '''
    Description of workflow.
        1. Construct model.
        2. Train model.
        3. Evaluate model ?
    '''

    # Scenarios can be either:
    # 'exponential'
    # 'cosinus'
    # 'laplace'
    # 'potential flow'

    (
        input_space, mesh_masks,
        analytical, output_dim,
        input_homeo, output_homeo, encoding
    ) = load_scenario(scenario, num_inputs=100)

    # Dimension of input space
    input_dim = input_space.ndimension()

    # 1. Construct model
    # TODO: Make the model converge without using
    # an activation function that is a hint for the solution
    pinn = PINN(
        nn_params={
            'input_dim': input_dim,
            'hidden_dim': 200,
            'output_dim': output_dim,
            'num_hidden_layers': 4,
            'activation': Tanh,
        },
        input_homeo=input_homeo,
        output_homeo=output_homeo,
        encoding=encoding,
    ).to(device)

    # 2. Define loss function
    loss = Loss(
        input_space=input_space,
        input_dim=input_dim,
        forward=pinn.forward,
        mesh_masks=mesh_masks,
    )

    # 3. Train model
    trainer = Trainer(
        model_params=pinn.parameters,
        loss=loss.get_loss(scenario),
        training_params={
            'learning_rate': 0.001,
            'epochs': epochs,
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
    main(scenario='exponential')  # pragma: no cover
