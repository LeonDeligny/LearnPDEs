'''
Training configuration of the PINN model.
'''

# ======= Imports =======

import os

from learnpdes.utils.utility import detach_to_numpy
from learnpdes.utils.plot import (
    save_plot,
    create_gif,
    save_2d_plot,
    save_airfoil_plot,
    ensure_directory_exists,
)

from torch import Tensor
from torch.optim import Adam
from torch.nn import Parameter
from typing import (
    Dict,
    Union,
    Tuple,
    Callable,
    Iterator,
)

from learnpdes import device

# ======= Class =======


class Trainer:
    device = device

    def __init__(
        self,
        model_params: Iterator[Parameter],
        loss: Callable[[], Tuple[Tensor, Tensor, Tensor]],
        training_params: Dict,
        dim_plot: int,
        analytical: Union[Callable, None] = None,
    ) -> None:
        '''
        Initialiyation of training process.
        '''

        # Model parameters
        self.model_params = model_params

        # Loss function to use
        self.loss = loss

        # Training parameters
        self.learning_rate: float = training_params.get('learning_rate')
        self.nb_epochs: int = training_params.get('epochs')

        self.dim_plot = dim_plot

        # Analytical solution if any
        self.analytical = analytical

        # Training parameters
        self.optimizer = Adam(
            self.model_params(),
            lr=self.learning_rate,
        )

        # Gif parameters
        os.makedirs('gifs', exist_ok=True)

    def train(self) -> None:
        # Train if there is at least one epoch
        if self.nb_epochs != 0:
            output_dir = ensure_directory_exists()

            # Training loop
            for epoch in range(0, self.nb_epochs):
                self.optimizer.zero_grad()

                loss, inputs, f, airfoil_mask = self.loss()

                loss.backward(retain_graph=True)
                self.optimizer.step()

                if epoch % 100 == 0:
                    print(f'Epoch {epoch}, Loss: {loss}')

                    # Back to CPU for plotting
                    x_ = detach_to_numpy(inputs)
                    f_ = detach_to_numpy(f)
                    if self.dim_plot == 1:
                        save_plot(
                            output_dir,
                            epoch=epoch,
                            x=x_,
                            f=f_,
                            loss=loss,
                            analytical=self.analytical
                        )
                    elif airfoil_mask is not None:
                        airfoil_mask_ = detach_to_numpy(airfoil_mask)
                        save_airfoil_plot(
                            output_dir,
                            epoch=epoch,
                            inputs=x_,
                            f=f_,
                            loss=loss,
                            airfoil_mask=airfoil_mask_
                        )
                    else:
                        save_2d_plot(
                            output_dir,
                            epoch=epoch,
                            inputs=x_,
                            f=f_,
                            loss=loss,
                            analytical=self.analytical
                        )

                create_gif()
