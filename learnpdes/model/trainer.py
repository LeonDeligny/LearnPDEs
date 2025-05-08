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
        output_dir = ensure_directory_exists()

        # Training loop
        for epoch in range(self.nb_epochs):
            self.optimizer.zero_grad()

            loss, x, y, f = self.loss()

            loss.backward(retain_graph=True)
            self.optimizer.step()

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

                # Back to CPU for plotting
                x_ = detach_to_numpy(x)
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
                else:
                    y_ = detach_to_numpy(y)
                    save_2d_plot(
                        output_dir,
                        epoch=epoch,
                        x1=x_,
                        x2=y_,
                        f=f_,
                        loss=loss,
                        analytical=self.analytical
                    )

        # Create GIF with saved plots
        create_gif()
