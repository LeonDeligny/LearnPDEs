'''
Training configuration of the PINN model.
'''

# ======= Imports =======

import os

from learnpdes.utils.utility import detach_to_numpy
from learnpdes.utils.plot import (
    create_gif,
    ensure_directory_exists,
)

from torch import Tensor
from torch.optim import Adam
from torch.nn import Parameter
from typing import (
    Union,
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
        loss: Callable[[], tuple[Tensor, Tensor, Tensor]],
        training_params: dict,
        plot: dict,
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

        # Plotting parameters
        self.dim_plot = plot.get('input_dim')
        self.plot_func = plot.get('plot_func')

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
            # res_dir = ensure_directory_exists('./gifs/residuals')

            # Training loop
            for epoch in range(0, self.nb_epochs):
                self.optimizer.zero_grad()

                loss, inputs, f, geometry_mask = self.loss()

                loss.backward(retain_graph=True)
                self.optimizer.step()

                if epoch % 100 == 0:
                    print(f'Epoch {epoch}, Loss: {loss}')

                    # Back to CPU for plotting
                    x_ = detach_to_numpy(inputs)
                    f_ = detach_to_numpy(f)
                    # res_ = detach_to_numpy(res)

                    self.plot_func(
                        output_dir,
                        epoch=epoch,
                        inputs=x_,
                        f=f_,
                        loss=loss,
                        geometry_mask=geometry_mask,
                        analytical=self.analytical,
                    )

                    # self.plot_func(
                    #     res_dir,
                    #     epoch=epoch,
                    #     inputs=x_,
                    #     f=res_,
                    #     loss=loss,
                    #     geometry_mask=geometry_mask,
                    #     analytical=self.analytical,
                    # )

                create_gif()
