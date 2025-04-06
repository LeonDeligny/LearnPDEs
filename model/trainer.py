'''
Training configuration of the PINN model.
'''

# ======= Imports =======

import os

from utils.plot import (
    save_plot,
    create_gif,
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

from __init__ import device

# ======= Class =======


class Trainer:
    device = device

    def __init__(
        self,
        model_params: Iterator[Parameter],
        loss: Callable[[], Tuple[float, Tensor, Tensor]],
        training_params: Dict,
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
        self.nb_epochs: int = training_params.get('nb_epochs')

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

        # Training loop
        for epoch in range(self.nb_epochs):
            self.optimizer.zero_grad()

            loss, x, y = self.loss()

            loss.backward(retain_graph=True)
            self.optimizer.step()

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

                # Back to CPU for plotting
                save_plot(epoch, x.cpu(), y.cpu(), loss, self.analytical)

        # Create GIF with saved plots
        create_gif()


# ======= Main =======


if __name__ == '__main__':
    print('Nothing to execute.')
