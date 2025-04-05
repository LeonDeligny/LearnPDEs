'''
Training configuration of the PINN model.
'''

# ======= Imports =======

import os
import torch

from torch import tensor
from torch.autograd import grad
from utils.utility import num_inputs
from utils.plot import (
    save_plot,
    create_gif,
)

from torch import Tensor
from torch.optim import Adam
from torch.nn import MSELoss
from model.pinn import PINN
from typing import (
    Dict,
    Tuple,
    Union,
    Callable,
)

from __init__ import device

# ======= Class =======


class Trainer:
    device = device
    zero_tensor = tensor([0.0]).view(-1, 1).to(device)
    one_tensor = tensor([1.0]).view(-1, 1).to(device)

    def __init__(
        self,
        pinn: PINN,
        input_space: Tensor,
        training_params: Dict,
        loss_func_name: str,
        analytical: Union[Callable, None] = None,
    ) -> None:
        """
        Handles training process for the PINN model.
        """

        # Physics Informed Neural Network model
        self.pinn = pinn
        self.forward = self.pinn.forward

        # Input space
        self.input_space = input_space

        # Training parameters
        self.learning_rate: float = training_params.get('learning_rate')
        self.nb_epochs: int = training_params.get('nb_epochs')

        # Name of the loss function to use
        self.loss_func_name = loss_func_name

        # Analytical solution if any
        self.analytical = analytical
        self.input_dim: int = len(input_space)

        # Input space [n, 1]
        self.x: Tensor = input_space.requires_grad_().to(self.device)

        # Training parameters
        self.optimizer = Adam(
            self.pinn.parameters(),
            lr=self.learning_rate,
        )
        self.mse_loss = MSELoss().to(self.device)

        # Gif parameters
        os.makedirs('gifs', exist_ok=True)

    def train(self) -> None:
        loss_function = getattr(self, self.loss_func_name)

        # Training loop
        for epoch in range(self.nb_epochs):
            self.optimizer.zero_grad()

            loss, y = loss_function(self.x)

            loss.backward(retain_graph=True)
            self.optimizer.step()

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")
                # Back to CPU for plotting
                save_plot(epoch, self.x.cpu(), y.cpu(), loss, self.analytical)

        # Create GIF with saved plots
        create_gif()

    def exponential_loss(self, x: Tensor) -> Tuple[float, Tensor]:
        f = self.forward(x)
        df_dx = grad(
            outputs=f,
            inputs=x,
            grad_outputs=torch.ones_like(f),
            create_graph=True,
        )[0].view(-1, 1).to(self.device)

        # f' = f
        physics_loss = self.mse_loss(f, df_dx)

        # f(0) = 1
        boundary_loss = self.mse_loss(
            self.forward(self.zero_tensor),
            self.one_tensor,
        )

        return physics_loss + (boundary_loss * num_inputs(f)), f

    def cosinus_loss(self, x: Tensor) -> Tuple[float, Tensor]:
        f = self.forward(x)
        df_dx = grad(
            outputs=f,
            inputs=x,
            grad_outputs=torch.ones_like(f),
            create_graph=True,
        )[0].view(-1, 1).to(self.device)

        ddf_dxdx = grad(
            outputs=df_dx,
            inputs=x,
            grad_outputs=torch.ones_like(f),
            create_graph=True,
        )[0].view(-1, 1).to(self.device)

        # f' = f
        physics_loss = self.mse_loss(f, -ddf_dxdx)

        # f(0) = 1, f'(0) = 0
        boundary_loss = self.mse_loss(
            self.forward(self.zero_tensor),
            self.one_tensor
        ) + self.mse_loss(
            ddf_dxdx[self.zero_tensor == x].view(-1, 1),
            self.zero_tensor
        )

        return physics_loss + (boundary_loss * num_inputs(f)), f



# ======= Main =======


if __name__ == '__main__':
    print('Nothing to execute.')
