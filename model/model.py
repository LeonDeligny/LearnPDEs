'''
Physics Informed Neural Network's loss will be an ODE,

Examples:
    1. Exponential ODE, f = exp

        f' = f, f(0) = 1.

    2. Cosinus ODE, f = cos

        f'' = -f, f(0) = 1, f'(0) = 0.

Unique analytical solution is f = exp.
'''

# ======= Imports =======

import os
import torch

from utils.utility import num_inputs
from torch.autograd import grad
from torch import tensor
from utils.plot import (
    save_plot,
    create_gif,
)
from torch.nn.init import (
    zeros_,
    xavier_uniform_,
)

from typing import Dict, Tuple, Union, Callable
from torch import Tensor
from torch.optim import Adam
from functools import partial

from torch.nn import (
    Tanh,
    Linear,
    Module,
    MSELoss,
    Sequential,
)

from __init__ import device, device_type

# ======= Class =======


class PINN(Module):
    """
    Physics Informed Neural Network (PINN) class.
    This class implements a PINN for solving ODEs using a neural network.
    """

    # Constants
    device = device
    device_type = device_type
    zero_tensor = tensor([0.0]).view(-1, 1).to(device)
    one_tensor = tensor([1.0]).view(-1, 1).to(device)

    def __init__(
        self: 'PINN',
        input_space: Tensor,
        nn_params: Dict,
        training_params: Dict,
        loss_func_name: str,
        input_homeo: Callable[[Tensor], Tensor],
        output_homeo: Callable[[Tensor], Tensor],
        encoding: Union[partial, Callable[[Tensor], Tensor]],
    ) -> None:
        super(PINN, self).__init__()

        # Input space [n, 1]
        self.x: Tensor = input_space.requires_grad_().to(self.device)

        # NN parameters
        self.hidden_dim: int = nn_params.get('hidden_dim')
        self.num_hidden_layers: int = nn_params.get('num_hidden_layers')

        # Training parameters
        learning_rate: float = training_params.get('learning_rate')
        self.nb_epochs: int = training_params.get('nb_epochs')

        # Loss function
        self.loss_func_name: str = loss_func_name

        # Homeomorphisms and encoding
        self.input_homeo = input_homeo
        self.output_homeo = output_homeo
        self.encoding = encoding
        self.encoding_dim: int = (
            self.encoding.keywords.get('dim', 1)
            if isinstance(self.encoding, partial)
            else 1
        )

        # Network parameters
        self.input_dim: int = len(input_space)
        self.network: Sequential = self.construct_nn().to(self.device)

        # Training parameters
        self.optimizer = Adam(
            self.parameters(),
            lr=learning_rate,
        )
        self.mse_loss = MSELoss().to(self.device)

        # Gif parameters
        os.makedirs('gifs', exist_ok=True)

    def forward(self, x: Tensor) -> Tensor:
        '''
        forward = NN o fourier o homeo
            - input_homeo: [n, 1] -> [n, 1]
            - encoding: [n, 1] -> [n, m]
            - NN: [n, m] -> [n, 1]
            - output_homeo: [n, 1] -> [n, 1]
        '''
        input = self.input_homeo(x)
        encoding = self.encoding(input)
        network = self.network(encoding)
        output = self.output_homeo(network)

        return output

    def construct_nn(self) -> Sequential:
        # Define constants
        output_dim = 1

        # Construct NN
        layers = [Linear(self.encoding_dim, self.hidden_dim), Tanh()]
        for _ in range(self.num_hidden_layers - 1):
            layers.append(Linear(self.hidden_dim, self.hidden_dim))
            layers.append(Tanh())
        layers.append(Linear(self.hidden_dim, output_dim))

        network = Sequential(*layers)

        # Log network structure
        print(network)

        # Initialize weights
        self._initialize_weights()

        return network

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Linear):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)

    def train(self) -> None:
        # Dynamically call the specified loss function
        loss_function = getattr(self, self.loss_func_name)

        # Training loop
        for epoch in range(self.nb_epochs):
            self.optimizer.zero_grad()

            loss, y = loss_function(self.x)

            loss.backward(retain_graph=True)
            self.optimizer.step()

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item()}')
                # Back to CPU for plotting
                save_plot(epoch, self.x.cpu(), y.cpu(), loss)

        # Crate Gif with saved plots
        create_gif()

    def exponential_loss(self, x: Tensor) -> Tuple[float, Tensor]:
        f = self.forward(x)
        df_dx = grad(
            outputs=f,
            inputs=x,
            grad_outputs=torch.ones_like(f),
            create_graph=True,
        )[0]

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
        )[0]

        ddf_dxdx = grad(
            outputs=df_dx,
            inputs=x,
            grad_outputs=torch.ones_like(f),
            create_graph=True,
        )[0]

        # f' = f
        physics_loss = self.mse_loss(f, -ddf_dxdx)

        # f(0) = 1
        boundary_loss = self.mse_loss(
            self.forward(self.zero_tensor),
            self.one_tensor
        ) + self.mse_loss(
            ddf_dxdx(self.zero_tensor),
            self.one_tensor
        )

        return physics_loss + (boundary_loss * num_inputs(f)), f


if __name__ == '__main__':
    print('Nothing to execute.')
