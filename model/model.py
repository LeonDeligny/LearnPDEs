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
from utils.homeomorphisms import input_homeo
from torch.autograd import grad
from model.encoding import fourier
from torch import tensor
from utils.plot import (
    save_plot,
    create_gif,
)
from torch.nn.init import (
    zeros_, 
    xavier_uniform_,
)

from torch import Tensor
from torch.optim import Adam

from typing import Dict

from torch.nn import (
    ReLU,
    Tanh,
    Linear,
    Module,
    MSELoss,
    Sequential,
)

# ======= Class =======


class PINN(Module):
    zero_tensor = tensor([0.0]).view(-1, 1)
    one_tensor = tensor([1.0]).view(-1, 1)

    def __init__(
        self: 'PINN',
        input_space: Tensor,
        nn_params: Dict,
        training_params: Dict,
        loss_func_name: str,
    ) -> None:
        super(PINN, self).__init__()

        # Input space
        self.x = input_space.requires_grad_()

        # NN parameters
        self.hidden_dim = nn_params.get('hidden_dim')
        self.num_hidden_layers = nn_params.get('num_hidden_layers')

        # Training parameters
        learning_rate = training_params.get('learning_rate')
        self.nb_epochs = training_params.get('nb_epochs')

        # Loss function
        self.loss_func_name = loss_func_name

        # Network parameters
        self.input_dim = len(input_space)
        self.network = self.construct_nn()

        # Training parameters
        self.optimizer = Adam(
            self.parameters(), 
            lr=learning_rate,
        )
        self.mse_loss = MSELoss()

        # Gif parameters
        os.makedirs('gifs', exist_ok=True)

    def forward(self, x: Tensor) -> Tensor:
        '''
        forward = NN o fourier o homeo
            - homeo: [n, 1] -> [n, 1]
            - fourier: [n, 1] -> [n, m]
            - NN: [n, m] -> [n, 1]
        '''
        return self.network(input_homeo(x))

#        return self.network(
#            fourier(
#                input_homeo(x),
#                dim=self.hidden_dim,
#            ),
#        )

    def construct_nn(self) -> None:
        # Define constants
        input_dim = 1
        output_dim = 1
        
        # Construct NN
        layers = [Linear(input_dim, self.hidden_dim), Tanh()]
        for _ in range(self.num_hidden_layers-1):
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
            
            # Compute loss
            loss, y = loss_function(self.x)

            # Backpropagation
            loss.backward(retain_graph=True)
            self.optimizer.step()
            
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item()}')
                save_plot(epoch, self.x, y)

        # Crate Gif with saved plots
        create_gif()

    def exponential_loss(self, x: Tensor) -> float:
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

    def cosinus_loss(self, x: Tensor) -> float:
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
