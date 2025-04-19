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

from learnpdes.model.encoding import complex_projection
from torch.nn.init import (
    zeros_,
    xavier_uniform_,
)

from torch import Tensor
from functools import partial
from complexPyTorch.complexLayers import ComplexLinear

from typing import (
    Dict,
    Union,
    Callable,
)
from torch.nn import (
    Tanh,
    Linear,
    Module,
    Sequential,
)

from learnpdes import (
    device,
    device_type,
)

# ======= Class =======


class PINN(Module):
    '''
    Physics Informed Neural Network (PINN) class.
    This class implements a PINN for solving ODEs using a neural network.
    '''

    # Constants
    device = device
    device_type = device_type

    def __init__(
        self: 'PINN',
        nn_params: Dict,
        input_homeo: Callable[[Tensor], Tensor],
        output_homeo: Callable[[Tensor], Tensor],
        encoding: Union[partial, Callable[[Tensor], Tensor]],
    ) -> None:
        super(PINN, self).__init__()

        # NN parameters
        self.input_dim: int = nn_params.get('input_dim')
        self.hidden_dim: int = nn_params.get('hidden_dim')
        self.num_hidden_layers: int = nn_params.get('num_hidden_layers')

        # Homeomorphisms and encoding
        self.input_homeo = input_homeo
        self.output_homeo = output_homeo
        self.encoding = encoding
        self.encoding_dim = self.get_encoding_dim()

        # Network parameters
        self.network = self.construct_nn().to(self.device)

    def get_encoding_dim(self) -> int:
        if isinstance(self.encoding, partial):
            return self.encoding.keywords.get('dim')
        elif self.input_homeo == complex_projection:
            print('Using encoding complex_projection')
            return 2 * self.input_dim
        else:
            print('Using no encoding')
            return self.input_dim

    def forward(self, x: Tensor) -> Tensor:
        '''
        forward = homeo o NN o fourier o homeo
            - input_homeo: [n, .] -> [n, .]
            - encoding: [n, .] -> [n, . * m]
            - NN: [n, . * m] -> [n, .]
            - output_homeo: [n, .] -> [n, .]
        '''
        input_homeo = self.input_homeo(x)
        encoding = self.encoding(input_homeo)
        network = self.network(encoding)
        output = self.output_homeo(network)

        return output.to(self.device)

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

    def construct_complexnn(self) -> Sequential:
        # Define constants
        output_dim = 1

        # Construct NN
        layers = [ComplexLinear(self.encoding_dim, self.hidden_dim), Tanh()]
        for _ in range(self.num_hidden_layers - 1):
            layers.append(ComplexLinear(
                self.hidden_dim,
                self.hidden_dim
            ))
            layers.append(Tanh())
        layers.append(ComplexLinear(self.hidden_dim, output_dim))

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


if __name__ == '__main__':
    print('Nothing to execute.')
