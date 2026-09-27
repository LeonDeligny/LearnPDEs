"""
Physics Informed Neural Network's loss will be an ODE,

Examples:
    1. Exponential ODE, f = exp

        f' = f, f(0) = 1.

    2. Cosinus ODE, f = cos

        f'' = -f, f(0) = 1, f'(0) = 0.

Unique analytical solution is f = exp.
"""

# ======= Imports =======

from torch.nn.init import (
    zeros_,
    xavier_uniform_,
)

from torch import Tensor
from learnpdes.model.derivatives import tanh_derivatives
from learnpdes.model.encodings import identity
from functools import partial
# from complexPyTorch.complexLayers import ComplexLinear

from typing import (
    Union,
    Callable,
)
from torch.nn import (
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
    """
    Physics Informed Neural Network (PINN) class.
    This class implements a PINN for solving ODEs using a neural network.
    """

    # Constants
    device = device
    device_type = device_type

    def __init__(
        self: 'PINN',
        nn_params: dict,
        input_homeo: Callable[[Tensor], Tensor],
        output_homeo: Callable[[Tensor], Tensor],
        encoding: Union[partial, Callable[[Tensor], Tensor]],
        *,
        output_transform: Callable[[Tensor, Tensor], Tensor] | None = None,
    ) -> None:
        super(PINN, self).__init__()

        # NN parameters
        self.input_dim: int = nn_params.get('input_dim')
        self.hidden_dim: int = nn_params.get('hidden_dim')
        self.output_dim: int = nn_params.get('output_dim')
        self.num_hidden_layers: int = nn_params.get('num_hidden_layers')
        self.activation: Module = nn_params.get('activation')

        # Homeomorphisms and encodings
        self.input_homeo = input_homeo
        self.output_homeo = output_homeo
        self.encoding = encoding
        self.output_transform = output_transform
        self.encoding_dim = self.get_encoding_dim()

        # Network parameters
        self.network = self.construct_nn().to(self.device)

    def get_encoding_dim(self) -> int:
        if isinstance(self.encoding, partial):
            return self.encoding.keywords.get('dim')
        else:
            print('Using no encoding')
            return self.input_dim

    def forward(self, x: Tensor) -> Tensor:
        """
        forward = homeo o NN o fourier o homeo
            - input_homeo: [n, .] -> [n, .]
            - encoding: [n, .] -> [n, . * m]
            - NN: [n, . * m] -> [n, .]
            - output_homeo: [n, .] -> [n, .]
        """
        input_homeo = self.input_homeo(x)
        encoding = self.encoding(input_homeo)
        network = self.network(encoding)
        output = self.output_homeo(network)
        if self.output_transform is not None:
            output = self.output_transform(x, output)

        return output

    def input_derivatives(self, x: Tensor, order: int) -> list[Tensor]:
        """Efficient high-order derivatives of the unencoded scalar Tanh network."""
        if (
            self.output_transform is not None
            or self.input_dim != 1
            or any(
                transform is not identity
                for transform in (self.input_homeo, self.output_homeo, self.encoding)
            )
        ):
            raise ValueError(
                'Taylor derivatives require an unencoded one-dimensional network.'
            )
        return tanh_derivatives(self.network, x, order)

    def construct_nn(self) -> Sequential:
        """
        Using standard neural network definition.
        By default add a biais.
        """
        # Construct NN
        layers = [
            Linear(self.encoding_dim, self.hidden_dim, bias=True),
            self.activation(),
        ]
        for _ in range(self.num_hidden_layers - 1):
            layers.append(Linear(self.hidden_dim, self.hidden_dim, bias=True))
            layers.append(self.activation())
        layers.append(Linear(self.hidden_dim, self.output_dim, bias=True))

        network = Sequential(*layers)

        # Log network structure
        print(network)

        # Initialize weights
        self._initialize_weights()

        return network

    def _initialize_weights(self):
        """
        Fill the input Tensor with values
        using a Xavier uniform distribution.
        Biais initialized to 0.
        """
        for m in self.modules():
            if isinstance(m, Linear):
                xavier_uniform_(m.weight)
                zeros_(m.bias)
