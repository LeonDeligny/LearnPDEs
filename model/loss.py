'''
Loss functions.
'''

# ======= Imports =======

import torch

from torch import tensor
from torch.autograd import grad
from utils.utility import num_inputs

from torch import Tensor
from torch.nn import MSELoss
from typing import Tuple, Callable

from __init__ import device

# ======= Class =======


class Loss:
    device = device
    mse_loss = MSELoss().to(device)
    zero_tensor = tensor([0.0]).view(-1, 1).to(device)
    one_tensor = tensor([1.0]).view(-1, 1).to(device)

    def __init__(
        self: 'Loss',
        x: Tensor,
        forward: Callable[[Tensor], Tensor],
    ) -> None:

        self.forward = forward
        self.x = x.requires_grad_().to(self.device)
        self.num_inputs: int = num_inputs(self.x)

    def process(
        self,
        physics_loss: float,
        boundary_loss: float
    ) -> float:
        """
        Process the losses to return a single loss value.
        TODO: Implement different methods to process the losses.
        """
        return physics_loss + (boundary_loss * self.num_inputs)

    def exponential_loss(self) -> Tuple[float, Tensor]:
        f = self.forward(self.x)
        df_dx = grad(
            outputs=f,
            inputs=self.x,
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

        return self.process(physics_loss, boundary_loss), self.x, f

    def cosinus_loss(self) -> Tuple[float, Tensor]:
        f = self.forward(self.x)
        df_dx = grad(
            outputs=f,
            inputs=self.x,
            grad_outputs=torch.ones_like(f),
            create_graph=True,
        )[0].view(-1, 1).to(self.device)

        ddf_dxdx = grad(
            outputs=df_dx,
            inputs=self.x,
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
            ddf_dxdx[self.zero_tensor == self.x].view(-1, 1),
            self.zero_tensor
        )

        return self.process(physics_loss, boundary_loss), self.x, f


# ======= Main =======


if __name__ == '__main__':
    print('Nothing to execute.')
