'''
Loss functions.
'''

# ======= Imports =======

import torch

from torch import tensor
from torch.autograd import grad

from torch import Tensor
from torch.nn import MSELoss
from typing import Tuple, Callable

from __init__ import device

# ======= Class =======


class Loss:
    device = device
    mse_loss = MSELoss().to(device)

    def __init__(
        self: 'Loss',
        input_space: Tensor,
        forward: Callable[[Tensor], Tensor],
    ) -> None:
        '''
        Initialization of the loss.
        '''

        self.forward = forward
        self.input_space = input_space

        # Transform input space into
        # 1D: (x)
        # 2D: (x, y)
        self.generate_inputs()

    def process(
        self,
        physics_loss: float,
        boundary_loss: float
    ) -> float:
        '''
        Process the losses to return a single loss value.
        TODO: Implement different methods to process the losses.
        '''
        return physics_loss + boundary_loss

    def generate_inputs(self) -> None:
        '''
        Generate input points based on the input space.
        :param input_space: Tensor defining the input domain.
        :return: Tensor of input points.
        '''
        if self.input_space.ndimension() == 1:
            self.x = (
                self.input_space.requires_grad_()
                .view(-1, 1).to(self.device)
            )
            self.y = None  # No y-dimension for 1D input
        elif self.input_space.ndimension() == 2:
            self.x, self.y = self.input_space[:, 0], self.input_space[:, 1]
            self.x = self.x.requires_grad_().view(-1, 1).to(self.device)
            self.y = self.y.requires_grad_().view(-1, 1).to(self.device)
            self.inputs = torch.cat([self.x, self.y], dim=1).to(self.device)
        else:
            # TODO: Implement 3D input space
            raise ValueError("Input space must be 1D or 2D.")

        # x = 0
        self.null_mask = (self.x.squeeze() == 0).to(self.device)
        
        # f(x = 0)
        self.forward_null = self.forward(self.x[self.null_mask])
        self.zero_tensor = (
            tensor([0.0]).expand_as(self.forward_null)
            .view(-1, 1).to(device)
        )
        self.one_tensor = (
            tensor([1.0]).expand_as(self.forward_null)
            .view(-1, 1).to(device)
        )

    def partial_derivative(self, f: Tensor, x: Tensor) -> Tensor:
        """
        Compute the first derivative of 1D outputs with respect to the inputs.
        """
        return grad(
            outputs=f,
            inputs=x,
            grad_outputs=torch.ones_like(f),
            create_graph=True,
        )[0].view(-1, 1).to(self.device)

    def laplace_loss(self) -> Tuple[float, Tensor, Tensor]:
        f = self.forward(self.inputs)

        # Compute the second derivatives
        df_dx = self.partial_derivative(f, self.x)
        ddf_dxdx = self.partial_derivative(df_dx, self.x)

        df_dy = self.partial_derivative(f, self.y)
        ddf_dydy = self.partial_derivative(df_dy, self.y)

        # Delta f = 0
        physics_loss = self.mse_loss(ddf_dxdx + ddf_dydy, self.zero_tensor)

        # Dirichlet boundary conditions
        boundary_loss = (
            self.mse_loss(
                # f(., 0) = 1
                f[self.zero_tensor == self.y.squeeze()].view(-1, 1),
                self.zero_tensor,
            ) + self.mse_loss(
                # f(., 1) = sin(pi x)
                f[self.one_tensor == self.y.squeeze()].view(-1, 1),
                torch.sin(
                    torch.pi * self.x[self.y.squeeze() == 1]
                ).view(-1, 1),
            ) + self.mse_loss(
                # f(0, .) = 0
                f[self.zero_tensor == self.x.squeeze()].view(-1, 1),
                self.zero_tensor,
            ) + self.mse_loss(
                # f(1, .) = 0
                f[self.one_tensor == self.x.squeeze()].view(-1, 1),
                self.zero_tensor,
            )
        )

        return self.process(physics_loss, boundary_loss), self.inputs, f

    def exponential_loss(self) -> Tuple[float, Tensor, Tensor]:
        f = self.forward(self.x)
        df_dx = self.partial_derivative(f, self.x)

        # f' = f
        physics_loss = self.mse_loss(f, df_dx)

        # f(0) = 1
        boundary_loss = self.mse_loss(
            f[self.null_mask].view(-1, 1),
            self.one_tensor,
        )

        return self.process(physics_loss, boundary_loss), self.x, f

    def cosinus_loss(self) -> Tuple[float, Tensor, Tensor]:
        f = self.forward(self.x)
        df_dx = self.partial_derivative(f, self.x)
        ddf_dxdx = self.partial_derivative(df_dx, self.x)

        # f' = f
        physics_loss = self.mse_loss(f, -ddf_dxdx)

        # f(0) = 1, f'(0) = 0
        boundary_loss = self.mse_loss(
            f[self.null_mask].view(-1, 1),
            self.one_tensor,
        ) + self.mse_loss(
            ddf_dxdx[self.null_mask].view(-1, 1),
            self.zero_tensor
        )

        return self.process(physics_loss, boundary_loss), self.x, f


# ======= Main =======


if __name__ == '__main__':
    print('Nothing to execute.')
