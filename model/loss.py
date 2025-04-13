'''
Loss functions.
'''

# ======= Imports =======

import torch

from torch import tensor
from torch.autograd import grad
from utils.utility import laplace_function

from torch import Tensor
from torch.nn import MSELoss
from typing import Tuple, Callable

from __init__ import (
    device,
    pi_tensor,
)

# ======= Class =======


class Loss:
    device = device
    mse_loss = MSELoss().to(device)
    zero = tensor([0.0]).to(device)
    one = tensor([1.0]).to(device)

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
        self.dim = self.input_space.ndimension()

        # Transform input space into
        # 1D: (x)
        # 2D: (x, y)
        self.generate_inputs()

        # Generate boundaries of input space
        # 1D: (x = 0)
        # 2D: (x = 0, y), (y = 0, x) etc.
        self.generate_boundaries()

    def process(
        self,
        physics_loss: Tensor,
        boundary_loss: Tensor
    ) -> Tensor:
        '''
        Process the losses to return a single loss value.
        TODO: Implement different methods to process the losses.
        '''
        total_loss = physics_loss + boundary_loss
        return total_loss

    def generate_inputs(self) -> None:
        '''
        Generate input points based on the input space.
        '''
        if self.dim == 1:
            self.x = (
                self.input_space.requires_grad_()
                .view(-1, 1).to(self.device)
            )
            self.y = None  # No y-dimension for 1D input
        elif self.dim == 2:
            self.x, self.y = self.input_space[:, 0], self.input_space[:, 1]
            self.x = self.x.requires_grad_().view(-1, 1).to(self.device)
            self.y = self.y.requires_grad_().view(-1, 1).to(self.device)
            self.inputs = torch.cat([self.x, self.y], dim=1).to(self.device)
            self.laplace = laplace_function(self.x, self.y)
        else:
            # TODO: Implement 3D input space
            raise ValueError("Input space must be 1D or 2D.")

    def generate_boundaries(self) -> None:
        if self.dim == 1:
            # x = 0
            self.zero_mask = (self.x.squeeze() == 0).to(self.device)
            # x = 1
            self.one_mask = (self.x.squeeze() == 1).to(self.device)

            # f(x = 0)
            self.forward_null = self.forward(self.x[self.zero_mask])

            self.zero_tensor = (
                self.zero.expand_as(self.forward_null)
                .view(-1, 1).to(device)
            )

            self.one_tensor = (
                self.one.expand_as(self.forward_null)
                .view(-1, 1).to(device)
            )

        if self.input_space.ndimension() == 2:
            # x = 0
            self.zero_x_mask = (self.x.squeeze() == 0).to(self.device)
            # y = 0
            self.zero_y_mask = (self.y.squeeze() == 0).to(self.device)
            # x = 1
            self.one_x_mask = (self.x.squeeze() == 1).to(self.device)
            # y = 0
            self.one_y_mask = (self.y.squeeze() == 1).to(self.device)
            # f(x = 0, y)
            self.forward_x_null = self.forward(self.inputs[self.zero_x_mask])
            # f(x = 1, y)
            self.forward_x_one = self.forward(self.inputs[self.one_x_mask])
            # f(x, y = 0)
            self.forward_y_null = self.forward(self.inputs[self.zero_y_mask])

            self.zero_x_tensor = (
                self.zero.expand_as(self.forward_x_null)
                .view(-1, 1).to(device)
            )

            self.zero_y_tensor = (
                self.zero.expand_as(self.forward_y_null)
                .view(-1, 1).to(device)
            )

            self.one_x_tensor = (
                self.one.expand_as(self.forward_x_one)
                .view(-1, 1).to(device)
            )

            self.sin = torch.sin(
                pi_tensor * self.x[self.y.squeeze() == 1]
            ).view(-1, 1)

        elif self.input_space.ndimension() == 3:
            # TODO: Implement 3D input space
            raise ValueError("Input space must be 1D or 2D.")

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

    def laplace_loss(self) -> Tuple[Tensor, Tensor, Tensor]:
        f = self.forward(self.inputs)

        # Compute the second derivatives
        df_dx = self.partial_derivative(f, self.x)
        ddf_dxdx = self.partial_derivative(df_dx, self.x)

        df_dy = self.partial_derivative(f, self.y)
        ddf_dydy = self.partial_derivative(df_dy, self.y)

        # Delta f = 0
        physics_loss = self.mse_loss(ddf_dxdx, -ddf_dydy)

        # physics_loss = self.mse_loss(
        #     f,
        #     self.laplace,
        # )

        # Dirichlet boundary conditions
        boundary_loss = (
            self.mse_loss(
                # f(., 0) = 0
                f[self.zero_y_mask].view(-1, 1),
                self.zero_y_tensor,
            ) + self.mse_loss(
                # f(., 1) = sin(pi x)
                f[self.one_y_mask].view(-1, 1),
                self.sin,
            ) + self.mse_loss(
                # f(0, .) = 0
                f[self.zero_x_mask].view(-1, 1),
                self.zero_x_tensor,
            ) + self.mse_loss(
                # f(1, .) = 0
                f[self.one_x_mask].view(-1, 1),
                self.zero_x_tensor,
            )
        )
        return self.process(physics_loss, boundary_loss), self.inputs, f

    def exponential_loss(self) -> Tuple[Tensor, Tensor, Tensor]:
        f = self.forward(self.x)
        df_dx = self.partial_derivative(f, self.x)

        # f' = f
        physics_loss = self.mse_loss(f, df_dx)

        # f(0) = 1
        boundary_loss = self.mse_loss(
            f[self.zero_mask].view(-1, 1),
            self.one_tensor,
        )

        return self.process(physics_loss, boundary_loss), self.x, f

    def cosinus_loss(self) -> Tuple[Tensor, Tensor, Tensor]:
        f = self.forward(self.x)
        df_dx = self.partial_derivative(f, self.x)
        ddf_dxdx = self.partial_derivative(df_dx, self.x)

        # f'' = -f
        physics_loss = self.mse_loss(f, -ddf_dxdx)

        # f(0) = 1, f'(0) = 0
        boundary_loss = self.mse_loss(
            f[self.zero_mask].view(-1, 1),
            self.one_tensor,
        ) + self.mse_loss(
            ddf_dxdx[self.zero_mask].view(-1, 1),
            self.zero_tensor
        )

        return self.process(physics_loss, boundary_loss), self.x, f


# ======= Main =======


if __name__ == '__main__':
    print('Nothing to execute.')
