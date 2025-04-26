'''
Loss functions.
'''

# ======= Imports =======

import torch

from torch import tensor
from torch.autograd import grad

from torch import Tensor
from ambiance import Atmosphere
from torch.nn import MSELoss
from typing import (
    Tuple,
    Callable,
)

from learnpdes import (
    device,
    pi_tensor,
)

# ======= Class =======


class Loss:
    device = device
    mse_loss = MSELoss().to(device)
    zero = tensor([0.0]).to(device)
    one = tensor([1.0]).to(device)
    # Density of water at alt = 0.0
    atm = Atmosphere(0.0)
    rho = atm.density

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
        print(f'Input space is of dimension {self.dim}.')

        # Transform input space into
        # 1D: (x)
        # 2D: (x, y)
        self.generate_inputs()

        # Generate boundaries of input space
        # 1D: (x = 0)
        # 2D: (x = 0, y), (x = 1, y), (x, y = 0) and (x, y = 1)
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
        # Always maximum 3d physical space
        if self.dim == 1:
            self.input_space = self.input_space.unsqueeze(1)
        self.x = self.setup_space(index=0)
        self.y = self.setup_space(index=1)
        self.inputs = (
            torch.cat([self.x, self.y], dim=1)
            if self.y is not None
            else self.x
        )
        # self.z = self.setup_space(index=2)

    def setup_space(self, index: int) -> Tensor:
        return (
            self.input_space[:, index].requires_grad_()
                .view(-1, 1).to(self.device)
            if self.dim > index
            else None
        )

    def generate_1d_boundaries(self) -> None:
        # x = 0
        self.zero_mask = (self.x.squeeze() == 0).to(self.device)

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

    def generate_2d_boundaries(self) -> None:
        # x = 0
        self.zero_x_mask = (self.x.squeeze() == 0).to(self.device)
        # y = 0
        self.zero_y_mask = (self.y.squeeze() == 0).to(self.device)
        # x = 1
        self.one_x_mask = (self.x.squeeze() == 1).to(self.device)
        # y = 1
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

        self.sin = torch.sin(pi_tensor * self.x[self.one_y_mask]).view(-1, 1)

    def generate_boundaries(self) -> None:
        if self.dim == 1:
            self.generate_1d_boundaries()

        if self.dim == 2:
            self.generate_2d_boundaries()

        elif self.dim == 3:
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

    def get_loss(self, scenario: str) -> Callable:
        if scenario == 'exponential':
            return self.exponential_loss
        elif scenario == 'cosinus':
            return self.cosinus_loss
        elif scenario == 'laplace':
            return self.laplace_loss
        elif scenario == 'potential flow':
            return self.potential_flow_loss
        else:
            raise ValueError('Scenario not found.')

    def laplace_loss(self) -> Tuple[Tensor, Tensor, Tensor]:
        f = self.forward(self.inputs)

        # Compute the second derivatives
        df_dx = self.partial_derivative(f, self.x)
        ddf_dxdx = self.partial_derivative(df_dx, self.x)

        df_dy = self.partial_derivative(f, self.y)
        ddf_dydy = self.partial_derivative(df_dy, self.y)

        # Delta f = 0
        physics_loss = self.mse_loss(ddf_dxdx, -ddf_dydy)

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

    def potential_flow_loss(self) -> Tuple[Tensor, Tensor, Tensor]:
        '''
        No mu term as the laplacian of u, v is null.
        '''
        u, v, p = self.forward(self.inputs)

        # u_x + v_y = 0 (Incompressibility)
        ic_loss, _, v_y = self.incompressibility_loss(u, v)

        # u u_x + v u_y + p_x / rho = 0
        u_y = self.partial_derivative(u, self.y)
        p_x = self.partial_derivative(p, self.x)
        flow_x_loss = self.mse_loss(v * u_y - u * v_y, -p_x / self.rho)

        # u v_x + v v_y + p_y / rho = 0
        v_x = self.partial_derivative(v, self.x)
        p_y = self.partial_derivative(p, self.y)
        flow_y_loss = self.mse_loss(u * v_x + v * v_y, -p_y / self.rho)

        physics_loss = flow_x_loss + flow_y_loss + ic_loss

        # u(x = 1, .) = 1 and v(x = 0, .) = 0
        # Inlet boundary condition
        inlet_loss = (
            self.mse_loss(u[self.zero_x_mask], self.one_tensor)
            + self.mse_loss(v[self.zero_x_mask], self.zero_tensor)
            + self.mse_loss(p[self.zero_x_mask], self.one_tensor)
        )
        # Outlet boundary condition
        outlet_loss = (
            self.mse_loss(u[self.one_x_mask], self.one_tensor)
            + self.mse_loss(v[self.one_x_mask], self.zero_tensor)
        )
        # Wall boundary condition
        wall_loss = (
            self.mse_loss(u[self.zero_y_mask], self.one_tensor)
            + self.mse_loss(v[self.zero_y_mask], self.zero_tensor)
            + self.mse_loss(u[self.one_y_mask], self.one_tensor)
            + self.mse_loss(v[self.one_y_mask], self.zero_tensor)
        )
        boundary_loss = inlet_loss + outlet_loss + wall_loss

        return (
            self.process(physics_loss, boundary_loss),
            self.inputs,
            (u, v, p),
        )

    # def navier_stokes_loss(self) -> Tuple[Tensor, Tensor, Tensor]:
    #     u, v, p = self.forward(self.inputs)

    #     # u_x + v_y = 0 (Incompressibility)
    #     u_x = self.partial_derivative(u, self.x)
    #     v_y = self.partial_derivative(v, self.y)

    #    u u_x + v u_y + p_x / rho - mu / rho (u_xx + u_yy) = 0
    #    u v_x + v v_y + p_y / rho - mu / rho (v_xx + v_yy) = 0

    def incompressibility_loss(self, u, v) -> Tuple[float, Tensor, Tensor]:
        # u_x + v_y = 0 (Incompressibility)
        u_x = self.partial_derivative(u, self.x)
        v_y = self.partial_derivative(v, self.y)

        return self.mse_loss(u_x, -v_y), u_x, v_y

# ======= Main =======


if __name__ == '__main__':
    print('Nothing to execute.')
