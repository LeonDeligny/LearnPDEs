'''
Loss functions.
'''

# ======= Imports =======

import torch

from numpy import array
from torch import tensor
from torch.autograd import grad
from learnpdes.utils.utility import compute_normals

from torch import Tensor
from typing import Callable
from torch.nn import MSELoss
from functools import partial
from ambiance import Atmosphere

from learnpdes import (
    device,
    pi_tensor,
)

from learnpdes import (
    EXPONENTIAL_SCENARIO,
    COSINUS_SCENARIO,
    LAPLACE_SCENARIO,
    POTENTIAL_FLOW_SCENARIO,
    SOLENOIDAL_FLOW_SCENARIO,
)

# ======= Class =======


class Loss:
    device = device
    mse_loss = MSELoss().to(device)
    zero = tensor([0.0]).to(device)
    one = tensor([1.0]).to(device)

    # Density of air at water level
    atm = Atmosphere(h=0.0)
    density = array([atm.density])
    print(f'Density of fluid {atm.density[0]}')
    rho = torch.tensor(density, dtype=torch.float32).to(device)

    def __init__(
        self: 'Loss',
        scenario: str,
        input_space: Tensor,
        input_dim: int,
        forward: Callable[[Tensor], Tensor],
        mesh_masks: dict[str, Tensor]
    ) -> None:
        '''
        Initialization of the loss.
        '''

        self.forward = forward
        self.input_space = input_space
        self.dim = input_dim
        self.mesh_masks = mesh_masks
        self.scenario = scenario
        print(f'Input space is of dimension {self.dim}.')

        # Transform input space into
        # 1D: (x)
        # 2D: (x, y)
        self.generate_inputs()

        # Generate boundaries of input space
        # 1D: (x = 0)
        # 2D: (x = 0, y), (x = 1, y), (x, y = 0) and (x, y = 1)
        self.generate_boundaries()

        # Generate scenario specific boundaries
        if self.scenario == LAPLACE_SCENARIO:
            self.generate_laplace_boundary()
        elif self.scenario in [
            POTENTIAL_FLOW_SCENARIO,
            SOLENOIDAL_FLOW_SCENARIO
        ]:
            n_x, n_y = compute_normals(
                xy=input_space,
                airfoil_mask=mesh_masks['airfoil'],
            )
            self.n_x, self.n_y = n_x.to(self.device), n_y.to(self.device)

    def process(
        self: 'Loss',
        physics_loss: Tensor,
        boundary_loss: Tensor
    ) -> Tensor:
        '''
        Process the losses to return a single loss value.
        TODO: Implement different methods to process the losses.
        '''
        total_loss = 3 * physics_loss + boundary_loss
        return total_loss

    def generate_inputs(self: 'Loss') -> None:
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
        self.inputs_mask = (self.inputs < 10) & (self.inputs > -1)
        # self.z = self.setup_space(index=2)

    def setup_space(self: 'Loss', index: int) -> Tensor:
        return (
            self.input_space[:, index].requires_grad_()
                .view(-1, 1).to(self.device)
            if self.dim > index
            else None
        )

    def generate_1d_boundaries(self: 'Loss') -> None:
        # x = 0
        self.zero_mask = self.mesh_masks["zero"].to(self.device)

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

    def generate_2d_boundaries(self: 'Loss') -> None:
        for name, mask in self.mesh_masks.items():
            if name == 'inlet':
                self.inlet_mask = mask.to(self.device)
                self.forward_inlet = self.forward(
                    self.inputs[self.inlet_mask]
                )[:, 0:1]
                self.inlet_zero_tensor = (
                    self.zero.expand_as(self.forward_inlet)
                    .view(-1, 1).to(device)
                )
                self.inlet_one_tensor = (
                    self.one.expand_as(self.forward_inlet)
                    .view(-1, 1).to(device)
                )
            elif name == 'outlet':
                self.outlet_mask = mask.to(self.device)
                self.forward_outlet = self.forward(
                    self.inputs[self.outlet_mask]
                )[:, 0:1]
                self.outlet_zero_tensor = (
                    self.zero.expand_as(self.forward_outlet)
                    .view(-1, 1).to(device)
                )
                self.outlet_one_tensor = (
                    self.one.expand_as(self.forward_outlet)
                    .view(-1, 1).to(device)
                )
            elif name == 'wall':
                self.wall_mask = mask.to(self.device)
                self.forward_wall = self.forward(
                    self.inputs[self.wall_mask]
                )[:, 0:1]
                self.wall_zero_tensor = (
                    self.zero.expand_as(self.forward_wall)
                    .view(-1, 1).to(device)
                )
                self.wall_one_tensor = (
                    self.one.expand_as(self.forward_wall)
                    .view(-1, 1).to(device)
                )
            elif name == 'top':
                self.top_mask = mask.to(self.device)
                self.forward_top = self.forward(
                    self.inputs[self.top_mask]
                )[:, 0:1]
                self.top_zero_tensor = (
                    self.zero.expand_as(self.forward_top)
                    .view(-1, 1).to(device)
                )
            elif name == 'bottom':
                self.bottom_mask = mask.to(self.device)
                self.forward_bottom = self.forward(
                    self.inputs[self.bottom_mask]
                )[:, 0:1]
                self.bottom_zero_tensor = (
                    self.zero.expand_as(self.forward_bottom)
                    .view(-1, 1).to(device)
                )
            elif name == 'airfoil':
                self.airfoil_mask = mask.to(self.device)
                self.forward_airfoil = self.forward(
                    self.inputs[self.airfoil_mask]
                )[:, 0:1]
                self.airfoil_zero_tensor = (
                    self.zero.expand_as(self.forward_airfoil)
                    .view(-1, 1).to(device)
                )
            else:
                raise ValueError(f'{name=} not known as a boundary name.')

    def generate_laplace_boundary(self: 'Loss') -> None:
        self.sin = torch.sin(pi_tensor * self.x[self.top_mask]).view(-1, 1)

    def generate_boundaries(self: 'Loss') -> None:
        if self.dim == 1:
            self.generate_1d_boundaries()

        elif self.dim == 2:
            self.generate_2d_boundaries()

        else:
            raise ValueError(f'{self.dim=} should be either 1 or 2.')

    def partial_derivative(self: 'Loss', f: Tensor, x: Tensor) -> Tensor:
        """
        Compute the first derivative of 1D outputs with respect to the inputs.
        """
        return grad(
            outputs=f,
            inputs=x,
            grad_outputs=torch.ones_like(f),
            create_graph=True,
        )[0].view(-1, 1).to(self.device)

    def get_loss(self: 'Loss', scenario: str) -> Callable:
        print("\n ----- Started training -----\n")
        if scenario == EXPONENTIAL_SCENARIO:
            return self.exponential_loss
        elif scenario == COSINUS_SCENARIO:
            return self.cosinus_loss
        elif scenario == LAPLACE_SCENARIO:
            return self.laplace_loss
        elif scenario in POTENTIAL_FLOW_SCENARIO:
            return self.potential_irrotational_flow_loss
        elif scenario in SOLENOIDAL_FLOW_SCENARIO:
            return self.solenoidal_flow_loss
        else:
            raise ValueError(f"{scenario=} is not a valid scenario.")

    def get_pre_loss(self: 'Loss', scenario: str) -> Callable:
        print("\n ----- Started pre-training -----\n")
        if scenario == EXPONENTIAL_SCENARIO:
            return self.exponential_loss
        elif scenario == COSINUS_SCENARIO:
            return self.cosinus_loss
        elif scenario == LAPLACE_SCENARIO:
            return self.laplace_loss
        elif scenario == POTENTIAL_FLOW_SCENARIO:
            return partial(
                self.potential_irrotational_flow_loss,
                pre=True,
            )
        elif scenario == SOLENOIDAL_FLOW_SCENARIO:
            return partial(
                self.solenoidal_flow_loss,
                pre=True,
            )
        else:
            raise ValueError(
                f"{scenario=} is not a valid scenario for a pre-training."
            )

    def laplace_loss(self: 'Loss') -> tuple[Tensor, Tensor, Tensor, None]:
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
                # f(bottom) = 0
                f[self.bottom_mask].view(-1, 1),
                self.bottom_zero_tensor,
            ) + self.mse_loss(
                # f(top) = sin(pi x)
                f[self.top_mask].view(-1, 1),
                self.sin,
            ) + self.mse_loss(
                # f(inlet) = 0
                f[self.inlet_mask].view(-1, 1),
                self.inlet_zero_tensor,
            ) + self.mse_loss(
                # f(outlet) = 0
                f[self.outlet_mask].view(-1, 1),
                self.outlet_zero_tensor,
            )
        )
        return self.process(physics_loss, boundary_loss), self.inputs, f, None

    def exponential_loss(self: 'Loss') -> tuple[Tensor, Tensor, Tensor, None]:
        f = self.forward(self.x)
        df_dx = self.partial_derivative(f, self.x)

        # f' = f
        physics_loss = self.mse_loss(f, df_dx)

        # f(0) = 1
        boundary_loss = self.mse_loss(
            f[self.zero_mask].view(-1, 1),
            self.one_tensor,
        )

        return (
            self.process(physics_loss, boundary_loss),
            self.inputs,
            f,
            self.inputs_mask,
        )

    def cosinus_loss(self: 'Loss') -> tuple[Tensor, Tensor, Tensor, None]:
        f = self.forward(self.inputs)
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
        return (
            self.process(physics_loss, boundary_loss),
            self.inputs,
            f,
            self.inputs_mask,
        )

    def potential_irrotational_flow_loss(
        self: 'Loss',
        pre: bool = False,
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor, Tensor]]:
        """
        (u, v) = nabla phi = (phi_x, phi_y)
        Observe: u_y = phi_xy = phi_yx = v_x

        Potential flow:
            u u_x + v u_y = u u_x + v v_x = p_x / rho
            u v_x + v v_y = u u_y + v v_y = p_y / rho
            therefore,
            1 / 2 u^2 + v^2 = p / rho
        """
        outputs = self.forward(self.inputs)
        phi = outputs[:, 0:1]
        u = self.partial_derivative(phi, self.x)
        v = self.partial_derivative(phi, self.y)
        ke = (u**2 + v**2)
        p = self.rho * ke / 2.0

        ic_loss, _, _ = self.incompressibility_loss(u, v)
        physics_loss = ic_loss

        # Inlet boundary condition
        # u(inlet) = 1 and v(inlet) = 0
        inlet_loss = (
            self.mse_loss(u[self.inlet_mask], self.inlet_one_tensor)
            + self.mse_loss(v[self.inlet_mask], self.inlet_zero_tensor)
        )
        # Outlet boundary condition
        # u(outlet) = 1 and v(outlet) = 0
        outlet_loss = (
            self.mse_loss(u[self.outlet_mask], self.outlet_one_tensor)
            + self.mse_loss(v[self.outlet_mask], self.outlet_zero_tensor)
        )
        # Wall boundary condition
        # v(wall) = 0
        wall_loss = (
            self.mse_loss(u[self.wall_mask], self.wall_one_tensor)
            + self.mse_loss(v[self.wall_mask], self.wall_zero_tensor)
        )

        boundary_loss = inlet_loss + outlet_loss + wall_loss

        if not pre:
            # Surface boundary condition
            # (u(airfoil), v(airfoil)) n_airfoil = 0
            airfoil_loss = (
                self.mse_loss(
                    u[self.airfoil_mask] * self.n_x,
                    -v[self.airfoil_mask] * self.n_y,
                )
            )
            boundary_loss += 3 * airfoil_loss

        airfoil_mask = self.airfoil_mask if not pre else None

        return (
            self.process(physics_loss, boundary_loss),
            self.inputs,
            (u, v, p),
            airfoil_mask,
        )

    def solenoidal_flow_loss(
        self: 'Loss',
        pre: bool = False,
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor, Tensor]]:
        """
        (u, v) = nabla^{perp} phi = (phi_y, -phi_x)
        Observe:
            u_x + v_y = phi_xy - phi_yx = 0 (if phi is C^2)

        NS PDE for pressure:
            u u_x + v u_y + nu (u_xx + u_yy) = p_x / rho
            u v_x + v v_y + nu (v_xx + v_yy) = p_y / rho
        Assuming nu = 0:
            u u_x + v u_y = p_x / rho
            u v_x - v u_x = p_y / rho
        """
        outputs = self.forward(self.inputs)
        phi = outputs[:, 0:1]
        u = self.partial_derivative(phi, self.y)
        v = self.partial_derivative(-phi, self.x)
        p = torch.zeros_like(u)

        u_y = self.partial_derivative(u, self.y)
        v_x = self.partial_derivative(-v, self.x)

        lap_phi = u_y + v_x
        lap_phi_x = self.partial_derivative(lap_phi, self.x)
        lap_phi_y = self.partial_derivative(lap_phi, self.x)

        physics_loss = self.mse_loss(
            u * lap_phi_x, - v * lap_phi_y
        )

        # Inlet boundary condition
        # u(inlet) = 1 and v(inlet) = 0
        inlet_loss = (
            self.mse_loss(u[self.inlet_mask], self.inlet_one_tensor)
            + self.mse_loss(v[self.inlet_mask], self.inlet_zero_tensor)
        )
        # Outlet boundary condition
        # u(outlet) = 1 and v(outlet) = 0
        outlet_loss = (
            self.mse_loss(u[self.outlet_mask], self.outlet_one_tensor)
            + self.mse_loss(v[self.outlet_mask], self.outlet_zero_tensor)
        )
        # Wall boundary condition
        # v(wall) = 0
        wall_loss = (
            self.mse_loss(u[self.wall_mask], self.wall_one_tensor)
            + self.mse_loss(v[self.wall_mask], self.wall_zero_tensor)
        )

        boundary_loss = inlet_loss + outlet_loss + wall_loss

        if not pre:
            # Surface boundary condition
            # u(airfoil) = v(airfoil) = 0
            # airfoil_loss = (
            #     self.mse_loss(u[self.airfoil_mask], self.airfoil_zero_tensor)
            #     + self.mse_loss(
            # v[self.airfoil_mask],
            # self.airfoil_zero_tensor
            # )
            # )
            airfoil_loss = (
                self.mse_loss(
                    u[self.airfoil_mask] * self.n_x,
                    -v[self.airfoil_mask] * self.n_y,
                )
            )

            boundary_loss += 3 * airfoil_loss

        airfoil_mask = self.airfoil_mask if not pre else None

        return (
            self.process(physics_loss, boundary_loss),
            self.inputs,
            (u, v, p),
            airfoil_mask,
        )

    def incompressibility_loss(
        self: 'Loss',
        u: Tensor,
        v: Tensor,
    ) -> tuple[float, Tensor, Tensor]:
        # u_x + v_y = 0 (Incompressibility)
        u_x = self.partial_derivative(u, self.x)
        v_y = self.partial_derivative(v, self.y)

        return self.mse_loss(u_x, -v_y), u_x, v_y
