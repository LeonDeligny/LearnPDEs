"""Prescribed tunnel/airfoil boundary objectives for potential formulations."""

from __future__ import annotations

import torch
from torch import Tensor

from learnpdes.model.objectives import CollocationObjective
from learnpdes.types import LossFunction, LossResult
from learnpdes.utils.utility import compute_normals


class FlowObjective(CollocationObjective):
    def loss(self, pre: bool = False) -> LossResult:
        raise NotImplementedError('Select a potential or streamfunction objective.')

    def generate_boundaries(self) -> None:
        super().generate_boundaries()
        n_x, n_y = compute_normals(self.input_space, self.mesh_masks['airfoil'])
        self.n_x = n_x[self.mesh_masks['airfoil']].view(-1, 1).to(self.device)
        self.n_y = n_y[self.mesh_masks['airfoil']].view(-1, 1).to(self.device)

    def get_loss(self, scenario: str) -> LossFunction:
        if scenario != self.scenario:
            raise ValueError(f'Unknown flow scenario: {scenario}')
        return lambda: self.loss()

    def get_pre_loss(self, scenario: str) -> LossFunction:
        self.get_loss(scenario)  # Validate the scenario before binding pretraining.
        return lambda: self.loss(pre=True)

    def potential_irrotational_flow_loss(
        self,
        pre: bool = False,
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor, Tensor], Tensor | None]:
        """(u, v) = nabla phi = (phi_x, phi_y).

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
        ke = u**2 + v**2
        p = self.rho * ke / 2.0

        ic_loss, _, _ = self.incompressibility_loss(u, v)
        physics_loss = ic_loss

        # Inlet boundary condition
        # u(inlet) = 1 and v(inlet) = 0
        inlet_loss = self.mse_loss(
            u[self.inlet_mask], self.inlet_one_tensor
        ) + self.mse_loss(v[self.inlet_mask], self.inlet_zero_tensor)
        # Outlet boundary condition
        # u(outlet) = 1 and v(outlet) = 0
        outlet_loss = self.mse_loss(
            u[self.outlet_mask], self.outlet_one_tensor
        ) + self.mse_loss(v[self.outlet_mask], self.outlet_zero_tensor)
        # Wall boundary condition
        # v(wall) = 0
        wall_loss = self.mse_loss(
            u[self.wall_mask], self.wall_one_tensor
        ) + self.mse_loss(v[self.wall_mask], self.wall_zero_tensor)

        boundary_loss = inlet_loss + outlet_loss + wall_loss

        if not pre:
            # Surface boundary condition
            # (u(airfoil), v(airfoil)) n_airfoil = 0
            airfoil_loss = self.mse_loss(
                u[self.airfoil_mask] * self.n_x,
                -v[self.airfoil_mask] * self.n_y,
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
        self,
        pre: bool = False,
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor, Tensor], Tensor | None]:
        """(u, v) = nabla^{perp} phi = (phi_y, -phi_x).

        Observe:
            u_x + v_y = phi_xy - phi_yx = 0 (if phi is C^2)

        NS PDE for pressure:
            u u_x + v u_y + nu (u_xx + u_yy) = p_x / rho
            u v_x + v v_y + nu (v_xx + v_yy) = p_y / rho
        Assuming nu = 0:
            u u_x + v u_y = p_x / rho
            u v_x - v u_x = p_y / rho
        """
        phi = self.forward(self.inputs)[:, 0:1]
        u = self.partial_derivative(phi, self.y)
        v = self.partial_derivative(-phi, self.x)
        p = torch.zeros_like(u)

        u_y = self.partial_derivative(u, self.y)
        v_x = self.partial_derivative(-v, self.x)

        lap_phi = u_y + v_x
        lap_phi_x = self.partial_derivative(lap_phi, self.x)
        lap_phi_y = self.partial_derivative(lap_phi, self.y)

        physics_loss = self.mse_loss(u * lap_phi_x, -v * lap_phi_y)

        # Inlet boundary condition
        # u(inlet) = 1 and v(inlet) = 0
        inlet_loss = self.mse_loss(
            u[self.inlet_mask], self.inlet_one_tensor
        ) + self.mse_loss(v[self.inlet_mask], self.inlet_zero_tensor)
        # Outlet boundary condition
        # u(outlet) = 1 and v(outlet) = 0
        outlet_loss = self.mse_loss(
            u[self.outlet_mask], self.outlet_one_tensor
        ) + self.mse_loss(v[self.outlet_mask], self.outlet_zero_tensor)
        # Wall boundary condition
        # v(wall) = 0
        wall_loss = self.mse_loss(
            u[self.wall_mask], self.wall_one_tensor
        ) + self.mse_loss(v[self.wall_mask], self.wall_zero_tensor)

        boundary_loss = inlet_loss + outlet_loss + wall_loss

        if not pre:
            # Surface boundary condition
            # (u(airfoil), v(airfoil)) n_airfoil = 0
            airfoil_loss = self.mse_loss(
                u[self.airfoil_mask] * self.n_x,
                -v[self.airfoil_mask] * self.n_y,
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
        self,
        u: Tensor,
        v: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        # u_x + v_y = 0 (Incompressibility)
        u_x = self.partial_derivative(u, self.x)
        v_y = self.partial_derivative(v, self.y)

        return self.mse_loss(u_x, -v_y), u_x, v_y


class PotentialObjective(FlowObjective):
    loss = FlowObjective.potential_irrotational_flow_loss


class StreamfunctionObjective(FlowObjective):
    loss = FlowObjective.solenoidal_flow_loss
