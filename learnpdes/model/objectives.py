"""Shared machinery for objectives on fixed collocation grids."""

from __future__ import annotations

from collections.abc import Callable

import torch
from ambiance import Atmosphere
from numpy import array
from torch import Tensor, tensor
from torch.autograd import grad
from torch.nn import MSELoss

from learnpdes import device
from learnpdes.types import LossFunction, LossResult, PointGroup, PointGroups


class CollocationObjective:
    """Common collocation/autograd mechanics; subclasses supply the problem loss."""

    supports_derivative_order = False
    cosinus_derivatives: Callable[[Tensor, int], list[Tensor]] | None
    input_space: Tensor
    inputs: Tensor
    x: Tensor
    y: Tensor
    device = device
    mse_loss = MSELoss().to(device)
    zero = tensor([0.0]).to(device)
    one = tensor([1.0]).to(device)

    # Density of air at water level
    atm = Atmosphere(h=0.0)
    density = array([atm.density])
    rho = torch.tensor(density, dtype=torch.float32).to(device)

    def __init__(
        self,
        scenario: str,
        input_space: Tensor,
        input_dim: int,
        forward: Callable[[Tensor], Tensor],
        mesh_masks: dict[str, Tensor],
        *,
        cosinus_order: int = 2,
        cosinus_derivatives: Callable[[Tensor, int], list[Tensor]] | None = None,
    ) -> None:
        """Initialization of the loss."""
        if (
            isinstance(cosinus_order, bool)
            or not isinstance(cosinus_order, int)
            or cosinus_order < 2
            or cosinus_order % 2
        ):
            raise ValueError('Cosinus derivative order must be an even integer >= 2.')
        if not self.supports_derivative_order and cosinus_order != 2:
            raise ValueError('cosinus_order only applies to the cosinus scenario.')
        self.cosinus_order = cosinus_order
        self.cosinus_derivatives = cosinus_derivatives
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

    def process(self, physics_loss: Tensor, boundary_loss: Tensor) -> Tensor:
        """Process the losses to return a single loss value.

        TODO: Implement different methods to process the losses.
        """
        total_loss = 3 * physics_loss + boundary_loss
        return total_loss

    def collocation_points(self) -> PointGroups:
        # PDE evaluation includes boundary nodes in these fixed-grid objectives.
        # Keep every loss role even when coordinates overlap across groups.
        groups: dict[str, PointGroup] = {
            'equation': {'kind': 'pde', 'coordinates': self.inputs}
        }
        groups.update(
            {
                name: {
                    'kind': 'anchor' if name == 'zero' else 'boundary',
                    'coordinates': self.inputs[mask.to(self.inputs.device)],
                }
                for name, mask in self.mesh_masks.items()
                if mask.any()
            }
        )
        return groups

    def generate_inputs(self) -> None:
        """Generate input points based on the input space."""
        # Always maximum 3d physical space
        if self.dim == 1:
            self.input_space = self.input_space.unsqueeze(1)
        self.x = self.setup_space(index=0)
        self.y = self.setup_space(index=1)
        self.inputs = torch.cat([self.x, self.y], dim=1) if self.dim > 1 else self.x
        self.inputs_mask = (self.inputs < 10) & (self.inputs > -1)
        # self.z = self.setup_space(index=2)

    def setup_space(self, index: int) -> Tensor:
        if self.dim <= index:
            return torch.zeros_like(self.input_space[:, :1]).to(self.device)
        return self.input_space[:, index].requires_grad_().view(-1, 1).to(self.device)

    def generate_1d_boundaries(self) -> None:
        # x = 0
        self.zero_mask = self.mesh_masks['zero'].to(self.device)

        # f(x = 0)
        self.forward_null = self.forward(self.x[self.zero_mask])

        self.zero_tensor = self.zero.expand_as(self.forward_null).view(-1, 1).to(device)

        self.one_tensor = self.one.expand_as(self.forward_null).view(-1, 1).to(device)

    def generate_2d_boundaries(self) -> None:
        for name, mask in self.mesh_masks.items():
            if name == 'inlet':
                self.inlet_mask = mask.to(self.device)
                self.forward_inlet = self.forward(self.inputs[self.inlet_mask])[:, 0:1]
                self.inlet_zero_tensor = (
                    self.zero.expand_as(self.forward_inlet).view(-1, 1).to(device)
                )
                self.inlet_one_tensor = (
                    self.one.expand_as(self.forward_inlet).view(-1, 1).to(device)
                )
            elif name == 'outlet':
                self.outlet_mask = mask.to(self.device)
                self.forward_outlet = self.forward(self.inputs[self.outlet_mask])[
                    :, 0:1
                ]
                self.outlet_zero_tensor = (
                    self.zero.expand_as(self.forward_outlet).view(-1, 1).to(device)
                )
                self.outlet_one_tensor = (
                    self.one.expand_as(self.forward_outlet).view(-1, 1).to(device)
                )
            elif name == 'wall':
                self.wall_mask = mask.to(self.device)
                self.forward_wall = self.forward(self.inputs[self.wall_mask])[:, 0:1]
                self.wall_zero_tensor = (
                    self.zero.expand_as(self.forward_wall).view(-1, 1).to(device)
                )
                self.wall_one_tensor = (
                    self.one.expand_as(self.forward_wall).view(-1, 1).to(device)
                )
            elif name == 'top':
                self.top_mask = mask.to(self.device)
                self.forward_top = self.forward(self.inputs[self.top_mask])[:, 0:1]
                self.top_zero_tensor = (
                    self.zero.expand_as(self.forward_top).view(-1, 1).to(device)
                )
            elif name == 'bottom':
                self.bottom_mask = mask.to(self.device)
                self.forward_bottom = self.forward(self.inputs[self.bottom_mask])[
                    :, 0:1
                ]
                self.bottom_zero_tensor = (
                    self.zero.expand_as(self.forward_bottom).view(-1, 1).to(device)
                )
            elif name == 'airfoil':
                self.airfoil_mask = mask.to(self.device)
                self.forward_airfoil = self.forward(self.inputs[self.airfoil_mask])[
                    :, 0:1
                ]
                self.airfoil_zero_tensor = (
                    self.zero.expand_as(self.forward_airfoil).view(-1, 1).to(device)
                )
            else:
                raise ValueError(f'{name=} not known as a boundary name.')

    def generate_boundaries(self) -> None:
        if self.dim == 1:
            self.generate_1d_boundaries()

        elif self.dim == 2:
            self.generate_2d_boundaries()

        else:
            raise ValueError(f'{self.dim=} should be either 1 or 2.')

    def partial_derivative(self, f: Tensor, x: Tensor) -> Tensor:
        """Compute the first derivative of 1D outputs with respect to the inputs."""
        # Exact fields can be constant or affine: their derivatives must still
        # support further differentiation when evaluating viscous residuals.
        if not f.requires_grad:
            return x * 0
        derivative = grad(
            outputs=f,
            inputs=x,
            grad_outputs=torch.ones_like(f),
            create_graph=True,
            allow_unused=True,
        )[0]
        return (x * 0 if derivative is None else derivative).view(-1, 1)

    def get_loss(self, scenario: str) -> LossFunction:
        if scenario != self.scenario:
            raise ValueError('Scenario does not match the collocation objective.')
        return self.loss

    def get_pre_loss(self, scenario: str) -> LossFunction:
        return self.get_loss(scenario)

    def loss(self) -> LossResult:
        raise NotImplementedError(
            'A scenario must supply its physics and boundary loss.'
        )

    def cosinus_loss(self) -> LossResult:
        return self.loss()

    def poiseuille_loss(self) -> LossResult:
        return self.loss()
