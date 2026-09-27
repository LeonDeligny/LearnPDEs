"""Reusable sampling types, Navier–Stokes objective, and model construction."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch

from learnpdes.types import Analytical, LossResult, PointGroup, PointGroups

if TYPE_CHECKING:
    from learnpdes.config import RunConfig
    from learnpdes.model.pinn import PINN


class FluidProblem(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def x_bounds(self) -> tuple[float, float]: ...

    @property
    def y_bounds(self) -> tuple[float, float]: ...

    def contains(self, xy: torch.Tensor) -> torch.Tensor: ...

    def sample(
        self,
        interior_count: int,
        boundary_count: int,
        *,
        generator: torch.Generator,
        near_cylinder: bool = True,
    ) -> 'FluidSamples': ...

    def residuals(
        self, xy: torch.Tensor, fields: torch.Tensor
    ) -> dict[str, torch.Tensor]: ...

    def boundary_residuals(
        self,
        forward: Callable[[torch.Tensor], torch.Tensor],
        boundary: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]: ...

    def conservation_residuals(
        self,
        forward: Callable[[torch.Tensor], torch.Tensor],
        template: torch.Tensor,
    ) -> dict[str, torch.Tensor]: ...


@dataclass
class FluidSamples:
    interior: torch.Tensor
    boundary: dict[str, torch.Tensor]


class FluidObjective:
    """Evaluate fresh graphs on samples owned/refreshed by the trainer."""

    cosinus_derivatives: Callable[[torch.Tensor, int], list[torch.Tensor]] | None = None

    def cosinus_loss(self) -> LossResult:
        return self.loss()

    def __init__(
        self,
        problem: FluidProblem,
        model: torch.nn.Module,
        points: int,
        *,
        boundary_points: int | None = None,
    ) -> None:
        """Bind a fluid model and draw the initial collocation samples."""
        self.problem, self.model = problem, model
        self.interior_count = points**2
        self.boundary_count = boundary_points or 4 * points
        self.generator = torch.Generator().manual_seed(torch.initial_seed())
        self.components: dict[str, float] = {}
        self.rho = torch.tensor(1.0)
        self.inputs: torch.Tensor
        self.input_space: torch.Tensor
        self.samples: FluidSamples
        self.resample()

    def resample(self) -> None:
        self.samples = self.problem.sample(
            self.interior_count,
            self.boundary_count,
            generator=self.generator,
        )
        # Domain corners are for metadata/visualization only, never PDE samples.
        self.input_space = torch.tensor(
            [[x, y] for x in self.problem.x_bounds for y in self.problem.y_bounds]
        )
        self.inputs = self.samples.interior

    def get_loss(
        self, scenario: str
    ) -> Callable[
        [], tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...], None]
    ]:
        if scenario != self.problem.name:
            raise ValueError('Scenario does not match the fluid problem.')
        return self.loss

    def collocation_points(self) -> PointGroups:
        parameter = next(self.model.parameters())
        groups: dict[str, PointGroup] = {
            'interior': {
                'kind': 'pde',
                'coordinates': self.samples.interior.to(parameter),
            }
        }
        groups.update(
            {
                name: {
                    'kind': 'anchor' if name == 'pressure' else 'boundary',
                    'coordinates': xy.to(parameter),
                }
                for name, xy in self.samples.boundary.items()
            }
        )
        quadrature = getattr(self.problem, 'collocation_quadrature', lambda: {})()
        groups.update(
            {
                name: {'kind': 'integral', 'coordinates': xy.to(parameter)}
                for name, xy in quadrature.items()
            }
        )
        return groups

    def loss(self) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...], None]:
        parameter = next(self.model.parameters())

        def fresh(xy: torch.Tensor) -> torch.Tensor:
            return xy.to(parameter).detach().clone().requires_grad_(True)

        xy = fresh(self.samples.interior)
        fields = self.model(xy)
        residuals = self.problem.residuals(xy, fields)
        residuals.update(
            self.problem.boundary_residuals(
                self.model,
                {name: fresh(values) for name, values in self.samples.boundary.items()},
            )
        )
        residuals.update(self.problem.conservation_residuals(self.model, xy))
        terms = {name: value.square().mean() for name, value in residuals.items()}
        self.components = {name: value.detach().item() for name, value in terms.items()}
        # Velocity boundary constraints receive extra weight in Kovasznay.
        total = sum(
            (
                value
                * (1 if name in ('continuity', 'momentum_u', 'momentum_v') else 10)
                for name, value in terms.items()
            ),
            torch.zeros((), device=xy.device),
        )
        return total, xy, tuple(fields.split(1, dim=1)), None


def build_fluid(
    problem: FluidProblem,
    config: 'RunConfig',
    analytical: Analytical | None = None,
) -> tuple['PINN', FluidObjective, Analytical | None]:
    from learnpdes import device
    from learnpdes.model.encodings import identity
    from learnpdes.model.pinn import PINN

    if config.points is None:
        raise ValueError('Fluid flow requires a point count.')
    if config.points < 3:
        raise ValueError('Fluid flow requires at least 3 points per axis.')
    if config.hidden_dim is None:
        raise ValueError('Fluid flow requires a hidden dimension.')
    model = PINN(
        {
            'input_dim': 2,
            'hidden_dim': config.hidden_dim,
            'output_dim': 3,
            'num_hidden_layers': config.hidden_layers,
            'activation': torch.nn.Tanh,
        },
        identity,
        identity,
        identity,
        output_transform=getattr(problem, 'output_transform', None),
    ).to(device)
    return model, FluidObjective(problem, model, config.points), analytical
