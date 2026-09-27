"""Small, explicit contract between runnable scenarios and the shared runner."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from torch.nn import Module

from learnpdes.types import Analytical

if TYPE_CHECKING:
    from learnpdes.config import RunConfig
    from learnpdes.evaluation.fluid import FluidEvaluation
    from learnpdes.model.fluid import FluidObjective
    from learnpdes.model.objectives import CollocationObjective
    from learnpdes.model.pinn import PINN
    from learnpdes.visualization.grids import VisualizationGrid

    Problem = tuple[PINN, CollocationObjective | FluidObjective, Analytical | None]


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    points: int
    epochs: int = 5000
    hidden_dim: int = 20
    fluid: bool = False
    mesh: bool = False
    formulation: str | None = None
    build: Callable[['RunConfig'], 'Problem'] = field(kw_only=True, repr=False)
    grid: Callable[..., 'VisualizationGrid'] = field(kw_only=True, repr=False)
    evaluate: Callable[[Module, Analytical | None], dict[str, float]] | None = field(
        default=None, kw_only=True, repr=False
    )
    fluid_evaluation: 'FluidEvaluation | None' = field(
        default=None, kw_only=True, repr=False
    )
    output_kind: Literal[
        'scalar', 'velocity_pressure', 'potential', 'streamfunction'
    ] = field(default='scalar', kw_only=True)
    reference_type: Literal['analytical', 'none'] = field(default='none', kw_only=True)
    supports_derivative_order: bool = field(default=False, kw_only=True)

    @property
    def physics(self) -> str:
        return self.formulation or self.name

    def as_dict(self) -> dict[str, object]:
        """Keep the public CLI catalog JSON independent of executable hooks."""
        return {
            name: getattr(self, name)
            for name in (
                'name',
                'description',
                'points',
                'epochs',
                'hidden_dim',
                'fluid',
                'mesh',
                'formulation',
            )
        }
