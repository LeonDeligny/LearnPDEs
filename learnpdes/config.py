"""Shared run configuration; each scenario supplies its defaults and capabilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from learnpdes.scenarios.base import Scenario

import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from learnpdes.resources import DEFAULT_MESH


@dataclass(frozen=True)
class RunConfig:
    scenario: str
    epochs: int | None = None
    points: int | None = None
    learning_rate: float = 0.001
    hidden_dim: int | None = None
    hidden_layers: int = 4
    seed: int = 0
    threads: int = 1
    lbfgs_steps: int = 0
    resample_every: int = 100
    cosinus_order: int = 2
    resolution: int = 51
    max_frames: int = 40
    save_gif: bool = True
    output_dir: Path = Path('assets')
    mesh_path: Path | None = None

    def resolved(self) -> 'RunConfig':
        """Resolve scenario defaults and reject errors before allocating a run."""
        from learnpdes.scenarios.registry import get_scenario

        case = get_scenario(self.scenario)
        config = replace(
            self,
            scenario=case.name,
            epochs=case.epochs if self.epochs is None else self.epochs,
            points=case.points if self.points is None else self.points,
            hidden_dim=case.hidden_dim if self.hidden_dim is None else self.hidden_dim,
            output_dir=Path(self.output_dir),
            mesh_path=Path(self.mesh_path or DEFAULT_MESH).resolve()
            if case.mesh
            else None,
        )
        config._validate_counts()
        config._validate_numerics()
        config._validate_capabilities(case, supplied_mesh=self.mesh_path is not None)
        return config

    def _validate_counts(self) -> None:
        for field, minimum in (
            ('epochs', 1),
            ('points', 3),
            ('hidden_dim', 1),
            ('hidden_layers', 1),
            ('threads', 1),
            ('lbfgs_steps', 0),
            ('resample_every', 1),
            ('resolution', 2),
            ('max_frames', 2),
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(
                    f'--{field.replace("_", "-")} must be an integer >= {minimum}'
                )

    def _validate_numerics(self) -> None:
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError('--learning-rate must be finite and positive')
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError('--seed must be an integer in [0, 4294967295]')
        if (
            not isinstance(self.cosinus_order, int)
            or self.cosinus_order < 2
            or self.cosinus_order % 2
        ):
            raise ValueError('--cosinus-order must be an even integer >= 2')

    def _validate_capabilities(self, case: Scenario, *, supplied_mesh: bool) -> None:
        if not case.supports_derivative_order and self.cosinus_order != 2:
            raise ValueError('--cosinus-order only applies to cosinus')
        if self.lbfgs_steps and not case.fluid:
            raise ValueError(
                '--lbfgs-steps only applies to cylinder, kovasznay, and circular-couette'
            )
        if supplied_mesh and not case.mesh:
            raise ValueError(
                '--mesh only applies to potential-flow and solenoidal-flow'
            )
        if self.mesh_path is not None and not self.mesh_path.is_file():
            raise ValueError(f'Mesh file does not exist: {self.mesh_path}')

    def as_dict(self) -> dict[str, object]:
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(self).items()
        }
