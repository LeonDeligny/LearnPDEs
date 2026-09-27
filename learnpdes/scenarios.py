"""Scenario catalog and reproducible training configuration.

Physical equations and boundary conditions live in the scenario modules and
loss implementations; this catalog defines the supported runnable cases.

Every scenario forbids simulation data in training and evaluation, including
published numerical benchmark scalars. Use analytical/manufactured references
only; cases without one remain exploratory.
"""

from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path
from types import MappingProxyType

from learnpdes import (
    COSINUS_SCENARIO,
    CYLINDER_SCENARIO,
    EXPONENTIAL_SCENARIO,
    KOVASZNAY_SCENARIO,
    LAPLACE_SCENARIO,
    POISEUILLE_SCENARIO,
    POTENTIAL_FLOW_SCENARIO,
    SOLENOIDAL_FLOW_SCENARIO,
)

DEFAULT_MESH = Path(__file__).parent / 'data' / 'mesh_airfoil_ch10sm.su2'
WIND_TUNNEL_SCENARIO = 'wind_tunnel_no_geometry'


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

    @property
    def physics(self) -> str:
        return self.formulation or self.name


SCENARIOS = MappingProxyType(
    {
        case.name: case
        for case in (
            Scenario(EXPONENTIAL_SCENARIO, "Exponential ODE: f'=f, f(0)=1", 256, 10000),
            Scenario(COSINUS_SCENARIO, "Harmonic oscillator: f''+f=0 on [-pi, pi]", 64),
            Scenario(LAPLACE_SCENARIO, 'Laplace equation on the unit square', 21),
            Scenario(POISEUILLE_SCENARIO, 'Pressure-driven viscous channel flow', 21),
            Scenario(
                KOVASZNAY_SCENARIO,
                'Navier-Stokes exact benchmark, Re=40',
                31,
                10000,
                64,
                fluid=True,
            ),
            Scenario(
                CYLINDER_SCENARIO,
                'Exploratory DFG cylinder, Re=20; no exact reference',
                45,
                hidden_dim=64,
                fluid=True,
            ),
            Scenario(
                WIND_TUNNEL_SCENARIO,
                'Potential flow in a channel without an obstacle',
                32,
                1000,
                formulation=POTENTIAL_FLOW_SCENARIO,
            ),
            Scenario(
                'potential-flow',
                'Exploratory airfoil potential flow; no exact reference',
                32,
                mesh=True,
                formulation=POTENTIAL_FLOW_SCENARIO,
            ),
            Scenario(
                'solenoidal-flow',
                'Exploratory airfoil streamfunction; no exact reference',
                32,
                mesh=True,
                formulation=SOLENOIDAL_FLOW_SCENARIO,
            ),
        )
    }
)


def get_scenario(name: str) -> Scenario:
    """Accept canonical CLI names and the original Python flow identifiers."""
    name = {
        POTENTIAL_FLOW_SCENARIO: 'potential-flow',
        SOLENOIDAL_FLOW_SCENARIO: 'solenoidal-flow',
    }.get(name, name)
    try:
        return SCENARIOS[name]
    except KeyError:
        raise ValueError(
            f'Unknown scenario {name!r}. Choose from: {", ".join(SCENARIOS)}'
        ) from None


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
            value = getattr(config, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(
                    f'--{field.replace("_", "-")} must be an integer >= {minimum}'
                )
        if not math.isfinite(config.learning_rate) or config.learning_rate <= 0:
            raise ValueError('--learning-rate must be finite and positive')
        if not isinstance(config.seed, int) or not 0 <= config.seed < 2**32:
            raise ValueError('--seed must be an integer in [0, 4294967295]')
        if (
            not isinstance(config.cosinus_order, int)
            or config.cosinus_order < 2
            or config.cosinus_order % 2
        ):
            raise ValueError('--cosinus-order must be an even integer >= 2')
        if case.name != COSINUS_SCENARIO and config.cosinus_order != 2:
            raise ValueError('--cosinus-order only applies to cosinus')
        if config.lbfgs_steps and not case.fluid:
            raise ValueError('--lbfgs-steps only applies to cylinder and kovasznay')
        if self.mesh_path is not None and not case.mesh:
            raise ValueError(
                '--mesh only applies to potential-flow and solenoidal-flow'
            )
        if config.mesh_path is not None and not config.mesh_path.is_file():
            raise ValueError(f'Mesh file does not exist: {config.mesh_path}')
        return config

    def as_dict(self) -> dict:
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(self).items()
        }
