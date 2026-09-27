"""Isolated, portable training runs and explicit publication of example results."""

import csv
import hashlib
import json
import os
import platform
import re
import shutil

# Git metadata uses fixed arguments, an absolute executable and no command shell.
import subprocess  # nosec B404
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

import plotly
import torch

from learnpdes.model.fluid import FluidObjective
from learnpdes.types import ExportPaths

if TYPE_CHECKING:
    from learnpdes.model.objectives import CollocationObjective
    from learnpdes.model.pinn import PINN


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slug(value: str) -> str:
    result = re.sub(r'[^a-z0-9_-]+', '_', value.lower()).strip('_-')
    if not result:
        raise ValueError('A run needs a nonempty scenario name.')
    return result


def atomic_json(path: str | Path, data: object) -> None:
    """Readers see either the previous complete JSON document or the new one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, 'w') as file:
            json.dump(data, file, indent=2, allow_nan=False)
            file.write('\n')
            file.flush()
            os.fsync(file.fileno())
        # Close the file before replacing it, including on Windows.
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def source_revision() -> dict[str, str | bool | None]:
    root = Path(__file__).resolve().parents[2]
    try:
        executable = shutil.which('git')
        if executable is None:
            return {'commit': None, 'dirty': None}
        # Resolve PATH once, before changing cwd; disable repository fsmonitor hooks.
        git = [str(Path(executable).resolve()), '-c', 'core.fsmonitor=false']
        # Both Git commands are fixed and contain no user-supplied arguments.
        revision = subprocess.run(  # nosec B603
            [*git, 'rev-parse', 'HEAD'],
            cwd=root,
            shell=False,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty = subprocess.run(  # nosec B603
            [*git, 'status', '--porcelain'],
            cwd=root,
            shell=False,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        return {'commit': revision, 'dirty': bool(dirty)}
    except (OSError, subprocess.SubprocessError):
        return {'commit': None, 'dirty': None}


def problem_settings(
    model: 'PINN',
    objective: 'CollocationObjective | FluidObjective',
    *,
    seed: int | None = None,
    **settings: object,
) -> dict[str, object]:
    coordinates = objective.inputs.detach()
    return {
        'seed': seed,
        'device': str(next(model.parameters()).device),
        'dtype': str(next(model.parameters()).dtype),
        'model': {
            'input_dim': model.input_dim,
            'output_dim': model.output_dim,
            'hidden_dim': model.hidden_dim,
            'num_hidden_layers': model.num_hidden_layers,
            'activation': model.activation.__name__,
        },
        'training_samples': len(coordinates),
        **(
            {
                'fluid': {
                    **asdict(cast(Any, objective.problem)),
                    'interior_samples': objective.interior_count,
                    'samples_per_boundary': objective.boundary_count,
                    'boundary_weight': 10,
                }
            }
            if isinstance(objective, FluidObjective)
            else {}
        ),
        'training_bounds': [
            coordinates.min(dim=0).values.cpu().tolist(),
            coordinates.max(dim=0).values.cpu().tolist(),
        ],
        **settings,
    }


class TrainingRun:
    """A new directory per invocation; a completed run is never reused."""

    def __init__(
        self,
        root: str | Path,
        scenario: str,
        *,
        settings: dict[str, object] | None = None,
        category: str = 'runs',
    ) -> None:
        """Create an isolated run directory and write its initial manifest."""
        if category not in ('runs', 'comparisons'):
            raise ValueError('Unknown run category.')
        self.scenario = slug(scenario)
        self.id = (
            datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
            + '-'
            + uuid4().hex[:8]
        )
        self.directory = Path(root) / category / self.scenario / self.id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.manifest = {
            'schema_version': 1,
            'run_id': self.id,
            'scenario': scenario,
            'status': 'created',
            'created_at': utc_now(),
            'settings': dict(settings or {}),
            'environment': {
                'python': platform.python_version(),
                'torch': torch.__version__,
                'plotly': plotly.__version__,
                'platform': platform.platform(),
            },
            'source': source_revision(),
            'artifacts': {},
        }
        self.update()

    def update(self, **values: object) -> None:
        self.manifest.update(values)
        atomic_json(self.directory / 'run.json', self.manifest)

    def plot_options(self, *, gif: bool = True) -> ExportPaths:
        return {
            'output_dir': self.directory,
            'html_path': self.directory / 'training.html',
            'gif_path': self.directory / 'training.gif' if gif else None,
            'frame_dir': self.directory / 'frames',
        }

    def save_training(
        self,
        history: list[tuple[int, float]],
        model: torch.nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        completed_steps: int = 0,
        *,
        components: list[dict[str, object]] | None = None,
    ) -> None:
        """Persist numerical results before rendering can fail."""
        path = self.directory / 'loss.csv'
        with path.open('w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(('step', 'loss'))
            writer.writerows(history)
        if components:
            with (self.directory / 'residuals.csv').open('w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=list(components[0]))
                writer.writeheader()
                writer.writerows(components)
        if model is not None:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
                'completed_steps': completed_steps,
            }
            temporary = self.directory / '.model.pt.tmp'
            torch.save(checkpoint, temporary)
            temporary.replace(self.directory / 'model.pt')

    def finish(self, **values: object) -> None:
        artifacts = {}
        for name in (
            'training.html',
            'training.gif',
            'plotly.min.js',
            'loss.csv',
            'residuals.csv',
            'model.pt',
            'comparison.html',
            'summary.csv',
            'results.json',
            'verification.json',
            'collocation.json',
        ):
            path = self.directory / name
            if path.is_file():
                artifacts[name] = {
                    'bytes': path.stat().st_size,
                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                }
        self.update(
            status='completed', completed_at=utc_now(), artifacts=artifacts, **values
        )
        # Each run has its own manifest. The small discovery pointer is replaced
        # atomically only after success; failed runs cannot hide a good result.
        atomic_json(self.directory.parent / 'latest.json', {'run_id': self.id})


def _publish_plots(
    directory: Path, stage: Path, metadata: dict[str, Any]
) -> dict[str, Any]:
    """Publish only plots recorded for the selected run's checkpoint."""
    diagnostics = directory / 'plots/diagnostics.json'
    if not diagnostics.is_file():
        return {}
    report = json.loads(diagnostics.read_text())
    checkpoint_hash = metadata['artifacts'].get('model.pt', {}).get('sha256')
    if (
        report.get('run_id') != metadata['run_id']
        or not checkpoint_hash
        or report.get('checkpoint_sha256') != checkpoint_hash
    ):
        raise ValueError('Plots must come from the selected run and checkpoint.')
    names = {'diagnostics.json', 'plotly.min.js'}
    for name in report['figures']:
        if Path(name).name != name or Path(name).suffix != '.html':
            raise ValueError(f'Invalid plot filename: {name}')
        names.add(name)
        png = Path(name).with_suffix('.png').name
        if (directory / 'plots' / png).is_file():
            names.add(png)
    (stage / 'plots').mkdir()
    selected = {}
    for name in sorted(names):
        relative = f'plots/{name}'
        source = directory / relative
        selected[relative] = {
            'bytes': source.stat().st_size,
            'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        }
        shutil.copy2(source, stage / relative)
    return selected


def _copy_training_artifacts(
    directory: Path, stage: Path, metadata: dict[str, Any]
) -> dict[str, Any]:
    selected = {}
    for name, info in metadata['artifacts'].items():
        if name not in (
            'training.gif',
            'training.html',
            'plotly.min.js',
            'loss.csv',
            'residuals.csv',
            'collocation.json',
        ):
            continue
        source = directory / name
        if hashlib.sha256(source.read_bytes()).hexdigest() != info['sha256']:
            raise ValueError(f'Artifact checksum does not match: {name}')
        shutil.copy2(source, stage / name)
        selected[name] = info
    return selected


def _validate_replacement(target: Path, selected: dict[str, Any]) -> None:
    if 'training.html' not in selected or 'plotly.min.js' not in selected:
        raise ValueError('Publishing requires the HTML figure and its Plotly runtime.')
    if (target / 'training.gif').exists() and 'training.gif' not in selected:
        raise ValueError(
            'This example includes a GIF; publish a run with GIF export '
            'enabled to keep its documentation links working.'
        )
    missing_plots = [
        path.relative_to(target).as_posix()
        for path in (target / 'plots').rglob('*')
        if path.suffix in ('.html', '.png')
        and path.relative_to(target).as_posix() not in selected
    ]
    if missing_plots:
        raise ValueError(
            'Regenerate the existing published plots before replacing this '
            f'example: {", ".join(sorted(missing_plots))}'
        )


def publish_run(
    directory: str | Path, root: str | Path = 'assets', *, replace: bool = False
) -> Path:
    """Copy a completed run into the stable, version-controlled examples area.

    Include any saved diagnostic plots from the same checkpoint. A lock prevents
    simultaneous publishers; staging and rollback preserve the previous example
    if copying or replacement fails.
    """
    directory = Path(directory)
    metadata = json.loads((directory / 'run.json').read_text())
    if metadata.get('status') != 'completed':
        raise ValueError('Only completed runs can be published.')
    scenario = slug(metadata['scenario'])
    parent = Path(root) / 'examples'
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / scenario
    lock = parent / f'.{scenario}.publish-lock'
    lock.mkdir()  # Fail rather than race another publisher.
    try:
        if target.exists() and not replace:
            raise FileExistsError(
                f'{target} already exists; use --replace to update it.'
            )
        with tempfile.TemporaryDirectory(
            prefix=f'.{scenario}-', dir=parent
        ) as temporary:
            stage = Path(temporary) / 'new'
            stage.mkdir()
            selected = _copy_training_artifacts(directory, stage, metadata)
            selected.update(_publish_plots(directory, stage, metadata))
            _validate_replacement(target, selected)
            atomic_json(stage / 'run.json', {**metadata, 'artifacts': selected})
            backup = Path(temporary) / 'previous'
            if target.exists():
                target.rename(backup)
            try:
                stage.rename(target)
            except BaseException:
                if backup.exists():
                    backup.rename(target)
                raise
    finally:
        lock.rmdir()
    return target
