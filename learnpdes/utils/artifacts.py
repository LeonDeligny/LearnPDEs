"""Isolated, portable training runs and explicit publication of example results."""

import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile
from uuid import uuid4

import plotly
import torch


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def slug(value):
    result = re.sub(r'[^a-z0-9_-]+', '_', value.lower()).strip('_-')
    if not result:
        raise ValueError('A run needs a nonempty scenario name.')
    return result


def atomic_json(path, data):
    """Readers see either the previous complete JSON document or the new one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            json.dump(data, file, indent=2, allow_nan=False)
            file.write('\n')
            file.flush()
            os.fsync(file.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def source_revision():
    root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty = subprocess.run(
            ['git', 'status', '--porcelain'],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        return {'commit': revision, 'dirty': bool(dirty)}
    except (OSError, subprocess.SubprocessError):
        return {'commit': None, 'dirty': None}


def problem_settings(model, objective, *, seed=None, **settings):
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
                    **asdict(objective.problem),
                    'interior_samples': objective.interior_count,
                    'samples_per_boundary': objective.boundary_count,
                    'boundary_weight': 10,
                }
            }
            if hasattr(objective, 'problem')
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

    def __init__(self, root, scenario, *, settings=None, category='runs'):
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

    def update(self, **values):
        self.manifest.update(values)
        atomic_json(self.directory / 'run.json', self.manifest)

    def plot_options(self, *, gif=True):
        return {
            'output_dir': self.directory,
            'html_path': self.directory / 'training.html',
            'gif_path': self.directory / 'training.gif' if gif else None,
            'frame_dir': self.directory / 'frames',
        }

    def save_training(
        self, history, model=None, optimizer=None, completed_steps=0, *, components=None
    ):
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

    def finish(self, **values):
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


def publish_run(directory, root='assets', *, replace=False):
    """Copy a completed run into the stable, version-controlled examples area.

    A lock prevents simultaneous publishers. Staging and rollback keep the
    previous example intact if copying or replacement fails.
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
            selected = {}
            for name, info in metadata['artifacts'].items():
                if name not in (
                    'training.gif',
                    'training.html',
                    'plotly.min.js',
                    'loss.csv',
                    'residuals.csv',
                ):
                    continue
                source = directory / name
                if hashlib.sha256(source.read_bytes()).hexdigest() != info['sha256']:
                    raise ValueError(f'Artifact checksum does not match: {name}')
                shutil.copy2(source, stage / name)
                selected[name] = info
            if 'training.html' not in selected or 'plotly.min.js' not in selected:
                raise ValueError(
                    'Publishing requires the HTML figure and its Plotly runtime.'
                )
            if (target / 'training.gif').exists() and 'training.gif' not in selected:
                raise ValueError(
                    'This example includes a GIF; publish a run with GIF export '
                    'enabled to keep its documentation links working.'
                )
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
