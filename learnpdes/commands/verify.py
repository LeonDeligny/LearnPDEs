"""Write separate verification records for completed fluid scenario runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Any, cast

import torch
import typer

from learnpdes.model.fluid import FluidObjective
from learnpdes.scenarios.registry import get_scenario
from learnpdes.training import build_problem
from learnpdes.utils.artifacts import atomic_json, source_revision, utc_now


def verify_run(directory: str | Path, output_dir: str | Path) -> dict[str, Any]:
    directory = Path(directory)
    metadata = json.loads((directory / 'run.json').read_text())
    if metadata['status'] != 'completed':
        raise ValueError('Verification requires a completed run.')
    case = get_scenario(metadata['scenario'])
    fluid_evaluation = case.fluid_evaluation
    if fluid_evaluation is None:
        raise ValueError('Verification requires a fluid scenario run.')
    architecture = metadata['settings']['model']
    model, objective, _ = build_problem(
        metadata['scenario'],
        3,
        hidden_dim=architecture['hidden_dim'],
        hidden_layers=architecture['num_hidden_layers'],
    )
    objective = cast(FluidObjective, objective)
    checkpoint = directory / 'model.pt'
    model.load_state_dict(
        torch.load(checkpoint, map_location='cpu', weights_only=True)[
            'model_state_dict'
        ]
    )
    report = fluid_evaluation.report(
        model, objective.problem, training_seed=metadata['settings'].get('seed')
    )
    report.update(
        {
            'run_id': metadata['run_id'],
            'training_settings': metadata['settings'],
            'training': metadata['training'],
            'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'evaluated_at': utc_now(),
            'evaluator_source': source_revision(),
        }
    )
    destination = Path(output_dir) / metadata['scenario'] / f'{metadata["run_id"]}.json'
    atomic_json(destination, report)
    return {
        'scenario': metadata['scenario'],
        'run_id': metadata['run_id'],
        'report': str(destination),
        'single_run_checks_passed': report['single_run_checks_passed'],
        'reference_type': report['reference_type'],
        'project_acceptance': False,
    }


def verify(
    runs: Annotated[list[Path], typer.Argument(help='Completed run directories.')],
    output_dir: Annotated[Path, typer.Option()] = Path('assets/verification'),
) -> None:
    """Write separate verification records for completed fluid scenario runs."""
    torch.set_num_threads(1)
    records = [verify_run(directory, output_dir) for directory in runs]
    atomic_json(output_dir / 'index.json', {'scenarios': records})
    print(json.dumps(records, indent=2))
