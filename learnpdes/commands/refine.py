"""Refine an existing fluid run on a denser fixed collocation set with L-BFGS."""

import json
from pathlib import Path
from typing import Annotated, cast

import torch
import typer

from learnpdes.model.fluid import FluidObjective
from learnpdes.model.trainer import Trainer
from learnpdes.scenarios.registry import get_scenario
from learnpdes.training import build_problem
from learnpdes.utils.artifacts import TrainingRun, atomic_json, problem_settings
from learnpdes.utils.plot import get_plot_func
from learnpdes.visualization.fields import ModelEvaluator


def refine(
    run: Annotated[Path, typer.Argument()],
    points: Annotated[int, typer.Option(min=3)] = 81,
    lbfgs_steps: Annotated[int, typer.Option(min=1)] = 3000,
    seed: Annotated[int, typer.Option(min=0, max=2**32 - 1)] = 0,
    output_dir: Annotated[Path, typer.Option()] = Path('assets'),
) -> None:
    """Refine a completed fluid run on a denser collocation set with L-BFGS."""
    source = json.loads((run / 'run.json').read_text())
    if source['status'] != 'completed':
        raise typer.BadParameter('Select a completed run.', param_hint='run')
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    case = get_scenario(source['scenario'])
    fluid_evaluation = case.fluid_evaluation
    if fluid_evaluation is None:
        raise typer.BadParameter('Select a fluid scenario run.', param_hint='run')
    architecture = source['settings']['model']
    model, objective, analytical = build_problem(
        source['scenario'],
        points,
        hidden_dim=architecture['hidden_dim'],
        hidden_layers=architecture['num_hidden_layers'],
    )
    objective = cast(FluidObjective, objective)
    checkpoint = run / 'model.pt'
    model.load_state_dict(
        torch.load(checkpoint, weights_only=True, map_location='cpu')[
            'model_state_dict'
        ]
    )
    artifact_run = TrainingRun(
        output_dir,
        source['scenario'],
        settings=problem_settings(
            model,
            objective,
            seed=seed,
            points_per_axis=points,
            refined_from=str(checkpoint.resolve()),
        ),
    )
    grid = case.grid(objective.input_space, 41)
    trainer = Trainer(
        model.parameters,
        objective.loss,
        {'learning_rate': 0.001, 'epochs': 0, 'lbfgs_steps': lbfgs_steps},
        {
            'plot_func': get_plot_func(source['scenario']),
            'evaluate': ModelEvaluator(model, source['scenario'], grid),
            **artifact_run.plot_options(gif=False),
            'max_frames': 20,
        },
        analytical=analytical,
        run=artifact_run,
        model=model,
        objective=objective,
        validation=lambda: fluid_evaluation.evaluate(
            model, objective.problem, count=4096, seed=2027
        ),
    )
    trainer.train()
    report = fluid_evaluation.report(model, objective.problem, training_seed=seed)
    atomic_json(artifact_run.directory / 'verification.json', report)
    artifact_run.finish(verification=report)
    print(json.dumps(report, indent=2))
