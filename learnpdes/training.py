"""Scenario-independent construction adapters and artifact-producing training."""

from pathlib import Path

import numpy as np
import torch

from learnpdes.config import RunConfig
from learnpdes.model.fluid import FluidObjective
from learnpdes.model.objectives import CollocationObjective
from learnpdes.model.pinn import PINN
from learnpdes.model.trainer import Trainer
from learnpdes.scenarios.registry import get_scenario
from learnpdes.types import Analytical
from learnpdes.utils.artifacts import TrainingRun, atomic_json, problem_settings
from learnpdes.utils.plot import get_plot_func, require_gif_export
from learnpdes.visualization.fields import ModelEvaluator


def build_problem(
    scenario: str,
    points: int,
    *,
    cosinus_order: int = 2,
    hidden_dim: int | None = None,
    hidden_layers: int = 4,
    mesh_path: Path | None = None,
) -> tuple[PINN, CollocationObjective | FluidObjective, Analytical | None]:
    """Construct every registered case using the same defaults as the CLI."""
    config = RunConfig(
        scenario,
        points=points,
        cosinus_order=cosinus_order,
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        mesh_path=mesh_path,
    ).resolved()
    return get_scenario(config.scenario).build(config)


def evaluate(
    model: torch.nn.Module, scenario: str, analytical: Analytical | None
) -> dict[str, float]:
    """Use the selected scenario's independent evaluator, when available."""
    evaluator = get_scenario(scenario).evaluate
    return evaluator(model, analytical) if evaluator is not None else {}


def _finish_validation(
    trainer: Trainer,
    config: RunConfig,
    model: PINN,
    objective: CollocationObjective | FluidObjective,
    run: TrainingRun,
) -> None:
    case = get_scenario(config.scenario)
    fluid_evaluation = case.fluid_evaluation
    fluid_objective = objective if isinstance(objective, FluidObjective) else None
    if fluid_evaluation is not None and fluid_objective is not None:
        report = fluid_evaluation.report(
            model, fluid_objective.problem, training_seed=config.seed
        )
        atomic_json(run.directory / 'verification.json', report)
        run.finish(verification=report)
    print(f'Final loss: {trainer.loss_history[-1][1]:.6e}')
    final = trainer.validation_result or {}
    labels = {
        'rmse': 'Final RMSE',
        'max_error': 'Maximum absolute error',
        'relative_l2': 'Relative L2 error',
    }
    for name, value in final.items():
        print(f'{labels.get(name, name)}: {value:.6e}')


def train(config: RunConfig) -> Trainer:
    """Run one validated case and retain its resolved settings and artifacts."""
    config = config.resolved()
    if config.points is None or config.epochs is None:
        raise ValueError('Resolved configuration is missing a point count.')
    case = get_scenario(config.scenario)
    if config.save_gif:
        require_gif_export()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.set_num_threads(config.threads)
    model, objective, analytical = build_problem(
        case.name,
        config.points,
        cosinus_order=config.cosinus_order,
        hidden_dim=config.hidden_dim,
        hidden_layers=config.hidden_layers,
        mesh_path=config.mesh_path,
    )
    loss_function = objective.get_loss(case.physics)
    grid = case.grid(
        objective.input_space, config.resolution, mesh_path=config.mesh_path
    )
    has_validation = case.evaluate is not None
    if has_validation:
        initial = evaluate(model, case.name, analytical)
        if 'rmse' in initial:
            print(f'Initial RMSE: {initial["rmse"]:.6e}')
    model.train()
    run = TrainingRun(
        config.output_dir,
        case.name,
        settings=problem_settings(
            model,
            objective,
            seed=config.seed,
            points_per_axis=config.points,
            visualization_resolution=config.resolution,
            cosinus_order=config.cosinus_order,
            config=config.as_dict(),
        ),
    )
    trainer = Trainer(
        model.parameters,
        loss_function,
        {
            'learning_rate': config.learning_rate,
            'epochs': config.epochs,
            'lbfgs_steps': config.lbfgs_steps,
            'resample_every': config.resample_every,
        },
        {
            'plot_func': get_plot_func(
                case.physics, cosinus_order=config.cosinus_order
            ),
            'evaluate': ModelEvaluator(
                model, case.name, grid, density=objective.rho.item()
            ),
            **run.plot_options(gif=config.save_gif),
            'max_frames': config.max_frames,
        },
        analytical=analytical,
        run=run,
        model=model,
        validation=(lambda: evaluate(model, case.name, analytical))
        if has_validation
        else None,
        objective=objective if isinstance(objective, FluidObjective) else None,
        collocation=objective.collocation_points,
    )
    trainer.train()
    _finish_validation(trainer, config, model, objective, run)
    return trainer
