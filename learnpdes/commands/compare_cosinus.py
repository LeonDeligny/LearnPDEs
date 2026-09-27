"""Compare cumulative cosine derivative constraints using matched initializations.

uv run learnpdes compare-cosinus --orders 2 --orders 4 --epochs 25000 --seeds 0
Repeat --orders and --seeds for multiple values; omitted orders default to 2–12 (even).
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any, cast

import numpy as np
import plotly.graph_objects as go
import torch
import typer
from plotly.subplots import make_subplots

from learnpdes import COSINUS_SCENARIO, device
from learnpdes.model.objectives import CollocationObjective
from learnpdes.model.trainer import Trainer
from learnpdes.scenarios import cosinus
from learnpdes.scenarios.registry import get_scenario
from learnpdes.training import build_problem
from learnpdes.utils.artifacts import TrainingRun, atomic_json, problem_settings
from learnpdes.utils.collocation import add_overlay, snapshot
from learnpdes.utils.cosinus_comparison import write_comparison
from learnpdes.utils.interactive import InteractivePlot, cosinus_axes
from learnpdes.utils.plot_style import COLORS, GREEN, scientific_style
from learnpdes.visualization.fields import ModelEvaluator


def run(
    order: int,
    seed: int,
    epochs: int,
    points: int,
    output_dir: str | Path,
    max_frames: int = 100,
) -> dict[str, Any]:
    """Start afresh for each order; no wider-domain targets enter training."""
    output_dir = Path(output_dir)
    cosinus.validate_order(order)
    torch.manual_seed(seed)
    model, objective, analytical = build_problem(
        COSINUS_SCENARIO, points, cosinus_order=order
    )
    objective = cast(CollocationObjective, objective)
    grid = get_scenario(COSINUS_SCENARIO).grid(
        objective.input_space, cosinus.EVALUATION_POINTS
    )
    plotter = InteractivePlot(COSINUS_SCENARIO, cosinus_order=order)
    artifact_run = TrainingRun(
        output_dir,
        COSINUS_SCENARIO,
        settings=problem_settings(
            model,
            objective,
            seed=seed,
            cosinus_order=order,
            points_per_axis=points,
            derivative_method='taylor' if objective.cosinus_derivatives else 'autograd',
        ),
    )
    filename = (
        (artifact_run.directory / 'training.html').relative_to(output_dir).as_posix()
    )
    trainer = Trainer(
        model.parameters,
        objective.cosinus_loss,
        {'learning_rate': 0.001, 'epochs': epochs},
        {
            'plot_func': plotter,
            'evaluate': ModelEvaluator(model, COSINUS_SCENARIO, grid),
            'max_frames': max_frames,
            **artifact_run.plot_options(gif=False),
        },
        analytical=analytical,
        run=artifact_run,
        model=model,
    )
    # Keep the previous experiment's budget available as an exact checkpoint.
    if epochs >= 5000:
        trainer.checkpoints.add(5000)
    started = perf_counter()
    trainer.train()
    return {
        'order': order,
        'seed': seed,
        'epochs': epochs,
        'training_samples': len(objective.inputs),
        'collocation': snapshot(objective.collocation_points()),
        'derivative_method': 'taylor' if objective.cosinus_derivatives else 'autograd',
        'elapsed_seconds': perf_counter() - started,
        'html': filename,
        'history': [
            {
                'step': frame['step'],
                'training_loss': frame['loss'],
                'prediction': frame['prediction'].tolist(),
                **{
                    f'{region}_mse': value
                    for region, value in frame.get('evaluation_mse', {}).items()
                },
            }
            for frame in plotter.checkpoints
        ],
        'coordinates': grid.coordinates.ravel().tolist(),
        'final_prediction': plotter.checkpoints[-1]['prediction'].tolist(),
    }


def _comparison_traces(
    fig: go.Figure,
    runs: Sequence[dict[str, Any]],
    regions: dict[str, str],
    orders: list[int],
    seeds: list[int],
) -> None:
    colors = COLORS
    dashes = ['solid', 'dash', 'dot', 'dashdot']
    for run in runs:
        name = f'Order {run["order"]} · seed {run["seed"]}'
        line = {
            'color': colors[orders.index(run['order']) % len(colors)],
            'dash': dashes[seeds.index(run['seed']) % len(dashes)],
        }
        fig.add_trace(
            go.Scatter(
                x=run['coordinates'],
                y=run['final_prediction'],
                name=name,
                legendgroup=name,
                line=line,
            ),
            row=1,
            col=1,
        )
        for row, region in enumerate(regions, 2):
            fig.add_trace(
                go.Scatter(
                    x=[frame['step'] for frame in run['history']],
                    y=[
                        max(frame[f'{region}_mse'], np.finfo(np.float32).tiny)
                        for frame in run['history']
                    ],
                    name=name,
                    legendgroup=name,
                    showlegend=False,
                    line=line,
                    hovertemplate='Step %{x:,.0f}<br>MSE %{y:.3e}<extra>%{fullData.name}</extra>',
                ),
                row=row,
                col=1,
            )


def comparison_figure(runs: Sequence[dict[str, Any]]) -> go.Figure:
    regions = {
        key: label
        for key, label in cosinus.REGION_LABELS.items()
        if f'{key}_mse' in runs[0]['history'][0]
    }
    fig = make_subplots(
        rows=1 + len(regions),
        cols=1,
        vertical_spacing=0.3 / len(regions),
        subplot_titles=[
            'Final predictions<br>Shaded: training [−π, π]',
            *[f'{label} · MSE' for label in regions.values()],
        ],
    )
    x = np.asarray(runs[0]['coordinates'])
    fig.add_trace(
        go.Scatter(
            x=x, y=np.cos(x), name='cos(x)', line={'color': GREEN, 'dash': 'dash'}
        ),
        row=1,
        col=1,
    )
    orders = list(dict.fromkeys(run['order'] for run in runs))
    seeds = list(dict.fromkeys(run['seed'] for run in runs))
    _comparison_traces(fig, runs, regions, orders, seeds)
    cosinus_axes(fig, rows=(1,))
    fig.update_xaxes(range=list(cosinus.EVALUATION_BOUNDS), row=1, col=1)
    fig.update_yaxes(title_text='f(x)', row=1, col=1)
    for row in range(2, len(regions) + 2):
        fig.update_xaxes(title_text='Training step', row=row, col=1)
        fig.update_yaxes(title_text='MSE', type='log', dtick=1, row=row, col=1)
    fig.update_layout(
        template='plotly_white',
        height=300 * (len(regions) + 1),
        title={
            'text': f'Cosine · {runs[0]["epochs"]:,} updates per run',
            'x': 0.03,
            'y': 0.98,
            'yanchor': 'top',
            'font': {'size': 17},
        },
        legend={'orientation': 'h', 'y': 1.035, 'yanchor': 'bottom'},
        margin={'t': 200, 'b': 65, 'l': 58, 'r': 20},
        font={'family': 'system-ui, sans-serif'},
    )
    scientific_style(fig)
    # Each run can use a different set; do not substitute the evaluation grid.
    groups = {
        f'order {run["order"]}, seed {run["seed"]}: {name}': group
        for run in runs
        for name, group in (run.get('collocation') or {}).items()
    }
    add_overlay(fig, groups or None, rug=True)
    return fig


def write_results(
    runs: Sequence[dict[str, Any]], output_dir: str | Path, points: int
) -> None:
    """Retain individual runs and their diagnostics without averaging away seeds."""
    output_dir = Path(output_dir)
    metadata = {
        'training_bounds': cosinus.TRAINING_BOUNDS,
        'evaluation_bounds': cosinus.EVALUATION_BOUNDS,
        'evaluation_points': cosinus.EVALUATION_POINTS,
        'evaluation_regions': cosinus.REGION_LABELS,
        'points': points,
        'optimizer': 'Adam',
        'learning_rate': 0.001,
        'loss': '3 * sum(even residual MSEs) + initial value + slope + added derivative anchors',
        'torch_version': torch.__version__,
        'device': str(device),
        'runs': runs,
    }
    atomic_json(output_dir / 'results.json', metadata)
    with (output_dir / 'summary.csv').open('w', newline='') as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                'order',
                'seed',
                'epochs',
                'elapsed_seconds',
                'training_loss',
                *[f'{region}_mse' for region in cosinus.REGION_LABELS],
            ],
        )
        writer.writeheader()
        for run in runs:
            final = run['history'][-1]
            writer.writerow(
                {
                    key: run[key] if key in run else final[key]
                    for key in writer.fieldnames
                }
            )
    write_comparison(runs, output_dir / 'comparison.html')


def _run_comparison(
    comparison: TrainingRun,
    orders: list[int],
    seeds: list[int],
    epochs: int,
    points: int,
    max_frames: int,
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for seed in dict.fromkeys(seeds):
        for order in dict.fromkeys(orders):
            result = run(
                order,
                seed,
                epochs,
                points,
                comparison.directory,
                max_frames,
            )
            runs.append(result)
            comparison.update(status='exporting', completed_runs=len(runs))
            write_results(runs, comparison.directory, points)
            metrics = {
                key: value
                for key, value in result['history'][-1].items()
                if key != 'prediction'
            }
            print(f'Order {order}, seed {seed}: {metrics}', flush=True)
            comparison.update(status='training')
    return runs


def _create_comparison_run(
    output_dir: Path,
    orders: list[int],
    seeds: list[int],
    epochs: int,
    points: int,
    max_frames: int,
) -> TrainingRun:
    for order in orders:
        try:
            cosinus.validate_order(order)
        except ValueError as error:
            raise typer.BadParameter(str(error), param_hint='--orders') from None
    return TrainingRun(
        output_dir,
        'cosinus_derivatives',
        category='comparisons',
        settings={
            'orders': orders,
            'seeds': seeds,
            'epochs': epochs,
            'points': points,
            'max_frames': max_frames,
            'extra_checkpoint_steps': [5000] if epochs >= 5000 else [],
            'evaluation_bounds': cosinus.EVALUATION_BOUNDS,
            'evaluation_points': cosinus.EVALUATION_POINTS,
        },
    )


def compare(
    results: Annotated[
        Path | None,
        typer.Option(help='Rebuild the viewer from saved results, without training.'),
    ] = None,
    output_html: Annotated[
        Path | None, typer.Option(help='Also export the viewer to this HTML path.')
    ] = None,
    orders: Annotated[
        list[int], typer.Option(help='Derivative orders; repeat for multiple values.')
    ] = [2, 4, 6, 8, 10, 12],
    seeds: Annotated[
        list[int], typer.Option(help='Random seeds; repeat for multiple values.')
    ] = [0],
    epochs: Annotated[int, typer.Option(min=1)] = 25000,
    points: Annotated[int, typer.Option(min=3)] = 64,
    max_frames: Annotated[int, typer.Option(min=2)] = 100,
    output_dir: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Compare cosine derivative orders with matched initializations.

    Repeat --orders and --seeds for multiple values; omitted orders default
    to 2–12 (even). Use --results to rebuild a saved viewer without training.
    """
    if results is not None:
        payload = json.loads(results.read_text())
        directory = output_dir or results.parent
        path = write_comparison(
            payload['runs'], output_html or directory / 'comparison.html'
        )
        print(f'Open {path}')
        return
    output_dir = output_dir or Path('assets')
    comparison = _create_comparison_run(
        output_dir, orders, seeds, epochs, points, max_frames
    )
    torch.set_num_threads(1)
    output_dir = comparison.directory
    runs = []
    comparison.update(status='training', completed_runs=0)
    try:
        runs = _run_comparison(comparison, orders, seeds, epochs, points, max_frames)
        comparison.update(status='exporting')
        if output_html is not None:
            write_comparison(runs, output_html)
        comparison.finish()
    except BaseException as error:
        try:
            comparison.update(
                status='interrupted'
                if isinstance(error, KeyboardInterrupt)
                else 'failed',
                error={'type': type(error).__name__, 'message': str(error)},
            )
        except Exception as save_error:
            error.add_note(f'Could not save the failure record: {save_error}')
        raise
    print(f'Open {output_dir / "comparison.html"}')
