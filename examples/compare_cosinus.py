"""Compare cumulative cosine derivative constraints using matched initializations.

uv run python -m examples.compare_cosinus --orders 2 4 6 8 10 12 --epochs 25000 --seeds 0
"""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

from learnpdes.training import build_problem
from learnpdes import COSINUS_SCENARIO, cosinus, device
from learnpdes.model.trainer import Trainer
from learnpdes.utils.artifacts import TrainingRun, problem_settings, atomic_json
from learnpdes.utils.cosinus_comparison import write_comparison
from learnpdes.utils.plot_style import COLORS, GREEN, scientific_style
from learnpdes.utils.interactive import InteractivePlot, cosinus_axes
from learnpdes.utils.visualization import ModelEvaluator, visualization_grid


def run(order, seed, epochs, points, output_dir, max_frames=100):
    """Start afresh for each order; no wider-domain targets enter training."""
    cosinus.validate_order(order)
    torch.manual_seed(seed)
    model, objective, analytical = build_problem(
        COSINUS_SCENARIO, points, cosinus_order=order
    )
    grid = visualization_grid(
        COSINUS_SCENARIO, objective.input_space, cosinus.EVALUATION_POINTS
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
                    for region, value in frame['evaluation_mse'].items()
                },
            }
            for frame in plotter.checkpoints
        ],
        'coordinates': grid.coordinates.ravel().tolist(),
        'final_prediction': plotter.checkpoints[-1]['prediction'].tolist(),
    }


def comparison_figure(runs):
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
    colors = COLORS
    orders = list(dict.fromkeys(run['order'] for run in runs))
    seeds = list(dict.fromkeys(run['seed'] for run in runs))
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
    return fig


def write_results(runs, output_dir, points):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--results',
        type=Path,
        help='Rebuild the viewer from saved results, without training.',
    )
    parser.add_argument(
        '--output-html', type=Path, help='Also export the viewer to this HTML path.'
    )
    parser.add_argument('--orders', type=int, nargs='+', default=[2, 4, 6, 8, 10, 12])
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    parser.add_argument('--epochs', type=int, default=25000)
    parser.add_argument('--points', type=int, default=64)
    parser.add_argument('--max-frames', type=int, default=100)
    parser.add_argument(
        '--output-dir',
        type=Path,
    )
    args = parser.parse_args()
    if args.results is not None:
        payload = json.loads(args.results.read_text())
        directory = args.output_dir or args.results.parent
        path = write_comparison(
            payload['runs'], args.output_html or directory / 'comparison.html'
        )
        print(f'Open {path}')
        return
    args.output_dir = args.output_dir or Path('assets')
    for order in args.orders:
        try:
            cosinus.validate_order(order)
        except ValueError as error:
            parser.error(str(error))
    if args.epochs < 1 or args.points < 3 or args.max_frames < 2:
        parser.error('Require --epochs >= 1, --points >= 3 and --max-frames >= 2.')
    torch.set_num_threads(1)
    comparison = TrainingRun(
        args.output_dir,
        'cosinus_derivatives',
        category='comparisons',
        settings={
            'orders': args.orders,
            'seeds': args.seeds,
            'epochs': args.epochs,
            'points': args.points,
            'max_frames': args.max_frames,
            'extra_checkpoint_steps': [5000] if args.epochs >= 5000 else [],
            'evaluation_bounds': cosinus.EVALUATION_BOUNDS,
            'evaluation_points': cosinus.EVALUATION_POINTS,
        },
    )
    args.output_dir = comparison.directory
    runs = []
    comparison.update(status='training', completed_runs=0)
    try:
        for seed in dict.fromkeys(args.seeds):
            for order in dict.fromkeys(args.orders):
                result = run(
                    order,
                    seed,
                    args.epochs,
                    args.points,
                    args.output_dir,
                    args.max_frames,
                )
                runs.append(result)
                comparison.update(status='exporting', completed_runs=len(runs))
                write_results(runs, args.output_dir, args.points)
                metrics = {
                    key: value
                    for key, value in result['history'][-1].items()
                    if key != 'prediction'
                }
                print(f'Order {order}, seed {seed}: {metrics}', flush=True)
                comparison.update(status='training')
        comparison.update(status='exporting')
        if args.output_html is not None:
            write_comparison(runs, args.output_html)
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
    print(f'Open {args.output_dir / "comparison.html"}')


if __name__ == '__main__':
    main()
