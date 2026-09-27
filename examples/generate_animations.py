"""Reproduce the four README animations with the training visualization pipeline.

Run from the repository root: uv run python -m examples.generate_animations
FFmpeg must be installed. PNG checkpoints are retained in gifs/<scenario>/epochs.
"""

import argparse
import json
import shutil
from functools import partial
from pathlib import Path

import numpy as np
import torch
from matplotlib.colors import Normalize

from examples.train_pinn import build_problem, evaluate
from learnpdes import POTENTIAL_FLOW_SCENARIO, device
from learnpdes.model.loss import Loss
from learnpdes.model.pinn import PINN
from learnpdes.model.trainer import Trainer
from learnpdes.utils.loadscenarios import load_wind_tunnel
from learnpdes.utils.plot import get_plot_func
from learnpdes.utils.visualization import (
    ModelEvaluator,
    VisualizationGrid,
    visualization_grid,
)

RUNS = {
    'exponential': (10000, 256),
    'cosinus': (5000, 64),
    'laplace': (5000, 21),
    'wind_tunnel_no_geometry': (1000, 32),
}


def wind_tunnel_problem(points):
    inputs, masks, output_dim, _, input_homeo, output_homeo, encoding = (
        load_wind_tunnel(points)
    )
    # An empty solid boundary allows use of the existing pre-training loss.
    masks['airfoil'] = torch.zeros(len(inputs), dtype=torch.bool)
    model = PINN(
        nn_params={
            'input_dim': 2,
            'hidden_dim': 20,
            'output_dim': output_dim,
            'num_hidden_layers': 4,
            'activation': torch.nn.Tanh,
        },
        input_homeo=input_homeo,
        output_homeo=output_homeo,
        encoding=encoding,
    ).to(device)
    objective = Loss(POTENTIAL_FLOW_SCENARIO, inputs, 2, model.forward, masks)
    return model, objective


def generate(name, epochs, points, output_dir, seed, max_frames):
    torch.manual_seed(seed)
    if name == 'wind_tunnel_no_geometry':
        model, objective = wind_tunnel_problem(points)
        scenario = POTENTIAL_FLOW_SCENARIO
        analytical = None
        loss_function = objective.get_pre_loss(scenario)
        x, y = np.meshgrid(
            np.linspace(0, 4, 600), np.linspace(0, 1, 150), indexing='ij'
        )
        grid = VisualizationGrid(np.column_stack((x.ravel(), y.ravel())))
        # Cache fluid connectivity once, rather than triangulating each frame.
        from matplotlib.tri import Triangulation

        grid.triangulation = Triangulation(
            grid.coordinates[:, 0], grid.coordinates[:, 1]
        )
    else:
        scenario = name
        model, objective, analytical = build_problem(name, points)
        loss_function = objective.get_loss(name)
        grid = visualization_grid(name, objective.input_space)
    plotter = get_plot_func(scenario)
    if name == 'wind_tunnel_no_geometry':
        # Fixed physical ranges include the expected unit inlet speed and its
        # pressure scale, avoiding saturation from tiny initial predictions.
        plotter = partial(
            plotter,
            color_norms={
                'u': Normalize(-1.1, 1.1),
                'v': Normalize(-1.1, 1.1),
                'p': Normalize(0, 1),
            },
        )
    trainer = Trainer(
        model.parameters,
        loss_function,
        {'learning_rate': 0.001, 'epochs': epochs},
        {
            'plot_func': plotter,
            'evaluate': ModelEvaluator(
                model, scenario, grid, density=objective.rho.item()
            ),
            'max_frames': max_frames,
            'output_dir': Path('gifs') / name / 'epochs',
            'gif_path': output_dir / f'{name}.gif',
        },
        analytical=analytical,
    )
    trainer.train()
    result = {
        'seed': seed,
        'epochs': epochs,
        'training_points_per_axis': points,
        'training_samples': len(objective.inputs),
        'visualization_samples': len(grid.coordinates),
        'checkpoint_steps': sorted(trainer.checkpoints),
        'final_loss': trainer.loss_history[-1][1],
        'frame_size': [1600, 1200 if name == 'wind_tunnel_no_geometry' else 900],
        'duration_ms': 100,
        'final_hold_ms': 2000,
        'torch_version': torch.__version__,
        'device': str(device),
    }
    if analytical is not None:
        result['validation'] = evaluate(model, name, analytical)
        result['validation']['grid'] = '41 x 41' if name == 'laplace' else '201 points'
    else:
        result['validation'] = None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', choices=['all', *RUNS], default='all')
    parser.add_argument(
        '--epochs', type=int, help='Override the scenario training duration.'
    )
    parser.add_argument('--max-frames', type=int, default=80)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output-dir', type=Path, default=Path('assets'))
    args = parser.parse_args()
    if args.epochs is not None and args.epochs < 1:
        parser.error('--epochs must be at least 1')
    if args.max_frames < 2:
        parser.error('--max-frames must be at least 2')
    if shutil.which('ffmpeg') is None:
        parser.error('Install FFmpeg before generating animations.')
    # These small networks are substantially faster with one CPU thread.
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / 'animation_runs.json'
    results = json.loads(manifest.read_text()) if manifest.exists() else {}
    for name in RUNS if args.scenario == 'all' else [args.scenario]:
        epochs, points = RUNS[name]
        results[name] = generate(
            name,
            args.epochs or epochs,
            points,
            args.output_dir,
            args.seed,
            args.max_frames,
        )
        manifest.write_text(json.dumps(results, indent=2) + '\n')
        print(f'{name}: {json.dumps(results[name])}', flush=True)


if __name__ == '__main__':
    main()
