"""Standalone field and equation-diagnostic plots from a saved fluid model.

These plots evaluate the network directly on a fresh grid. Solid points are
masked before model evaluation; no interpolation across the obstacle is used.
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

from learnpdes.fluid import CylinderProblem, build_fluid_problem
from learnpdes.fluid_evaluation import evaluate_fluid
from learnpdes.utils.artifacts import atomic_json


def sample_fields(model, problem, x_bounds, y_bounds, resolution=401):
    """Return physical fields and PDE residuals with NaNs in the solid."""
    if resolution < 3:
        raise ValueError('Plot resolution must be at least 3.')
    span = np.array([np.ptp(x_bounds), np.ptp(y_bounds)])
    counts = np.maximum(
        3, np.rint((resolution - 1) * span / span.max()).astype(int) + 1
    )
    x, y = np.linspace(*x_bounds, counts[0]), np.linspace(*y_bounds, counts[1])
    xx, yy = np.meshgrid(x, y)
    coordinates = torch.tensor(np.column_stack((xx.ravel(), yy.ravel())))
    inside = problem.contains(coordinates).numpy()
    parameter = next(model.parameters())
    fields = np.full((len(coordinates), 3), np.nan)
    residuals = np.full_like(fields, np.nan)
    indices = np.flatnonzero(inside)
    was_training = model.training
    model.eval()
    try:
        with torch.enable_grad():
            for selected in np.array_split(
                indices, max(1, int(np.ceil(len(indices) / 512)))
            ):
                xy = coordinates[selected].to(parameter).detach().requires_grad_(True)
                prediction = model(xy)
                terms = problem.residuals(xy, prediction)
                fields[selected] = prediction.detach().cpu().numpy()
                residuals[selected] = (
                    torch.cat(
                        [
                            terms[name]
                            for name in ('continuity', 'momentum_u', 'momentum_v')
                        ],
                        1,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
    finally:
        model.train(was_training)
    shape = (len(y), len(x), 3)
    return x, y, fields.reshape(shape), residuals.reshape(shape)


def _obstacle(figure, problem, row, col):
    if isinstance(problem, CylinderProblem):
        cx, cy = problem.center
        radius = problem.radius
        figure.add_shape(
            type='circle',
            x0=cx - radius,
            x1=cx + radius,
            y0=cy - radius,
            y1=cy + radius,
            fillcolor='#e2e8f0',
            line={'color': '#334155', 'width': 1.5},
            row=row,
            col=col,
        )


def field_figure(model, problem, *, resolution=601, full_domain=False):
    bounds = problem.x_bounds if full_domain else (0, 8)
    x, y, values, residuals = sample_fields(
        model, problem, bounds, problem.y_bounds, resolution
    )
    fields = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            'Speed |u|',
            'Pressure p',
            'Horizontal velocity u',
            'Vertical velocity v',
        ),
        horizontal_spacing=0.14,
        vertical_spacing=0.17,
    )
    speed = np.linalg.norm(values[:, :, :2], axis=2)
    for index, (z, title, colorscale) in enumerate(
        (
            (speed, '|u| / U', 'Viridis'),
            (values[:, :, 2], 'p / (ρU²)', 'Cividis'),
            (values[:, :, 0], 'u / U', 'Viridis'),
            (values[:, :, 1], 'v / U', 'RdBu_r'),
        )
    ):
        row, col = index // 2 + 1, index % 2 + 1
        diverging = title == 'v / U'
        limit = np.nanmax(abs(z))
        fields.add_trace(
            go.Heatmap(
                x=x,
                y=y,
                z=z,
                colorscale=colorscale,
                zsmooth=False,
                zmin=-limit if diverging else None,
                zmax=limit if diverging else None,
                colorbar={
                    'title': title,
                    'len': 0.34,
                    'thickness': 13,
                    'x': 0.43 if col == 1 else 1.02,
                    'y': 0.81 if row == 1 else 0.20,
                },
                hovertemplate='x=%{x:.3f}<br>y=%{y:.3f}<br>value=%{z:.5f}<extra></extra>',
            ),
            row=row,
            col=col,
        )
        _obstacle(fields, problem, row, col)
        axis = index + 1
        fields.update_xaxes(title_text='x / D', range=list(bounds), row=row, col=col)
        fields.update_yaxes(
            title_text='y / D',
            range=list(problem.y_bounds),
            scaleanchor=f'x{axis}' if axis > 1 else 'x',
            row=row,
            col=col,
        )
    fields.update_layout(
        title='Cylinder · Re = 20 · steady PINN prediction<br><sup>Exploratory: no exact solution reference; solid cylinder masked</sup>',
        template='plotly_white',
        width=1400,
        height=860,
        margin={'t': 105, 'b': 65, 'l': 70, 'r': 110},
    )
    diagnostics = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(
            'Continuity: log₁₀ |uₓ + vᵧ|',
            'Horizontal momentum: log₁₀ |rᵤ|',
            'Vertical momentum: log₁₀ |rᵥ|',
        ),
        vertical_spacing=0.09,
    )
    for i in range(3):
        diagnostics.add_trace(
            go.Heatmap(
                x=x,
                y=y,
                z=np.log10(np.maximum(abs(residuals[:, :, i]), 1e-7)),
                colorscale='Magma',
                zmin=-5,
                zmax=0,
                coloraxis='coloraxis',
            ),
            row=i + 1,
            col=1,
        )
        _obstacle(diagnostics, problem, i + 1, 1)
        diagnostics.update_yaxes(
            title_text='y / D',
            scaleanchor='x' if i == 0 else f'x{i + 1}',
            row=i + 1,
            col=1,
        )
        diagnostics.update_xaxes(title_text='x / D', row=i + 1, col=1)
    diagnostics.update_layout(
        title='Independent grid · absolute equation residuals<br><sup>Common logarithmic scale; values below 10⁻⁵ are clipped for display</sup>',
        template='plotly_white',
        width=1100,
        height=1500,
        coloraxis={
            'colorscale': 'Magma',
            'cmin': -5,
            'cmax': 0,
            'colorbar': {'title': 'log₁₀ |r|'},
        },
        margin={'t': 100},
    )
    return fields, diagnostics


def convergence_figure(path):
    with Path(path).open() as file:
        rows = list(csv.DictReader(file))
    figure = go.Figure()
    for name in (
        'continuity',
        'momentum_u',
        'momentum_v',
        'mass_flux',
        'outlet_normal',
        'outlet_tangent',
    ):
        if name in rows[0]:
            figure.add_scatter(
                x=[int(row['step']) for row in rows],
                y=[float(row[name]) for row in rows],
                name=name,
                mode='lines',
            )
    figure.update_layout(
        title='Training residuals · unweighted mean square',
        template='plotly_white',
        xaxis_title='Optimizer update',
        yaxis_title='MSE',
        yaxis_type='log',
        width=1100,
        height=600,
    )
    return figure


def export_cylinder_plots(directory, *, output_dir=None, resolution=601, png=False):
    """Reconstruct the saved architecture; recompute diagnostics, never old errors."""
    directory = Path(directory)
    metadata = json.loads((directory / 'run.json').read_text())
    if metadata['scenario'] != 'cylinder' or metadata['status'] != 'completed':
        raise ValueError('Select a completed cylinder run.')
    architecture = metadata['settings']['model']
    model, objective, _ = build_fluid_problem(
        'cylinder',
        3,
        hidden_dim=architecture['hidden_dim'],
        hidden_layers=architecture['num_hidden_layers'],
    )
    checkpoint = torch.load(
        directory / 'model.pt', map_location='cpu', weights_only=True
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cpu()
    output_dir = Path(output_dir or directory / 'plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = {}
    figures['flow-fields'], figures['equation-residuals'] = field_figure(
        model, objective.problem, resolution=resolution
    )
    figures['full-channel'], _ = field_figure(
        model, objective.problem, resolution=resolution, full_domain=True
    )
    figures['convergence'] = convergence_figure(directory / 'residuals.csv')
    for name, figure in figures.items():
        figure.write_html(output_dir / f'{name}.html', include_plotlyjs='directory')
        if png:
            figure.write_image(output_dir / f'{name}.png', scale=1.5)
    report = {
        'run_id': metadata['run_id'],
        'checkpoint_sha256': hashlib.sha256(
            (directory / 'model.pt').read_bytes()
        ).hexdigest(),
        'reference_type': 'none',
        'status': 'exploratory_accuracy_unverified',
        'interpretation': 'Residuals and conserved flux are diagnostics, not a bound on solution error. No simulation reference data used.',
        'evaluation': {'samples': 8192, 'boundary_samples_per_edge': 256, 'seed': 2027},
        'metrics': evaluate_fluid(model, objective.problem, count=8192, seed=2027),
        'units': {'length': 'D', 'velocity': 'mean inlet U', 'pressure': 'rho U^2'},
        'figures': [f'{name}.html' for name in figures],
    }
    atomic_json(output_dir / 'diagnostics.json', report)
    return output_dir
