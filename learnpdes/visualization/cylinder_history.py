"""Render real archived cylinder predictions without retraining or interpolating time."""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from numpy.typing import ArrayLike
from plotly.subplots import make_subplots

from learnpdes.types import Array, LossHistory, RecordedPoints
from learnpdes.utils.artifacts import atomic_json
from learnpdes.utils.collocation import add_overlay, point_traces
from learnpdes.utils.plot import create_gif, require_gif_export
from learnpdes.utils.plot_style import scientific_style


class Archive(TypedDict):
    directory: Path
    metadata: dict[str, Any]
    x: Array
    y: Array
    steps: list[int]
    predictions: list[Array]
    loss: LossHistory
    collocation: dict[int, RecordedPoints | None]


class HistoryFrame(TypedDict):
    step: int
    local_step: int
    stage: int
    phase: str
    count: int
    loss: float
    fields: tuple[Array, Array, Array, Array]
    collocation: RecordedPoints | None


def _array(value: ArrayLike | dict[str, Any]) -> Array:
    if not isinstance(value, dict):
        return np.asarray(value)
    result = np.frombuffer(base64.b64decode(value['bdata']), dtype=value['dtype'])
    if 'shape' in value:
        result = result.reshape(tuple(map(int, value['shape'].split(','))))
    return result


def _verify_archive(directory: Path, metadata: dict[str, Any]) -> None:
    for name in ('training.html', 'model.pt', 'loss.csv'):
        expected = metadata['artifacts'][name]['sha256']
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Archived {name} does not match the recorded checksum.')


def _archived_predictions(frames: list[dict[str, Any]]) -> list[Array]:
    predictions = []
    for frame in frames:
        traces = dict(zip(frame['traces'], frame['data']))
        predictions.append(np.stack([_array(traces[i]['z']) for i in range(3)]))
    return predictions


def _archive(directory: str | Path) -> Archive:
    """Read Plotly's JSON arguments as data; never execute archived JavaScript."""
    directory = Path(directory)
    metadata = json.loads((directory / 'run.json').read_text())
    if metadata['scenario'] != 'cylinder' or metadata['status'] != 'completed':
        raise ValueError('Choose a completed cylinder run.')
    _verify_archive(directory, metadata)
    html = (directory / 'training.html').read_text()
    decoder = json.JSONDecoder()
    start = html.index('[', html.index('Plotly.newPlot('))
    data, _ = decoder.raw_decode(html[start:])
    start = html.index('[', html.index('Plotly.addFrames('))
    frames, _ = decoder.raw_decode(html[start:])
    if [int(f['name']) for f in frames] != metadata['export']['checkpoint_steps']:
        raise ValueError('Archived frame steps do not match the run record.')
    if [trace.get('name') for trace in data[:3]] != ['u', 'v', 'p']:
        raise ValueError('Expected archived u, v, p heatmaps.')
    with (directory / 'loss.csv').open() as file:
        loss = [(int(r['step']), float(r['loss'])) for r in csv.DictReader(file)]
    predictions = _archived_predictions(frames)
    points = {}
    if (directory / 'collocation.json').is_file():
        recording = json.loads((directory / 'collocation.json').read_text())
        points = {
            c['step']: recording['point_sets'][c['point_set']]
            for c in recording['checkpoints']
        }
    return {
        'directory': directory,
        'metadata': metadata,
        'x': _array(data[0]['x']),
        'y': _array(data[0]['y']),
        'steps': [int(f['name']) for f in frames],
        'predictions': predictions,
        'loss': loss,
        'collocation': points,
    }


def _load_stages(directory: str | Path) -> list[Archive]:
    stages: list[Archive] = []
    seen: set[Path] = set()
    current = Path(directory).resolve()
    while True:
        if current in seen:
            raise ValueError('Cyclic refinement history.')
        seen.add(current)
        stage = _archive(current)
        stages.insert(0, stage)
        parent = stage['metadata']['settings'].get('refined_from')
        if not parent:
            break
        # A run can be moved together with its sibling runs; preserve that use.
        candidate = Path(parent).parent
        sibling = current.parent / candidate.name
        current = (sibling if sibling.is_dir() else candidate).resolve()
    for before, after in zip(stages, stages[1:]):
        for axis in ('x', 'y'):
            if not np.array_equal(before[axis], after[axis]):
                raise ValueError(
                    'Stage display grids differ; cannot join saved fields.'
                )
        if not np.array_equal(
            before['predictions'][-1], after['predictions'][0], equal_nan=True
        ):
            raise ValueError('Refinement does not start from the preceding prediction.')

    return stages


def _history_records(
    stages: list[Archive],
) -> tuple[Array, Array, list[HistoryFrame], list[tuple[str, LossHistory]], int]:
    x, y = stages[0]['x'], stages[0]['y']
    selected = (x >= 0) & (x <= 8)
    x = x[selected]
    records: list[HistoryFrame] = []
    histories: list[tuple[str, LossHistory]] = []
    offset = 0
    for index, stage in enumerate(stages):
        settings, training = (
            stage['metadata']['settings'],
            stage['metadata']['training'],
        )
        count = settings['training_samples']
        label = (
            f'{"Initial training" if index == 0 else "Refinement"} · {count:,} points'
        )
        histories.append(
            (label, [(step + offset, loss) for step, loss in stage['loss']])
        )
        losses = dict(stage['loss'])
        for step, prediction in zip(stage['steps'], stage['predictions']):
            u, v, p = prediction[:, :, selected]
            phase = 'Adam' if step < training['epochs'] else 'L-BFGS'
            if index:
                phase += ' refinement'
            records.append(
                {
                    'step': step + offset,
                    'local_step': step,
                    'stage': index,
                    'phase': phase,
                    'count': count,
                    'loss': losses[step],
                    'fields': (np.hypot(u, v), p, u, v),
                    'collocation': stage['collocation'].get(step),
                }
            )
        offset += stage['metadata']['completed_steps']
    return x, y, records, histories, offset


def _field_ranges(records: list[HistoryFrame]) -> list[tuple[float, float]]:
    ranges: list[tuple[float, float]] = []
    for i in range(4):
        low = min(float(np.nanmin(r['fields'][i])) for r in records)
        high = max(float(np.nanmax(r['fields'][i])) for r in records)
        if i == 3:
            high = max(abs(low), abs(high))
            low = -high
        ranges.append((low, high))

    return ranges


def _loss_range(histories: list[tuple[str, LossHistory]]) -> list[float]:
    losses = [value for _, history in histories for _, value in history]
    return [float(np.log10(min(losses))) - 0.15, float(np.log10(max(losses))) + 0.15]


def _history_fields(
    figure: go.Figure,
    x: Array,
    y: Array,
    records: list[HistoryFrame],
    ranges: list[tuple[float, float]],
) -> None:
    for i in range(4):
        row, col = i // 2 + 1, i % 2 + 1
        figure.add_trace(
            go.Heatmap(
                x=x,
                y=y,
                z=records[0]['fields'][i],
                colorscale=('Viridis', 'Cividis', 'Viridis', 'RdBu_r')[i],
                zmin=ranges[i][0],
                zmax=ranges[i][1],
                zsmooth=False,
                colorbar={
                    'len': 0.29,
                    'thickness': 12,
                    'x': 0.43 if col == 1 else 1.01,
                    'y': 0.84 if row == 1 else 0.42,
                },
            ),
            row=row,
            col=col,
        )
        figure.add_shape(
            type='circle',
            x0=1.5,
            x1=2.5,
            y0=1.5,
            y1=2.5,
            fillcolor='#e2e8f0',
            line={'color': '#334155', 'width': 1.5},
            row=row,
            col=col,
        )
        figure.update_xaxes(title_text='x / D', range=[0, 8], row=row, col=col)
        figure.update_yaxes(
            title_text='y / D',
            range=[0, 4.1],
            scaleanchor='x' if i == 0 else f'x{i + 1}',
            constrain='domain',
            row=row,
            col=col,
        )


def _history_figure(
    x: Array,
    y: Array,
    records: list[HistoryFrame],
    ranges: list[tuple[float, float]],
    histories: list[tuple[str, LossHistory]],
    offset: int,
) -> go.Figure:
    figure = make_subplots(
        rows=3,
        cols=2,
        specs=[[{}, {}], [{}, {}], [{'colspan': 2}, None]],
        row_heights=[0.4, 0.4, 0.2],
        vertical_spacing=0.13,
        horizontal_spacing=0.16,
        subplot_titles=(
            'Speed |u|',
            'Pressure p',
            'Horizontal velocity u',
            'Vertical velocity v',
            'Training objective',
        ),
    )
    _history_fields(figure, x, y, records, ranges)
    for label, history in histories:
        steps, losses = np.asarray(history).T
        figure.add_scatter(
            x=steps,
            y=losses,
            name=label,
            mode='lines',
            line={'width': 1.6},
            row=3,
            col=1,
        )
    figure.add_scatter(
        x=[0],
        y=[records[0]['loss']],
        mode='markers',
        marker={'size': 9, 'color': '#be123c'},
        showlegend=False,
        row=3,
        col=1,
    )
    figure.update_xaxes(
        title_text='Total optimizer updates', range=[0, offset], row=3, col=1
    )
    loss_range = _loss_range(histories)
    figure.update_yaxes(title_text='Loss', type='log', range=loss_range, row=3, col=1)
    scientific_style(figure)
    figure.update_layout(
        width=1400,
        height=1150,
        margin={'t': 125, 'b': 140, 'l': 80, 'r': 110},
        title={'y': 0.965, 'yanchor': 'top'},
        legend={'orientation': 'h', 'y': -0.10, 'yanchor': 'top'},
    )
    figure.add_annotation(
        x=0.5,
        y=0,
        xref='paper',
        yref='paper',
        yshift=-100,
        showarrow=False,
        text='Recorded checkpoints; uneven update spacing. Fixed colour scales. Historical training points unavailable.'
        '<br>Optimizer progress for a steady flow, not physical time. Full-field accuracy unverified.',
        font={'size': 10, 'color': '#606b73'},
    )
    return figure


def _render_history(
    directory: str | Path,
    output_dir: str | Path | None,
    figure: go.Figure,
    records: list[HistoryFrame],
    offset: int,
) -> Path:
    output = Path(output_dir or Path(directory) / 'plots')
    output.mkdir(parents=True, exist_ok=True)
    frame_root = Path(directory) / 'plots'
    frame_root.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix='training-animation-', dir=frame_root))
    cursor_index = len(figure.data) - 1
    has_points = any(r['collocation'] is not None for r in records)
    point_indices = add_overlay(figure, {} if has_points else None, toggle=False)
    snapshots, paths = [], []
    for index, record in enumerate(records):
        for i, values in enumerate(record['fields']):
            figure.data[i].z = values
        figure.data[cursor_index].x, figure.data[cursor_index].y = (
            [record['step']],
            [record['loss']],
        )
        for point_index, trace in zip(
            point_indices, point_traces(record['collocation'])
        ):
            figure.data[point_index].update(trace.to_plotly_json())
        sampling = (
            'Training points: navy PDE · orange boundaries · red anchors · purple flux quadrature.'
            if record['collocation'] is not None
            else 'Historical training points unavailable.'
        )
        cast(Any, figure.layout).annotations[-1].text = (
            'Recorded checkpoints; uneven update spacing. Fixed colour and loss scales. '
            + sampling
            + '<br>Optimizer progress for a steady flow, not physical time. Full-field accuracy unverified.'
        )
        figure.update_layout(
            title_text=(
                'Cylinder · Re = 20 · Training convergence'
                f'<br><sup>Update {record["step"]:,} / {offset:,} · {record["phase"]}'
                f' · {record["count"]:,} interior points · Loss {record["loss"]:.2e}</sup>'
            )
        )
        snapshots.append(figure.to_plotly_json())
        paths.append(folder / f'epoch_{index}.png')
    pio.write_images(snapshots, paths, format='png', width=1400, height=1150)
    create_gif(output / 'training.gif', folder, duration_ms=250, final_hold_ms=3000)
    return output


def export_cylinder_training_gif(
    directory: str | Path, *, output_dir: str | Path | None = None
) -> Path:
    """Follow the saved refinement chain and animate its measured field snapshots."""
    require_gif_export()
    stages = _load_stages(directory)
    x, y, records, histories, offset = _history_records(stages)
    ranges = _field_ranges(records)
    figure = _history_figure(x, y, records, ranges, histories, offset)
    output = _render_history(directory, output_dir, figure, records, offset)
    atomic_json(
        output / 'training-animation.json',
        {
            'schema_version': 1,
            'description': 'Actual saved prediction snapshots; no temporal interpolation or retraining.',
            'run_id': stages[-1]['metadata']['run_id'],
            'checkpoint_sha256': stages[-1]['metadata']['artifacts']['model.pt'][
                'sha256'
            ],
            'sources': [
                {
                    'run_id': s['metadata']['run_id'],
                    'html_sha256': s['metadata']['artifacts']['training.html'][
                        'sha256'
                    ],
                }
                for s in stages
            ],
            'frames': [
                {
                    **{
                        k: v for k, v in r.items() if k not in ('fields', 'collocation')
                    },
                    'collocation_recorded': r['collocation'] is not None,
                }
                for r in records
            ],
            'frame_duration_ms': 250,
            'final_hold_ms': 3000,
            'display_bounds': {'x': [0, 8], 'y': [0, 4.1]},
            'fixed_field_ranges': ranges,
            'fixed_log_loss_range': _loss_range(histories),
            'collocation': 'Recorded point sets shown when available; absent historical samples are not reconstructed.',
            'gif_sha256': hashlib.sha256(
                (output / 'training.gif').read_bytes()
            ).hexdigest(),
        },
    )
    return output / 'training.gif'
