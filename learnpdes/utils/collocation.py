"""Record actual loss-evaluation coordinates independently of rendering grids."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import plotly.graph_objects as go
from numpy.typing import ArrayLike
from torch import Tensor

from learnpdes.types import Array, PointGroups, RecordedPoints
from learnpdes.utils.artifacts import atomic_json

KINDS: dict[str, tuple[str, str, str, int]] = {
    'pde': ('PDE points', '#172554', 'circle', 3),
    'boundary': ('Boundary points', '#c2410c', 'diamond', 5),
    'anchor': ('Initial / pressure anchors', '#be123c', 'star', 9),
    'integral': ('Flux quadrature', '#7e22ce', 'cross', 4),
}


def _coordinates(values: Tensor | ArrayLike) -> Array:
    if isinstance(values, Tensor):
        values = values.detach().cpu().numpy()
    coordinates = np.asarray(values)
    if coordinates.size == 0:
        return coordinates
    if coordinates.ndim == 1:
        coordinates = coordinates[:, None]
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] not in (1, 2)
        or not np.isfinite(coordinates).all()
    ):
        raise ValueError('Collocation coordinates must be finite N×1 or N×2 arrays.')
    return coordinates


def snapshot(groups: PointGroups | None) -> RecordedPoints | None:
    """Copy only coordinates; retain overlapping loss roles and exact samples."""
    if groups is None:
        return None
    result: RecordedPoints = {}
    dimension = None
    for name, group in groups.items():
        kind = group['kind']
        if kind not in KINDS:
            raise ValueError(f'Unknown collocation kind: {kind}')
        coordinates = _coordinates(group['coordinates'])
        if coordinates.size == 0:
            continue
        if dimension is not None and dimension != coordinates.shape[1]:
            raise ValueError(
                'Collocation groups must have the same coordinate dimension.'
            )
        dimension = coordinates.shape[1]
        result[name] = {'kind': kind, 'coordinates': coordinates.tolist()}
    return result


def summary(groups: RecordedPoints | None) -> str:
    if groups is None:
        return 'Training points were not recorded'
    counts = {
        kind: sum(len(g['coordinates']) for g in groups.values() if g['kind'] == kind)
        for kind in KINDS
    }
    return 'Collocation: ' + ' · '.join(
        f'{KINDS[kind][0]} {count:,}' for kind, count in counts.items() if count
    )


class CollocationRecorder:
    """Store fixed point sets once, and map every displayed checkpoint to its set."""

    def __init__(self) -> None:
        """Start an empty, deduplicated record of checkpoint coordinates."""
        self.point_sets: list[RecordedPoints | None] = []
        self.checkpoints: list[dict[str, int]] = []
        self._indices: dict[str, int] = {}

    def capture(self, step: int, groups: PointGroups | None) -> RecordedPoints | None:
        copied = snapshot(groups)
        key = hashlib.sha256(json.dumps(copied, sort_keys=True).encode()).hexdigest()
        if key not in self._indices:
            self._indices[key] = len(self.point_sets)
            self.point_sets.append(copied)
        index = self._indices[key]
        self.checkpoints.append({'step': int(step), 'point_set': index})
        return self.point_sets[index]

    def save(self, path: str | Path) -> None:
        atomic_json(
            path,
            {
                'schema_version': 1,
                'description': 'Actual coordinates used to evaluate the loss at each recorded checkpoint. Point roles can overlap; display and validation grids are excluded.',
                'point_sets': self.point_sets,
                'checkpoints': self.checkpoints,
            },
        )


def load_final_points(directory: str | Path) -> RecordedPoints | None:
    path = Path(directory) / 'collocation.json'
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    if data.get('schema_version') != 1 or not data['checkpoints']:
        raise ValueError('Unsupported or empty collocation recording.')
    return snapshot(data['point_sets'][data['checkpoints'][-1]['point_set']])


def _kind_coordinates(
    groups: RecordedPoints | None, kind: str, dimensions: int
) -> tuple[Array, list[str]]:
    coordinates, names = [], []
    for name, group in (groups or {}).items():
        if group['kind'] == kind:
            coordinates.extend(group['coordinates'])
            names.extend([name] * len(group['coordinates']))
    xy = np.asarray(coordinates).reshape(-1, dimensions)
    return xy, names


def _point_marker(kind: str, rug: bool) -> dict[str, Any]:
    _, color, symbol, size = KINDS[kind]
    return {
        'color': color,
        'symbol': 'line-ns' if rug and kind == 'pde' else symbol,
        'size': 8 if rug and kind == 'pde' else size,
        'opacity': 0.55 if kind == 'pde' else 0.85,
        'line': {'width': 1 if rug else 0, 'color': color},
    }


def point_traces(
    groups: RecordedPoints | None,
    *,
    xaxis: str = 'x',
    yaxis: str = 'y',
    rug: bool = False,
) -> list[go.Scatter]:
    """Use a separate unit-height y axis for 1D rugs: no invented field values."""
    traces = []
    for index, (kind, (label, _, _, _)) in enumerate(KINDS.items()):
        xy, names = _kind_coordinates(groups, kind, 1 if rug else 2)
        traces.append(
            go.Scatter(
                x=xy[:, 0],
                y=np.full(len(xy), 0.025 + index * 0.045) if rug else xy[:, 1],
                xaxis=xaxis,
                yaxis=yaxis,
                mode='markers',
                name=f'{label} ({len(xy):,})',
                showlegend=False,
                customdata=names,
                meta={'role': 'collocation', 'kind': kind, 'count': len(xy)},
                marker=_point_marker(kind, rug),
                hovertemplate='x %{x:.6g}'
                + ('' if rug else '<br>y %{y:.6g}')
                + '<br>%{customdata}<extra>'
                + label
                + '</extra>',
            )
        )
    return traces


def _rug_axis(fig: go.Figure, xaxis: str, yaxis: str) -> str:
    number = (
        max(int(key[5:] or '1') for key in fig.layout if key.startswith('yaxis')) + 1
    )
    fig.update_layout(
        {
            f'yaxis{number}': {
                'overlaying': yaxis,
                'anchor': xaxis,
                'range': [0, 1],
                'visible': False,
                'fixedrange': True,
            }
        }
    )
    return f'y{number}'


def add_overlay(
    fig: go.Figure,
    groups: RecordedPoints | None,
    *,
    xaxis: str = 'x',
    yaxis: str = 'y',
    rug: bool = False,
    toggle: bool = True,
) -> list[int]:
    """Add visible points without changing field scales, plus independent controls."""
    if groups is None:
        return []
    if rug:
        yaxis = _rug_axis(fig, xaxis, yaxis)
    indices = []
    for trace in point_traces(groups, xaxis=xaxis, yaxis=yaxis, rug=rug):
        indices.append(len(fig.data))
        fig.add_trace(trace)
    if toggle:
        cast(Any, fig.layout).updatemenus += (
            dict(
                type='buttons',
                direction='left',
                x=0,
                y=1,
                xanchor='left',
                yanchor='bottom',
                pad={'b': 40},
                active=0,
                buttons=[
                    dict(
                        label=label,
                        method='update',
                        args=[{'visible': [show] * len(indices)}, {}, indices],
                    )
                    for label, show in (('Show points', True), ('Hide points', False))
                ],
            ),
        )
    return indices
