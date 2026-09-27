"""Plotly plotting entry points and optional GIF encoding."""

import shutil
import subprocess
import tempfile
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from numpy import ndarray

from learnpdes.utils.plot_style import scientific_style
from learnpdes.utils.utility import compute_normals, detach_to_numpy


def get_plot_func(scenario: str, *, color_limits=None, cosinus_order=2):
    """Create a Plotly checkpoint collector for any supported training scenario."""
    from learnpdes.utils.interactive import InteractivePlot

    return InteractivePlot(
        scenario, color_limits=color_limits, cosinus_order=cosinus_order
    )


def require_gif_export() -> None:
    """Check optional exporters before starting an animation training run."""
    if shutil.which('ffmpeg') is None:
        raise RuntimeError('Install FFmpeg before generating animations.')
    if find_spec('kaleido') is None:
        raise RuntimeError(
            'Run uv sync --group export and uv run --group export plotly_get_chrome '
            'before generating animations.'
        )


def create_gif(
    output_path: Path,
    input_folder: Path,
    duration_ms: int = 100,
    final_hold_ms: int = 2000,
) -> None:
    """Write looping GIFs with explicit millisecond delays, including a final hold."""
    for delay in (duration_ms, final_hold_ms):
        if not isinstance(delay, int) or not 10 <= delay <= 655350 or delay % 10:
            raise ValueError(
                'GIF delays must be multiples of 10 ms between 10 and 655350.'
            )
    files = sorted(
        Path(input_folder).glob('epoch_*.png'),
        key=lambda path: int(path.stem.split('_')[1]),
    )
    if not files:
        raise ValueError(f'No epoch frames found in {input_folder}.')
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        raise RuntimeError(
            'Install FFmpeg to encode animations. Rendered PNG frames have been retained.'
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # A numbered symlink sequence handles irregular checkpoint numbers without
    # loading all RGB frames into Python memory or quoting a concat manifest.
    with tempfile.TemporaryDirectory(
        prefix='.gif-', dir=output_path.parent
    ) as temporary:
        folder = Path(temporary)
        for index, frame in enumerate(files):
            (folder / f'frame_{index:06d}.png').symlink_to(frame.resolve())
        sequence = str(folder / 'frame_%06d.png')
        palette = str(folder / 'palette.png')
        encoded = folder / 'animation.gif'
        common = [
            ffmpeg,
            '-hide_banner',
            '-loglevel',
            'error',
            '-y',
            '-filter_threads',
            '1',
        ]
        source = ['-framerate', f'1000/{duration_ms}', '-i', sequence]
        commands = [
            common
            + source
            + [
                '-vf',
                'palettegen=stats_mode=full',
                '-frames:v',
                '1',
                '-update',
                '1',
                palette,
            ],
            common
            + source
            + [
                '-i',
                palette,
                '-lavfi',
                'paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle',
                '-loop',
                '0',
                '-final_delay',
                str(final_hold_ms // 10),
                '-fps_mode',
                'passthrough',
                str(encoded),
            ],
        ]
        for command in commands:
            try:
                subprocess.run(command, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as error:
                raise RuntimeError(
                    f'FFmpeg GIF encoding failed; PNG frames retained: {error.stderr.strip()}'
                ) from error
        encoded.replace(output_path)


def error_metrics(prediction: ndarray, reference: ndarray) -> dict[str, float]:
    """Discrete errors on the supplied evaluation samples, not the training loss."""
    prediction, reference = np.asarray(prediction), np.asarray(reference)
    if prediction.shape != reference.shape:
        raise ValueError('Prediction and reference shapes must match.')
    difference = prediction - reference
    denominator = np.linalg.norm(reference.ravel())
    numerator = np.linalg.norm(difference.ravel())
    relative = (
        numerator / denominator if denominator else (0.0 if numerator == 0 else np.inf)
    )
    return {
        'relative_l2': float(relative),
        'max_abs': float(np.max(np.abs(difference))),
    }


def value_range(*fields, symmetric=False):
    """Return finite, nondegenerate colour limits without a rendering dependency."""
    lo = min(float(np.min(field)) for field in fields)
    hi = max(float(np.max(field)) for field in fields)
    if not np.isfinite([lo, hi]).all():
        raise ValueError('Colour limits require finite field values.')
    if symmetric:
        bound = max(abs(lo), abs(hi)) or 1.0
        return -bound, bound
    if lo == hi:
        pad = abs(lo) * 0.01 or 1.0
        lo, hi = lo - pad, hi + pad
    return lo, hi


def plot_xy(xy, *, show=True) -> go.Figure:
    """Inspect mesh nodes interactively; return the figure for notebooks/export."""
    xy = detach_to_numpy(xy) if hasattr(xy, 'detach') else np.asarray(xy)
    fig = go.Figure(
        go.Scattergl(
            x=xy[:, 0],
            y=xy[:, 1],
            mode='markers',
            marker={'size': 3},
            name='Mesh nodes',
        )
    )
    fig.update_layout(
        title='Mesh node coordinates',
        xaxis_title='x',
        yaxis_title='y',
        yaxis_scaleanchor='x',
        height=650,
        margin={'l': 65, 'r': 30, 't': 100, 'b': 80},
        legend={'orientation': 'h', 'x': 0, 'y': -0.14},
    )
    scientific_style(fig)
    if show:
        fig.show()
    return fig


def plot_mesh(xy, mesh_masks, *, show=True) -> go.Figure:
    """Inspect boundary groups and normal directions with Plotly."""
    n_x, n_y = compute_normals(xy, mesh_masks['airfoil'])
    coordinates = detach_to_numpy(xy)
    masks = {name: detach_to_numpy(mask) for name, mask in mesh_masks.items()}
    fig = go.Figure()
    for name, mask in masks.items():
        fig.add_trace(
            go.Scattergl(
                x=coordinates[mask, 0],
                y=coordinates[mask, 1],
                mode='markers',
                marker={'size': 4},
                name=name,
            )
        )
    boundary = coordinates[masks['airfoil']]
    normals = np.column_stack((detach_to_numpy(n_x), detach_to_numpy(n_y)))[
        masks['airfoil']
    ]
    endpoints = boundary + 0.02 * np.ptp(coordinates, axis=0).max() * normals
    segments = np.full((len(boundary), 3, 2), np.nan)
    segments[:, 0], segments[:, 1] = boundary, endpoints
    segments = segments.reshape(-1, 2)
    fig.add_trace(
        go.Scatter(
            x=segments[:, 0],
            y=segments[:, 1],
            mode='lines',
            name='Boundary normals',
            line={'width': 1},
        )
    )
    fig.update_layout(
        title='Mesh boundaries and normals',
        xaxis_title='x',
        yaxis_title='y',
        yaxis_scaleanchor='x',
        height=650,
        margin={'l': 65, 'r': 30, 't': 100, 'b': 80},
        legend={'orientation': 'h', 'x': 0, 'y': -0.14},
    )
    scientific_style(fig)
    if show:
        fig.show()
    return fig
