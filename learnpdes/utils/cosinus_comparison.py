"""Export a synchronized comparison of cosine models from saved checkpoints."""

import json
from pathlib import Path

import numpy as np
from plotly.offline import get_plotlyjs

from learnpdes import cosinus
from learnpdes.utils.plot_style import COLORS


def comparison_data(runs):
    """Validate recordings and align real optimizer steps without interpolation."""
    if not runs:
        raise ValueError('At least one cosine run is required.')
    coordinates = np.asarray(runs[0]['coordinates'], dtype=float)
    if (
        coordinates.ndim != 1
        or len(coordinates) < 3
        or not np.isfinite(coordinates).all()
        or not (np.diff(coordinates) > 0).all()
    ):
        raise ValueError(
            'Comparison coordinates must be finite and strictly increasing.'
        )
    if not all(mask.any() for mask in cosinus.region_masks(coordinates).values()):
        raise ValueError(
            'Comparison recordings must include both inside and outside samples.'
        )
    records, identities, shared_steps = [], set(), None
    for run in runs:
        cosinus.validate_order(run['order'])
        identity = (run['order'], run['seed'])
        if identity in identities:
            raise ValueError(
                'Each derivative order and seed must identify a unique run.'
            )
        identities.add(identity)
        if not np.array_equal(coordinates, run['coordinates']):
            raise ValueError('All runs must use the same evaluation coordinates.')
        checkpoints = {}
        previous = -1
        for frame in run['history']:
            step = frame['step']
            if not isinstance(step, int) or step <= previous:
                raise ValueError(
                    'Checkpoint steps must be nonnegative and strictly increasing.'
                )
            previous = step
            if 'prediction' not in frame:
                raise ValueError(
                    'Checkpoint predictions are missing. Regenerate the comparison '
                    'with examples.compare_cosinus to record synchronized curves.'
                )
            prediction = np.asarray(frame['prediction'], dtype=float)
            if (
                prediction.shape != coordinates.shape
                or not np.isfinite(prediction).all()
            ):
                raise ValueError(
                    'Checkpoint predictions must be finite and match the grid.'
                )
            checkpoints[step] = {
                'prediction': prediction.tolist(),
                'mse': cosinus.region_mse(coordinates, prediction),
            }
        if not checkpoints:
            raise ValueError('Every run needs at least one checkpoint.')
        steps = set(checkpoints)
        shared_steps = steps if shared_steps is None else shared_steps & steps
        records.append(
            {'order': run['order'], 'seed': run['seed'], 'checkpoints': checkpoints}
        )
    if not shared_steps:
        raise ValueError('The runs have no common recorded training step.')
    steps = sorted(shared_steps)
    for record in records:
        record['frames'] = [record['checkpoints'][step] for step in steps]
        del record['checkpoints']
    return {
        'coordinates': coordinates.tolist(),
        'reference': np.cos(coordinates).tolist(),
        'steps': steps,
        'orders': sorted({run['order'] for run in runs}),
        'seeds': sorted({run['seed'] for run in runs}),
        'runs': records,
        'colors': COLORS,
        'regions': {
            key: cosinus.REGION_LABELS[key]
            for key in cosinus.REGION_LABELS
            if key in cosinus.region_masks(coordinates)
        },
        'region_titles': cosinus.REGION_TITLES,
        'evaluation_pi': 3 if 'far_outside' in cosinus.region_masks(coordinates) else 2,
    }


def write_comparison(runs, path):
    """Write a portable offline viewer; the neural networks are not retrained."""
    data = comparison_data(runs)
    template = (
        Path(__file__).with_name('cosinus_comparison.html').read_text(encoding='utf-8')
    )
    payload = json.dumps(data, allow_nan=False, separators=(',', ':')).replace(
        '<', '\\u003c'
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(template.replace('__COMPARISON_DATA__', payload), encoding='utf-8')
    (path.parent / 'plotly.min.js').write_text(get_plotlyjs(), encoding='utf-8')
    return path
