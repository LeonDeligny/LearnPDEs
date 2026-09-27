"""Saved training exports retain real fields, points, scales, and provenance."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from learnpdes.config import RunConfig
from learnpdes.training import train
from learnpdes.visualization.cylinder_history import (
    _archive,
    _array,
    export_cylinder_training_gif,
)


class TestCylinderHistory(unittest.TestCase):
    def test_export_uses_saved_fields_and_points_and_rejects_changed_archive(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as folder:
            trainer = train(
                RunConfig(
                    'cylinder',
                    points=3,
                    epochs=1,
                    resolution=5,
                    max_frames=2,
                    save_gif=False,
                    output_dir=Path(folder),
                )
            )
            assert trainer.run is not None
            directory = trainer.run.directory
            source = _archive(directory)
            assert source is not None
            with (
                patch('learnpdes.visualization.cylinder_history.require_gif_export'),
                patch(
                    'learnpdes.visualization.cylinder_history.pio.write_images'
                ) as render,
                patch(
                    'learnpdes.visualization.cylinder_history.create_gif',
                    side_effect=lambda path, *args, **kwargs: path.write_bytes(
                        b'test-only'
                    ),
                ),
            ):
                output = export_cylinder_training_gif(directory)
            frames = render.call_args.args[0]
            self.assertEqual(len(frames), 2)
            selected = (source['x'] >= 0) & (source['x'] <= 8)
            for frame, values, step in zip(
                frames, source['predictions'], source['steps']
            ):
                # Pressure, horizontal velocity, and vertical velocity retain the archive's values.
                for displayed, source_index in ((1, 2), (2, 0), (3, 1)):
                    np.testing.assert_array_equal(
                        _array(frame['data'][displayed]['z']),
                        values[source_index][:, selected],
                    )
                markers = next(
                    t for t in frame['data'] if t.get('meta', {}).get('kind') == 'pde'
                )
                points = source['collocation'][step]
                assert points is not None
                xy = np.asarray(points['interior']['coordinates'])
                np.testing.assert_array_equal(_array(markers['x']), xy[:, 0])
                np.testing.assert_array_equal(_array(markers['y']), xy[:, 1])
            self.assertEqual(
                frames[0]['layout']['yaxis5']['range'],
                frames[1]['layout']['yaxis5']['range'],
            )
            for i in range(4):
                self.assertEqual(
                    frames[0]['data'][i]['zmin'], frames[1]['data'][i]['zmin']
                )
                self.assertEqual(
                    frames[0]['data'][i]['zmax'], frames[1]['data'][i]['zmax']
                )
            manifest = json.loads(
                output.with_name('training-animation.json').read_text()
            )
            self.assertEqual(manifest['frames'][-1]['step'], 1)
            self.assertTrue(all(f['collocation_recorded'] for f in manifest['frames']))
            html = directory / 'training.html'
            html.write_text(html.read_text() + '<!-- changed -->')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                _archive(directory)
