"""Atomic metadata updates and constrained Git provenance collection."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from learnpdes.utils import artifacts


class TestAtomicJSON(unittest.TestCase):
    def test_success_replaces_complete_document_and_cleans_up(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'nested' / 'run.json'
            artifacts.atomic_json(path, {'value': 1})
            artifacts.atomic_json(path, {'value': 2})
            self.assertEqual(json.loads(path.read_text()), {'value': 2})
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_serialization_failure_preserves_previous_record_and_cleans_up(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'run.json'
            artifacts.atomic_json(path, {'value': 1})
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                artifacts.atomic_json(path, {'value': float('nan')})
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_replace_failure_preserves_previous_record_and_cleans_up(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'run.json'
            artifacts.atomic_json(path, {'value': 1})
            before = path.read_bytes()
            with patch.object(Path, 'replace', side_effect=OSError('disk full')):
                with self.assertRaisesRegex(OSError, 'disk full'):
                    artifacts.atomic_json(path, {'value': 2})
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(path.parent.iterdir()), [path])


class TestSourceRevision(unittest.TestCase):
    def test_git_uses_absolute_executable_fixed_arguments_and_no_shell(self) -> None:
        for status in ('', ' M learnpdes/cli.py\n'):
            with (
                self.subTest(status=status),
                patch('learnpdes.utils.artifacts.shutil.which', return_value='bin/git'),
                patch(
                    'learnpdes.utils.artifacts.subprocess.run',
                    side_effect=[Mock(stdout='abc123\n'), Mock(stdout=status)],
                ) as run,
            ):
                self.assertEqual(
                    artifacts.source_revision(),
                    {'commit': 'abc123', 'dirty': bool(status)},
                )
                self.assertEqual(run.call_count, 2)
                for invocation, arguments in zip(
                    run.call_args_list,
                    (['rev-parse', 'HEAD'], ['status', '--porcelain']),
                    strict=True,
                ):
                    self.assertEqual(
                        invocation.args[0],
                        [
                            str(Path('bin/git').resolve()),
                            '-c',
                            'core.fsmonitor=false',
                            *arguments,
                        ],
                    )
                    self.assertIs(invocation.kwargs['shell'], False)
                    self.assertEqual(invocation.kwargs['timeout'], 5)
                    self.assertEqual(
                        invocation.kwargs['cwd'],
                        Path(artifacts.__file__).resolve().parents[2],
                    )

    def test_missing_git_returns_unknown_without_launching_a_process(self) -> None:
        with (
            patch('learnpdes.utils.artifacts.shutil.which', return_value=None),
            patch('learnpdes.utils.artifacts.subprocess.run') as run,
        ):
            self.assertEqual(
                artifacts.source_revision(), {'commit': None, 'dirty': None}
            )
            run.assert_not_called()

    def test_git_failures_return_unknown(self) -> None:
        failures = (
            OSError('Git unavailable'),
            artifacts.subprocess.CalledProcessError(128, 'git'),
            artifacts.subprocess.TimeoutExpired('git', 5),
        )
        for failure in failures:
            for failed_call in (0, 1):
                with (
                    self.subTest(failure=type(failure).__name__, call=failed_call),
                    patch(
                        'learnpdes.utils.artifacts.shutil.which',
                        return_value='bin/git',
                    ),
                    patch(
                        'learnpdes.utils.artifacts.subprocess.run',
                        side_effect=[Mock(stdout='abc123\n')] * failed_call + [failure],
                    ),
                ):
                    self.assertEqual(
                        artifacts.source_revision(), {'commit': None, 'dirty': None}
                    )


if __name__ == '__main__':
    unittest.main()
