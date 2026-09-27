"""CLI help and completion generation without launching child processes."""

import unittest
from unittest.mock import patch

from typer.testing import CliRunner

from learnpdes.cli import CompletionShell, app


class TestCLI(unittest.TestCase):
    def test_training_help_does_not_start_training(self) -> None:
        with patch('learnpdes.training.train') as train:
            result = CliRunner().invoke(app, ['train', '--help'])
        self.assertEqual(result.exit_code, 0, result.output)
        for option in ('--learning-rate', '--no-gif', '--output-dir'):
            self.assertIn(option, result.stdout)
        train.assert_not_called()

    def test_completion_formats_all_supported_shells_without_processes(self) -> None:
        with (
            patch(
                'shellingham.detect_shell',
                side_effect=AssertionError('Completion must not detect a shell'),
            ),
            patch(
                'subprocess.Popen',
                side_effect=AssertionError('Completion must not start a process'),
            ),
        ):
            for completion_shell in CompletionShell:
                with self.subTest(completion_shell=completion_shell.value):
                    result = CliRunner().invoke(
                        app, ['completion', completion_shell.value]
                    )
                    self.assertEqual(result.exit_code, 0, result.output)
                    self.assertIn('_LEARNPDES_COMPLETE', result.stdout)
                    self.assertNotIn('\x1b[', result.stdout)

    def test_completion_rejects_unknown_shells_before_generating_script(self) -> None:
        with patch('typer._completion_shared.get_completion_script') as generate:
            for value in ('unknown', 'bash; echo unexpected', '$(echo unexpected)'):
                with self.subTest(value=value):
                    result = CliRunner().invoke(app, ['completion', value])
                    self.assertEqual(result.exit_code, 2, result.output)
            generate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
