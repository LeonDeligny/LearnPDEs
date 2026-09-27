"""Public CLI contracts, validation, and reproducible parameter overrides."""

import contextlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from typer.main import get_command
from typer.testing import CliRunner

from learnpdes.cli import app, main
from learnpdes.scenarios import RunConfig, SCENARIOS


class TestCLI(unittest.TestCase):
    def test_help_for_all_entry_points(self):
        for module in (
            'learnpdes',
            'learnpdes.main',
            'examples.train_pinn',
            'examples.generate_animations',
            'examples.generate_interactive',
        ):
            with self.subTest(module=module):
                result = subprocess.run(
                    [sys.executable, '-m', module, 'train', '--help'],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=30,
                    env={**os.environ, 'NO_COLOR': '1', 'COLUMNS': '120'},
                )
                self.assertIn('Usage:', result.stdout)
                self.assertIn('--learning-rate', result.stdout)
                self.assertIn('--no-gif', result.stdout)
                for panel in ('Training', 'Network', 'Reproducibility', 'Output'):
                    self.assertIn(panel, result.stdout)
                self.assertNotIn('Started training', result.stdout)

    def test_command_discovery_help_and_no_arguments(self):
        runner = CliRunner()
        for arguments, code in ((['--help'], 0), (['-h'], 0), ([], 2)):
            with self.subTest(arguments=arguments):
                result = runner.invoke(app, arguments)
                self.assertEqual(result.exit_code, code, result.output)
                for command in ('train', 'scenarios', 'examples', '--show-completion'):
                    self.assertIn(command, result.output)

    def test_scenario_catalog_in_terminal_and_json(self):
        runner = CliRunner()
        result = runner.invoke(app, ['scenarios', '--json'])
        self.assertEqual(result.exit_code, 0, result.output)
        catalog = json.loads(result.stdout)
        self.assertEqual([case['name'] for case in catalog], list(SCENARIOS))
        for width in (60, 80, 120):
            with self.subTest(width=width):
                result = runner.invoke(
                    app, ['scenarios'], env={'COLUMNS': str(width), 'NO_COLOR': '1'}
                )
                self.assertEqual(result.exit_code, 0, result.output)
                normalized = ' '.join(result.stdout.split())
                for case in SCENARIOS.values():
                    self.assertIn(case.name, result.stdout)
                    self.assertIn(case.description, normalized)
                self.assertNotIn('\x1b[', result.stdout)

    def test_example_commands_are_copyable_and_valid(self):
        runner = CliRunner()
        result = runner.invoke(
            app, ['examples'], env={'COLUMNS': '60', 'NO_COLOR': '1'}
        )
        self.assertEqual(result.exit_code, 0, result.output)
        commands = [
            line
            for line in result.stdout.splitlines()
            if line.startswith('uv run learnpdes ')
        ]
        for name in SCENARIOS:
            self.assertIn(f'uv run learnpdes train {name} --no-gif', commands)
        with patch('learnpdes.training.train') as train:
            for command in commands:
                with self.subTest(command=command):
                    args = shlex.split(command)[3:]
                    if '--dry-run' not in args:
                        args.append('--dry-run')
                    result = runner.invoke(app, args)
                    self.assertEqual(result.exit_code, 0, result.output)
                    self.assertTrue(json.loads(result.stdout))
            train.assert_not_called()

    def test_shell_completion_uses_scenario_registry(self):
        command = get_command(app).commands['train']
        with command.make_context(
            'learnpdes train', [], resilient_parsing=True
        ) as context:
            for name in ('scenario', 'scenario_option'):
                parameter = next(
                    param for param in command.params if param.name == name
                )
                choices = parameter.shell_complete(context, 'po')
                self.assertEqual(
                    {choice.value for choice in choices},
                    {'poiseuille', 'potential-flow'},
                )
                self.assertTrue(all(choice.help for choice in choices))

    def test_explicit_completion_needs_no_shell_detection(self):
        runner = CliRunner()
        with patch(
            'shellingham.detect_shell',
            side_effect=AssertionError('Shell detection must not run'),
        ):
            for shell in ('bash', 'zsh', 'fish', 'powershell', 'pwsh'):
                with self.subTest(shell=shell):
                    result = runner.invoke(app, ['completion', shell])
                    self.assertEqual(result.exit_code, 0, result.output)
                    self.assertIn('_LEARNPDES_COMPLETE', result.stdout)
                    self.assertNotIn('\x1b[', result.stdout)
        result = runner.invoke(app, ['completion', 'unknown'])
        self.assertEqual(result.exit_code, 2)

    def test_shorthand_and_explicit_command_resolve_identical_settings(self):
        for selection in (['laplace'], ['--scenario', 'laplace'], ['potential flow']):
            outputs = []
            for args in (selection, ['train', *selection]):
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    main([*args, '--dry-run'])
                outputs.append(json.loads(output.getvalue()))
            self.assertEqual(*outputs)

    def test_dry_run_reports_every_case_without_side_effects(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            contextlib.chdir(folder),
            contextlib.redirect_stdout(io.StringIO()) as output,
            patch('learnpdes.training.train') as train,
            patch('learnpdes.utils.plot.require_gif_export') as export,
        ):
            main(['all', '--dry-run'])
            configs = json.loads(output.getvalue())
            self.assertEqual([item['scenario'] for item in configs], list(SCENARIOS))
            for item in configs:
                case = SCENARIOS[item['scenario']]
                self.assertEqual(item['points'], case.points)
                self.assertEqual(item['epochs'], case.epochs)
                if case.mesh:
                    self.assertTrue(Path(item['mesh_path']).is_file())
            train.assert_not_called()
            export.assert_not_called()
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_invalid_parameters_fail_before_any_training(self):
        commands = [
            ['missing'],
            ['laplace', '--epochs', '0'],
            ['laplace', '--points', '2'],
            ['laplace', '--learning-rate', 'nan'],
            ['laplace', '--learning-rate', 'inf'],
            ['laplace', '--learning-rate', '-1'],
            ['laplace', '--hidden-dim', '0'],
            ['laplace', '--hidden-layers', '0'],
            ['laplace', '--seed', '-1'],
            ['laplace', '--threads', '0'],
            ['laplace', '--resolution', '1'],
            ['laplace', '--max-frames', '1'],
            ['laplace', '--resample-every', '0'],
            ['laplace', '--lbfgs-steps', '-1'],
            ['laplace', '--lbfgs-steps', '1'],
            ['laplace', '--cosinus-order', '4'],
            ['cosinus', '--cosinus-order', '3'],
            ['laplace', '--mesh', 'missing.su2'],
            ['potential-flow', '--mesh', 'missing.su2'],
            ['all', '--lbfgs-steps', '1'],
            ['laplace', '--scenario', 'cosinus'],
        ]
        with (
            patch('learnpdes.training.train') as train,
            patch('learnpdes.utils.plot.require_gif_export') as export,
        ):
            for command in commands:
                with (
                    self.subTest(command=command),
                    contextlib.redirect_stderr(io.StringIO()) as error,
                ):
                    with self.assertRaises(SystemExit) as stopped:
                        main(command)
                    self.assertEqual(stopped.exception.code, 2)
                    self.assertIn('error', error.getvalue().lower())
            train.assert_not_called()
            export.assert_not_called()

    def test_legacy_flow_identifiers_and_batch_defaults(self):
        for original, name in (
            ('potential flow', 'potential-flow'),
            ('solenoidal flow', 'solenoidal-flow'),
        ):
            self.assertEqual(RunConfig(original).resolved().scenario, name)
        from examples.generate_interactive import main as batch

        with contextlib.redirect_stdout(io.StringIO()) as output:
            batch(['--dry-run'])
        configs = json.loads(output.getvalue())
        self.assertEqual(len(configs), len(SCENARIOS))
        self.assertTrue(all(not item['save_gif'] for item in configs))

    def test_overrides_reach_training_and_are_saved(self):
        with (
            tempfile.TemporaryDirectory() as folder,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            main(
                [
                    'kovasznay',
                    '--epochs',
                    '1',
                    '--lbfgs-steps',
                    '1',
                    '--points',
                    '3',
                    '--learning-rate',
                    '0.002',
                    '--hidden-dim',
                    '8',
                    '--hidden-layers',
                    '2',
                    '--seed',
                    '17',
                    '--resample-every',
                    '2',
                    '--resolution',
                    '3',
                    '--max-frames',
                    '2',
                    '--no-gif',
                    '--output-dir',
                    folder,
                ]
            )
            path = next(Path(folder).glob('runs/kovasznay/*/run.json'))
            manifest = json.loads(path.read_text())
            self.assertEqual(manifest['status'], 'completed')
            self.assertEqual(manifest['completed_steps'], 2)
            self.assertEqual(manifest['training']['learning_rate'], 0.002)
            self.assertEqual(manifest['training']['resample_every'], 2)
            self.assertEqual(manifest['settings']['model']['hidden_dim'], 8)
            self.assertEqual(manifest['settings']['model']['num_hidden_layers'], 2)
            self.assertEqual(manifest['settings']['seed'], 17)
            self.assertEqual(manifest['settings']['config']['points'], 3)
            self.assertTrue((path.parent / 'model.pt').is_file())


if __name__ == '__main__':
    unittest.main()
