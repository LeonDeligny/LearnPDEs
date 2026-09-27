"""Typer commands, Rich help, and compatibility entry points for training."""

from dataclasses import asdict
from enum import Enum
import json
from pathlib import Path
import sys
from typing import Annotated

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text
import typer

from learnpdes.scenarios import RunConfig, SCENARIOS, get_scenario

CONTEXT_SETTINGS = {'help_option_names': ['-h', '--help']}


class CompletionShell(str, Enum):
    bash = 'bash'
    zsh = 'zsh'
    fish = 'fish'
    powershell = 'powershell'
    pwsh = 'pwsh'


TRAINING_HELP = (
    'Train a scenario or run all cases sequentially.\n\n'
    'Each run saves HTML, weights, losses, and resolved settings to '
    '<output-dir>/runs/<scenario>/<run-id>/. GIF export requires Chrome and FFmpeg; '
    'use --no-gif for HTML only.\n\n'
    'Use learnpdes scenarios for per-case defaults and learnpdes examples for '
    'copyable commands. Short runs check execution, not convergence.'
)


def complete_scenario(incomplete: str):
    """Offer registry names with descriptions in shell completion."""
    choices = [('all', 'Run every scenario sequentially')]
    choices.extend((case.name, case.description) for case in SCENARIOS.values())
    return [
        (name, help_text) for name, help_text in choices if name.startswith(incomplete)
    ]


def show_scenarios(*, as_json: bool = False) -> None:
    if as_json:
        typer.echo(json.dumps([asdict(case) for case in SCENARIOS.values()], indent=2))
        return
    table = Table(title='LearnPDEs scenarios', box=box.SIMPLE, header_style='bold cyan')
    table.add_column('Scenario / problem', overflow='fold', ratio=1)
    table.add_column('Adam\nupdates', justify='right')
    table.add_column('Points', justify='right')
    table.add_column('Width', justify='right')
    for case in SCENARIOS.values():
        table.add_row(
            Text.assemble((case.name, 'cyan'), '\n', (case.description, 'dim')),
            str(case.epochs),
            str(case.points),
            str(case.hidden_dim),
        )
    console = Console()
    console.print(table)
    console.print(
        'Points: ODE samples; rectangular points per axis; cylinder/Kovasznay sqrt(interior samples). Airfoil cases retain all mesh vertices and add a grid.',
        markup=False,
    )
    console.print(
        '\nRun: learnpdes train SCENARIO --no-gif\nParameters: learnpdes train --help\nCopyable commands: learnpdes examples',
        markup=False,
    )


def list_scenarios_callback(value: bool) -> None:
    if value:
        show_scenarios()
        raise typer.Exit()


def show_examples() -> None:
    """Display complete, copyable commands without running any of them."""
    console = Console()
    console.print('Run a scenario with its default training budget', style='bold')
    for case in SCENARIOS.values():
        console.print(f'# {case.description}', style='dim', markup=False)
        # Keep each command on one logical line, including when output is piped.
        console.print(
            Text(f'uv run learnpdes train {case.name} --no-gif', style='cyan'),
            soft_wrap=True,
        )
    examples = (
        (
            'Check every case (one update; execution only)',
            'train all --epochs 1 --points 3 --resolution 5 --max-frames 2 --no-gif',
        ),
        ('Inspect all resolved defaults as JSON', 'train all --dry-run'),
        (
            'Change the learning rate and network',
            'train laplace --epochs 2000 --points 31 --learning-rate 0.0005 --hidden-dim 32 --hidden-layers 3 --seed 7 --no-gif',
        ),
        ('Use higher cosine derivatives', 'train cosinus --cosinus-order 6 --no-gif'),
        (
            'Refine a fluid solution with L-BFGS',
            'train kovasznay --epochs 1500 --lbfgs-steps 1500 --no-gif',
        ),
        (
            'Choose an output directory',
            'train poiseuille --output-dir ./results --no-gif',
        ),
    )
    for label, command in examples:
        console.print(f'\n{label}', style='bold', markup=False)
        console.print(Text(f'uv run learnpdes {command}', style='cyan'), soft_wrap=True)


def create_app(
    *, default_scenario: str | None = None, save_gif: bool = True
) -> typer.Typer:
    """Build the same typed CLI with the batch wrappers' documented defaults."""
    application = typer.Typer(
        name='learnpdes',
        help='Train physics-informed neural networks with reproducible scenario settings.',
        epilog='Start with learnpdes scenarios, learnpdes examples, or learnpdes train --help. The shorthand learnpdes SCENARIO remains supported.',
        context_settings=CONTEXT_SETTINGS,
        no_args_is_help=True,
        rich_markup_mode='rich',
        pretty_exceptions_show_locals=False,
    )

    @application.callback()
    def root(
        list_scenarios: Annotated[
            bool,
            typer.Option(
                '--list-scenarios',
                callback=list_scenarios_callback,
                is_eager=True,
                help='List cases and defaults (alias for scenarios).',
            ),
        ] = False,
    ) -> None:
        pass

    @application.command('train', help=TRAINING_HELP, rich_help_panel='Training')
    def train_command(
        scenario: Annotated[
            str | None,
            typer.Argument(
                help='Scenario name or all. See learnpdes scenarios.',
                autocompletion=complete_scenario,
                show_default=False,
            ),
        ] = None,
        scenario_option: Annotated[
            str | None,
            typer.Option(
                '--scenario',
                help='Alternative to the positional scenario.',
                autocompletion=complete_scenario,
                rich_help_panel='Scenario',
                show_default=False,
            ),
        ] = None,
        epochs: Annotated[
            int | None,
            typer.Option(
                min=1,
                help='Adam updates.',
                show_default='scenario default; see scenarios',
                rich_help_panel='Training',
            ),
        ] = None,
        points: Annotated[
            int | None,
            typer.Option(
                min=3,
                help='ODE samples or grid points per axis. Cylinder/Kovasznay: points squared interior, 4*points per boundary. Airfoil: added grid size.',
                show_default='scenario default; see scenarios',
                rich_help_panel='Training',
            ),
        ] = None,
        learning_rate: Annotated[
            float,
            typer.Option(
                help='Finite, positive Adam learning rate.', rich_help_panel='Training'
            ),
        ] = 0.001,
        hidden_dim: Annotated[
            int | None,
            typer.Option(
                min=1,
                help='Units per Tanh hidden layer.',
                show_default='64 for cylinder/Kovasznay; 20 otherwise',
                rich_help_panel='Network',
            ),
        ] = None,
        hidden_layers: Annotated[
            int,
            typer.Option(
                min=1, help='Number of Tanh hidden layers.', rich_help_panel='Network'
            ),
        ] = 4,
        seed: Annotated[
            int,
            typer.Option(
                min=0,
                max=2**32 - 1,
                help='Random seed, reset for each case.',
                rich_help_panel='Reproducibility',
            ),
        ] = 0,
        threads: Annotated[
            int,
            typer.Option(
                min=1, help='Torch CPU threads.', rich_help_panel='Reproducibility'
            ),
        ] = 1,
        lbfgs_steps: Annotated[
            int,
            typer.Option(
                min=0,
                help='Fixed-sample L-BFGS updates after Adam; cylinder/Kovasznay only.',
                rich_help_panel='Scenario-specific options',
            ),
        ] = 0,
        resample_every: Annotated[
            int,
            typer.Option(
                min=1,
                help='Refresh cylinder/Kovasznay samples every N Adam updates.',
                rich_help_panel='Scenario-specific options',
            ),
        ] = 100,
        cosinus_order: Annotated[
            int,
            typer.Option(
                min=2,
                help='Maximum even derivative order (2, 4, 6, ...); cosinus only.',
                rich_help_panel='Scenario-specific options',
            ),
        ] = 2,
        mesh_path: Annotated[
            Path | None,
            typer.Option(
                '--mesh',
                exists=True,
                dir_okay=False,
                readable=True,
                help='Airfoil SU2 mesh with inlet, outlet, wall, and airfoil markers; potential-flow/solenoidal-flow only.',
                show_default='bundled airfoil',
                rich_help_panel='Scenario-specific options',
            ),
        ] = None,
        output_dir: Annotated[
            Path,
            typer.Option(
                file_okay=False,
                help='Root for isolated training runs.',
                rich_help_panel='Output',
            ),
        ] = Path('assets'),
        max_frames: Annotated[
            int,
            typer.Option(
                min=2,
                help='Maximum checkpoints, including initial and final states.',
                rich_help_panel='Output',
            ),
        ] = 40,
        resolution: Annotated[
            int,
            typer.Option(
                min=2,
                help='Independent display samples per axis. Airfoil plots use mesh connectivity instead.',
                rich_help_panel='Output',
            ),
        ] = 51,
        gif: Annotated[
            bool,
            typer.Option(
                '--gif/--no-gif',
                help='Export a GIF in addition to HTML. Requires Chrome and FFmpeg.',
                rich_help_panel='Output',
            ),
        ] = save_gif,
        dry_run: Annotated[
            bool,
            typer.Option(
                '--dry-run',
                help='Print resolved configurations as JSON; do not train or write files.',
                rich_help_panel='Inspection',
            ),
        ] = False,
        list_scenarios: Annotated[
            bool,
            typer.Option(
                '--list-scenarios',
                callback=list_scenarios_callback,
                is_eager=True,
                hidden=True,
            ),
        ] = False,
    ) -> None:
        if scenario is not None and scenario_option is not None:
            raise typer.BadParameter(
                'Specify the scenario either positionally or with --scenario, not both.'
            )
        selected = scenario or scenario_option or default_scenario
        if selected is None:
            raise typer.BadParameter(
                'Choose a scenario or all; use learnpdes scenarios or --help.',
                param_hint='SCENARIO',
            )
        try:
            names = (
                list(SCENARIOS) if selected == 'all' else [get_scenario(selected).name]
            )
            configs = [
                RunConfig(
                    name,
                    epochs=epochs,
                    points=points,
                    learning_rate=learning_rate,
                    hidden_dim=hidden_dim,
                    hidden_layers=hidden_layers,
                    seed=seed,
                    threads=threads,
                    lbfgs_steps=lbfgs_steps,
                    resample_every=resample_every,
                    cosinus_order=cosinus_order,
                    mesh_path=mesh_path,
                    output_dir=output_dir,
                    max_frames=max_frames,
                    resolution=resolution,
                    save_gif=gif,
                ).resolved()
                for name in names
            ]
        except ValueError as error:
            raise typer.BadParameter(str(error)) from None
        if dry_run:
            # JSON stdout must stay free of colors, tables, and progress text.
            typer.echo(json.dumps([config.as_dict() for config in configs], indent=2))
            return

        from learnpdes.training import train
        from learnpdes.utils.plot import require_gif_export

        if gif:
            try:
                require_gif_export()
            except RuntimeError as error:
                raise typer.BadParameter(
                    f'{error} Use --no-gif for HTML without static export.'
                ) from None
        console = Console()
        for index, config in enumerate(configs, 1):
            console.print(
                Text(
                    f'[{index}/{len(configs)}] Training {config.scenario}',
                    style='bold cyan',
                )
            )
            train(config)

    @application.command('scenarios', rich_help_panel='Discovery')
    def scenarios_command(
        as_json: Annotated[
            bool,
            typer.Option(
                '--json', help='Emit the scenario catalog as JSON for scripts.'
            ),
        ] = False,
    ) -> None:
        """List every case with its description and training defaults."""
        show_scenarios(as_json=as_json)

    @application.command('examples', rich_help_panel='Discovery')
    def examples_command() -> None:
        """Show copyable commands for every case and common parameter overrides."""
        show_examples()

    @application.command('completion', rich_help_panel='Setup')
    def completion_command(
        shell: Annotated[
            CompletionShell,
            typer.Argument(
                help='Shell to generate completion for; no shell detection or installation.'
            ),
        ],
    ) -> None:
        """Print a completion script for an explicitly selected shell."""
        from typer.completion import get_completion_script

        typer.echo(
            get_completion_script(
                prog_name='learnpdes',
                complete_var='_LEARNPDES_COMPLETE',
                shell=shell.value,
            )
        )

    return application


app = create_app()


def main(argv=None, *, default_scenario=None, save_gif=True) -> None:
    """Run Typer while preserving the original scenario-first command syntax."""
    application = (
        app
        if default_scenario is None and save_gif
        else create_app(default_scenario=default_scenario, save_gif=save_gif)
    )
    args = list(sys.argv[1:] if argv is None else argv)
    commands = {command.name for command in application.registered_commands}
    root_options = {
        '--help',
        '-h',
        '--list-scenarios',
        '--install-completion',
        '--show-completion',
    }
    if args and args[0] not in commands and args[0].split('=')[0] not in root_options:
        args.insert(0, 'train')
    elif default_scenario is not None and (not args or args[0] in {'--help', '-h'}):
        args.insert(0, 'train')
    try:
        application(args=args, prog_name='learnpdes')
    except SystemExit as error:
        # Preserve the callable entry point's normal return on success, while
        # Typer formats usage errors and retains their nonzero exit status.
        if error.code:
            raise
