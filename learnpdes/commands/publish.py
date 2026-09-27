"""Publish a selected completed training run as a stable documentation example.

uv run learnpdes publish assets/runs/kovasznay/<run-id>
Use --replace to intentionally replace a previously published example.
"""

from pathlib import Path
from typing import Annotated

import typer

from learnpdes.utils.artifacts import publish_run


def publish(
    run_dir: Annotated[Path, typer.Argument()],
    output_dir: Annotated[Path, typer.Option()] = Path('assets'),
    replace: Annotated[bool, typer.Option('--replace')] = False,
) -> None:
    """Publish a completed run as a stable documentation example.

    Use --replace to intentionally replace a previously published example.
    """
    try:
        destination = publish_run(run_dir, output_dir, replace=replace)
    except (OSError, ValueError) as error:
        raise typer.BadParameter(str(error)) from None
    print(f'Published example: {destination}')
