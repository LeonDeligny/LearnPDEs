"""Export cylinder fields and independent residual maps from a completed run."""

from pathlib import Path
from typing import Annotated

import torch
import typer

from learnpdes.utils.fluid_plots import export_cylinder_plots


def plot(
    run: Annotated[
        Path, typer.Argument(help='Directory containing run.json and model.pt')
    ],
    output_dir: Annotated[Path | None, typer.Option()] = None,
    resolution: Annotated[int, typer.Option(min=3)] = 601,
    png: Annotated[
        bool,
        typer.Option('--png', help='Also export PNG; requires Chrome for Kaleido'),
    ] = False,
    training_gif: Annotated[
        bool,
        typer.Option(
            '--training-gif',
            help='Animate saved cylinder training and refinement snapshots; requires Chrome and FFmpeg',
        ),
    ] = False,
) -> None:
    """Export cylinder fields and residual maps from a completed run."""
    torch.set_num_threads(1)
    print(
        export_cylinder_plots(
            run,
            output_dir=output_dir,
            resolution=resolution,
            png=png,
        )
    )
    if training_gif:
        from learnpdes.visualization.cylinder_history import (
            export_cylinder_training_gif,
        )

        print(export_cylinder_training_gif(run, output_dir=output_dir))
