"""Python training API and compatibility module entry point."""

from pathlib import Path
import warnings

from learnpdes.scenarios import RunConfig


def main(
    scenario: str,
    epochs: int | None = None,
    pre_epochs: int = 0,
    visualization_resolution: int = 51,
    output_dir: str | Path = 'assets',
    cosinus_order: int = 2,
    save_gif: bool = True,
    max_frames: int = 40,
    num_inputs: int | None = None,
    seed: int = 0,
    lbfgs_steps: int = 0,
    resample_every: int = 100,
    *,
    learning_rate: float = 0.001,
    hidden_dim: int | None = None,
    hidden_layers: int = 4,
    threads: int = 1,
    mesh_path: str | Path | None = None,
) -> Path:
    """Train one registered case and return its interactive HTML path.

    Unspecified epochs and num_inputs use the same scenario defaults as the CLI.
    Set save_gif=False to run without Chrome or FFmpeg. pre_epochs is a deprecated
    compatibility argument: pretraining was never active in this entry point.
    """
    if pre_epochs:
        warnings.warn(
            'pre_epochs has no effect and is deprecated.',
            DeprecationWarning,
            stacklevel=2,
        )
    from learnpdes.training import train

    trainer = train(
        RunConfig(
            scenario,
            epochs=epochs,
            points=num_inputs,
            resolution=visualization_resolution,
            output_dir=Path(output_dir),
            cosinus_order=cosinus_order,
            save_gif=save_gif,
            max_frames=max_frames,
            seed=seed,
            lbfgs_steps=lbfgs_steps,
            resample_every=resample_every,
            learning_rate=learning_rate,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            threads=threads,
            mesh_path=Path(mesh_path) if mesh_path is not None else None,
        )
    )
    return trainer.html_path


if __name__ == '__main__':
    from learnpdes.cli import main as cli

    cli()
