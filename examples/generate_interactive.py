"""Generate HTML training runs with the shared scenario CLI.

uv run python -m examples.generate_interactive --help
"""

from learnpdes.cli import main as cli
from learnpdes.scenarios import RunConfig
from learnpdes.training import train


def generate(
    name, epochs, output_dir, seed=0, max_frames=40, resolution=51, cosinus_order=2
):
    trainer = train(
        RunConfig(
            name,
            epochs=epochs,
            output_dir=output_dir,
            seed=seed,
            max_frames=max_frames,
            resolution=resolution,
            cosinus_order=cosinus_order,
            save_gif=False,
        )
    )
    return trainer.run.manifest


def main(argv=None):
    cli(argv, default_scenario='all', save_gif=False)


if __name__ == '__main__':
    main()
