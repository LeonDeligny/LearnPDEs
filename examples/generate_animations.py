"""Generate GIF and HTML training runs with the shared scenario CLI.

uv run python -m examples.generate_animations --help
"""

from learnpdes.cli import main as cli
from learnpdes.scenarios import RunConfig, SCENARIOS
from learnpdes.training import train

RUNS = {name: (case.epochs, case.points) for name, case in SCENARIOS.items()}


def generate(name, epochs, points, output_dir, seed=0, max_frames=40):
    trainer = train(
        RunConfig(
            name,
            epochs=epochs,
            points=points,
            output_dir=output_dir,
            seed=seed,
            max_frames=max_frames,
        )
    )
    return trainer.run.manifest


def main(argv=None):
    cli(argv, default_scenario='all')


if __name__ == '__main__':
    main()
