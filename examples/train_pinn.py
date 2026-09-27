"""Compatibility entry point: uv run python -m examples.train_pinn --help."""

from learnpdes.cli import main
from learnpdes.training import build_problem, evaluate

__all__ = ['main', 'build_problem', 'evaluate']

if __name__ == '__main__':
    main()
