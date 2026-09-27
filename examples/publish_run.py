"""Publish a selected completed training run as a stable documentation example.

uv run python -m examples.publish_run assets/runs/kovasznay/<run-id>
Use --replace to intentionally replace a previously published example.
"""

import argparse
from pathlib import Path

from learnpdes.utils.artifacts import publish_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('assets'))
    parser.add_argument('--replace', action='store_true')
    args = parser.parse_args()
    try:
        destination = publish_run(args.run_dir, args.output_dir, replace=args.replace)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f'Published example: {destination}')


if __name__ == '__main__':
    main()
