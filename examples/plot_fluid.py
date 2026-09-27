"""Export cylinder fields and independent residual maps from a completed run."""

from pathlib import Path
import argparse

import torch

from learnpdes.utils.fluid_plots import export_cylinder_plots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'run', type=Path, help='Directory containing run.json and model.pt'
    )
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--resolution', type=int, default=601)
    parser.add_argument(
        '--png',
        action='store_true',
        help='Also export PNG; requires Chrome for Kaleido',
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(
        export_cylinder_plots(
            args.run,
            output_dir=args.output_dir,
            resolution=args.resolution,
            png=args.png,
        )
    )


if __name__ == '__main__':
    main()
