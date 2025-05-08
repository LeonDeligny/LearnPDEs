'''
Plot functions.
'''
#  ======= Imports =======

import os
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt

from pathlib import Path
from numpy import ndarray
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import (
    Union,
    Callable,
)

# ======= Functions =======


def create_gif(
    output_path: Path = './gifs/training_process.gif',
    input_folder: Path = './gifs/epochs',
    duration: float = 0.5,
) -> None:
    images = []
    # Sort files by epoch number
    sorted_files = sorted(
        [
            file_name
            for file_name in os.listdir(input_folder)
            if file_name.endswith('.png') and file_name.startswith('epoch_')
        ],
        key=lambda x: int(x.split('_')[1].split('.')[0])
    )

    for file_name in sorted_files:
        file_path = os.path.join(input_folder, file_name)
        images.append(imageio.imread(file_path))

    # Save the GIF using only the image data
    imageio.mimsave(output_path, images, duration=duration)


def ensure_directory_exists() -> Path:
    '''
    Ensure directory exists and
    that the directory is cleaned before each run.
    '''
    output_dir = './gifs/epochs'
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            if file.endswith('.png'):
                os.remove(os.path.join(output_dir, file))
        print('Removing old epochs folder')
    else:
        os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_plot(
    output_dir: Path,
    epoch: int,
    x: ndarray,
    f: ndarray,
    loss: float,
    analytical: Callable
) -> None:

    mask = x < 10
    x = x[mask]
    f = f[mask]

    plt.figure()
    plt.plot(x, f, label='NN Prediction')
    plt.plot(x, analytical(x), label='Analytical Solution')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.title(f'Epoch: {epoch}, loss: {loss:.4f}')
    plt.savefig(f'{output_dir}/epoch_{epoch}.png')
    plt.close()


# Helper function to create a plot
def create_plot(
    x1: ndarray,
    x2: ndarray,
    fig: Figure,
    ax: Axes,
    data: ndarray,
    title: str,
) -> None:
    im = ax.imshow(
        data,
        extent=[x1.min(), x1.max(), x2.min(), x2.max()],
        origin='lower',
        aspect='auto',
    )
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(im, ax=ax)


def save_2d_plot(
    output_dir: Path,
    epoch: int,
    x1: ndarray,
    x2: ndarray,
    f: ndarray,
    loss: float,
    analytical: Union[Callable, None],
) -> None:
    n = int(np.sqrt(len(x1)))
    x1_grid = x1.reshape(n, n).T
    x2_grid = x2.reshape(n, n).T
    f_grid = f.reshape(n, n).T

    if analytical is None:
        ncols = 1
    else:
        ncols = 3
    fig, axes = plt.subplots(1, ncols, figsize=(18, 6))
    if ncols == 1:
        axes = [axes]
    create_plot(x1_grid, x2_grid, fig, axes[0], f_grid, 'Model Output')

    if analytical is not None:
        ana_f: ndarray = analytical(x1_grid, x2_grid).T
        difference = f_grid - ana_f
        create_plot(x1_grid, x2_grid, fig, axes[1], ana_f, 'Analytical')
        create_plot(x1_grid, x2_grid, fig, axes[2], difference, 'Difference')

    plt.suptitle(f'Epoch: {epoch}, Loss: {loss:.4f}')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/epoch_{epoch}.png')
    plt.close()
