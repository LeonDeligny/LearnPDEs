'''
Plot functions.
'''
#  ======= Imports =======

import os

import imageio.v2 as imageio
import matplotlib.pyplot as plt

from pathlib import Path
from numpy import ndarray
from typing import Callable
from matplotlib.axes import Axes
from matplotlib.figure import Figure

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
    else:
        os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_plot(
    epoch: int,
    x: ndarray,
    f: ndarray,
    loss: float,
    analytical: Callable
) -> None:
    output_dir = ensure_directory_exists()

    mask = x < 10
    x = x[mask]
    f = f[mask]

    plt.figure(figsize=(6, 4))
    plt.plot(x, f, label='NN Prediction')
    plt.plot(x, analytical(x), label='Analytical Solution')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.title(f'Epoch: {epoch}, loss: {loss:.4f}')
    plt.xlim(-3, 3)
    plt.ylim(-2, 2)
    plt.tight_layout()
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
        aspect='auto'
    )
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(im, ax=ax)


def save_2d_plot(
    epoch: int,
    x: ndarray,
    f: ndarray,
    loss: float,
    analytical: Callable
) -> None:
    output_dir = ensure_directory_exists()
    grid_size = int(x[:, 0].size**0.5)

    # Reshape x and f for 2D plotting
    # Reshape x[:, 0] into a grid
    x1 = x[:, 0].reshape(grid_size, grid_size)
    # Reshape x[:, 1] into a grid
    x2 = x[:, 1].reshape(grid_size, grid_size)

    # Reshape f into the same grid as x1 and x2
    f = f.reshape(grid_size, grid_size)

    # Compute the analytical solution
    analytical_f: ndarray = analytical(x[:, 0], x[:, 1])
    analytical_f = analytical_f.reshape(grid_size, grid_size)
    difference = f - analytical_f

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    create_plot(x1, x2, fig, axes[0], f, 'Model Output')
    create_plot(x1, x2, fig, axes[1], analytical_f, 'Analytical Solution')
    create_plot(x1, x2, fig, axes[2], difference, 'Difference')

    # Save the plot
    plt.suptitle(f'Epoch: {epoch}, Loss: {loss:.4f}')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/epoch_{epoch}.png')
    plt.close()
