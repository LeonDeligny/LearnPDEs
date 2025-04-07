'''
Plot functions.
'''

#  ======= Imports =======

import os
import imageio

import matplotlib.pyplot as plt

from torch import Tensor
from pathlib import Path
from typing import Callable

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


def ensure_directory_exists() -> None:
    output_dir = './gifs/epochs'
    os.makedirs(output_dir, exist_ok=True)

    return output_dir


def save_plot(
    epoch: int,
    x: Tensor,
    f: Tensor,
    loss: float,
    analytical: Callable
) -> None:
    output_dir = ensure_directory_exists()

    # Convert tensors to numpy arrays
    x = x.detach().numpy()
    f = f.detach().numpy()
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


def save_2d_plot(
    epoch: int,
    x: Tensor,
    f: Tensor,
    loss: float,
    analytical: Callable
) -> None:
    output_dir = ensure_directory_exists()

    # Convert tensors to numpy arrays
    x = x.detach().numpy()
    f = f.detach().numpy()

    # Reshape x and f for 2D plotting
    x1 = x[:, 0].reshape(int(x[:, 0].size**0.5), -1)  # Reshape x[:, 0] into a grid
    x2 = x[:, 1].reshape(int(x[:, 1].size**0.5), -1)  # Reshape x[:, 1] into a grid
    f = f.reshape(x1.shape)  # Reshape f into the same grid as x1 and x2

    # Compute the analytical solution
    analytical_f = analytical(x[:, 0], x[:, 1])  # Compute analytical solution
    analytical_f = analytical_f.detach().numpy().reshape(x1.shape)  # Reshape to match grid

    # Compute the difference between the model and the analytical solution
    difference = f - analytical_f

    # Create the plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # Three subplots

    # Left plot: Model output
    im1 = axes[0].imshow(
        f,
        extent=[x1.min(), x1.max(), x2.min(), x2.max()],
        origin='lower',
        aspect='auto'
    )
    axes[0].set_title('Model Output')
    axes[0].set_xlabel('x1')
    axes[0].set_ylabel('x2')
    fig.colorbar(im1, ax=axes[0])

    # Middle plot: Analytical solution
    im2 = axes[1].imshow(
        analytical_f,
        extent=[x1.min(), x1.max(), x2.min(), x2.max()],
        origin='lower',
        aspect='auto'
    )
    axes[1].set_title('Analytical Solution')
    axes[1].set_xlabel('x1')
    axes[1].set_ylabel('x2')
    fig.colorbar(im2, ax=axes[1])

    # Right plot: Difference
    im3 = axes[2].imshow(
        difference,
        extent=[x1.min(), x1.max(), x2.min(), x2.max()],
        origin='lower',
        aspect='auto'
    )
    axes[2].set_title('Difference (Model - Analytical)')
    axes[2].set_xlabel('x1')
    axes[2].set_ylabel('x2')
    fig.colorbar(im3, ax=axes[2])

    # Save the plot
    plt.suptitle(f'Epoch: {epoch}, Loss: {loss:.4f}')
    plt.savefig(f'{output_dir}/epoch_{epoch}.png')
    plt.close()


# ======= Main =======


if __name__ == '__main__':
    print('Nothing to execute.')
