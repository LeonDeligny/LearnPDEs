'''
Plot functions.
'''

#  ======= Imports =======

import os
import imageio

import numpy as np
import matplotlib.pyplot as plt

from torch import Tensor
from pathlib import Path

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


def save_plot(epoch: int, x: Tensor, y: Tensor) -> None:
    # Ensure the directory exists
    output_dir = './gifs/epochs'
    os.makedirs(output_dir, exist_ok=True)

    # Convert tensors to numpy arrays
    x = x.detach().numpy()
    y = y.detach().numpy()
    mask = x < 10
    x = x[mask]
    y = y[mask]

    plt.figure()
    plt.plot(x, y, label='NN Prediction')
    plt.plot(x, np.exp(x), label='Analytical Solution')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.title(f'Epoch {epoch}')
    plt.savefig(f'{output_dir}/epoch_{epoch}.png')
    plt.close()

# ======= Main =======


if __name__ == '__main__':
    print('Nothing to execute.')
