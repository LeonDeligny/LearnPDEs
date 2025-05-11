'''
Plot functions.
'''
#  ======= Imports =======

import os
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from torch import Tensor
from pathlib import Path
from numpy import ndarray
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.tri import Triangulation
from typing import (
    Tuple,
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
    mesh = ax.scatter(
        x1, x2, c=data,
        cmap='viridis',
    )
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(mesh, ax=ax)


def save_htlm_airfoil_plot(
    output_dir: Path,
    epoch: int,
    inputs: ndarray,
    f: Tuple[ndarray, ndarray, ndarray],
    loss: float,
    airfoil_mask: ndarray,
) -> None:
    """
    Save an interactive 2D plot of the model output (u, v, p) using Plotly.
    """
    # Extract x1 and x2 from inputs
    x1, x2 = inputs[:, 0], inputs[:, 1]
    u, v, p = (np.asarray(component).flatten() for component in f)

    # Create a triangulation
    triang = Triangulation(x1, x2)
    triangles = triang.triangles
    triangle_mask = np.any(airfoil_mask[triangles], axis=1)
    triang.set_mask(triangle_mask)

    # Filter triangles based on the mask
    valid_triangles = triangles[~triangle_mask]

    # Prepare data for Plotly
    x = x1
    y = x2

    # Create subplots for u, v, and p
    fig = go.Figure()

    # Add u component
    fig.add_trace(go.Mesh3d(
        x=x, y=y, z=u,
        i=valid_triangles[:, 0], j=valid_triangles[:, 1], k=valid_triangles[:, 2],
        intensity=u,
        colorscale='RdBu',
        colorbar_title='u',
        name='u Component'
    ))

    # Add v component
    fig.add_trace(go.Mesh3d(
        x=x, y=y, z=v,
        i=valid_triangles[:, 0], j=valid_triangles[:, 1], k=valid_triangles[:, 2],
        intensity=v,
        colorscale='RdBu',
        colorbar_title='v',
        name='v Component'
    ))

    # Add p component
    fig.add_trace(go.Mesh3d(
        x=x, y=y, z=p,
        i=valid_triangles[:, 0], j=valid_triangles[:, 1], k=valid_triangles[:, 2],
        intensity=p,
        colorscale='RdBu',
        colorbar_title='p',
        name='p Component'
    ))

    # Update layout
    fig.update_layout(
        title=f'Epoch: {epoch}, Loss: {loss:.4f}',
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='Value'
        ),
        template='plotly_white'
    )

    # Save the interactive plot as an HTML file
    fig.write_html(f'{output_dir}/epoch_{epoch}.html')


def save_airfoil_plot(
    output_dir: Path,
    epoch: int,
    inputs: ndarray,
    f: Tuple[ndarray, ndarray, ndarray],
    loss: float,
    airfoil_mask: ndarray,
) -> None:
    """
    Save a 2D plot of the model output (u, v, p) using triangulation.
    """
    # Extract x1 and x2 from inputs
    x1, x2 = inputs[:, 0], inputs[:, 1]
    u, v, p = (np.asarray(component).flatten() for component in f)

    # Create a triangulation
    triang = Triangulation(x1, x2)
    triangles = triang.triangles
    triangle_mask = np.any(airfoil_mask[triangles], axis=1)
    triang.set_mask(triangle_mask)

    # Set up the colormap and normalization
    cmap = plt.cm.RdBu_r
    norm_u = plt.Normalize(vmin=0, vmax=1)
    norm_v = plt.Normalize(vmin=0, vmax=1)
    norm_p = plt.Normalize(vmin=-max(abs(p)), vmax=max(abs(p)))

    # Create a vertical layout for the plots
    # 3 rows, 1 column
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    # Plot u
    contour_u = axes[0].tricontourf(triang, u, cmap=cmap, levels=100, norm=norm_u)
    cbar_u = fig.colorbar(contour_u, ax=axes[0])
    cbar_u.ax.tick_params(labelsize=12)
    axes[0].set_title('u Component', fontsize=15)
    axes[0].set_xlabel('x', fontsize=12)
    axes[0].set_ylabel('y', fontsize=12)
    axes[0].triplot(triang, color='grey', lw=0.5)

    # Plot v
    contour_v = axes[1].tricontourf(triang, v, cmap=cmap, levels=100, norm=norm_v)
    cbar_v = fig.colorbar(contour_v, ax=axes[1])
    cbar_v.ax.tick_params(labelsize=12)
    axes[1].set_title('v Component', fontsize=15)
    axes[1].set_xlabel('x', fontsize=12)
    axes[1].set_ylabel('y', fontsize=12)
    axes[1].triplot(triang, color='grey', lw=0.5)

    # Plot p
    contour_p = axes[2].tricontourf(triang, p, cmap=cmap, levels=100, norm=norm_p)
    cbar_p = fig.colorbar(contour_p, ax=axes[2])
    cbar_p.ax.tick_params(labelsize=12)
    axes[2].set_title('p Component', fontsize=15)
    axes[2].set_xlabel('x', fontsize=12)
    axes[2].set_ylabel('y', fontsize=12)
    axes[2].triplot(triang, color='grey', lw=0.5)

    # Add title and save the plot
    plt.suptitle(f'Epoch: {epoch}, Loss: {loss:.4f}', fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'{output_dir}/epoch_{epoch}.png')
    plt.close()


def save_2d_plot(
    output_dir: Path,
    epoch: int,
    inputs: ndarray,
    f: ndarray,
    loss: float,
    analytical: Union[Callable, None],
) -> None:

    # Extract x1 and x2 from inputs
    x1, x2 = inputs[:, 0], inputs[:, 1]

    # Reshape x1, x2, and f into 2D grids
    x1_grid = x1
    x2_grid = x2
    f_grid = f

    ncols = 1 if analytical is None else 3
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


def plot_xy(xy: Tensor) -> None:
    """
    Plots the (x, y) coordinates from a tensor or numpy array.
    """
    if hasattr(xy, 'detach'):
        xy_np = xy.detach().cpu().numpy()
    else:
        xy_np = xy

    plt.figure(figsize=(6, 6))
    plt.scatter(xy_np[:, 0], xy_np[:, 1], s=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Mesh Node Coordinates')
    plt.axis('equal')
    plt.show()
