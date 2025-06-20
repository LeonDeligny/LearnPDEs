'''
Plot functions.
'''
#  ======= Imports =======

import os
import itertools
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt

from learnpdes.utils.utility import compute_normals

from torch import Tensor
from pathlib import Path
from numpy import ndarray
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.tri import Triangulation
from matplotlib.colors import (
    Normalize,
    TwoSlopeNorm,
)
from typing import (
    Union,
    Callable,
)

from learnpdes import (
    EXPONENTIAL_SCENARIO,
    COSINUS_SCENARIO,
    LAPLACE_SCENARIO,
)

# ======= Functions =======


def get_plot_func(scenario: str) -> Callable:
    if scenario in [EXPONENTIAL_SCENARIO, COSINUS_SCENARIO]:
        return save_plot
    elif scenario == LAPLACE_SCENARIO:
        return save_2d_plot
    else:
        return save_airfoil_plot


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


def ensure_directory_exists(
    output_dir: str = './gifs/epochs',
) -> Path:
    '''
    Ensure directory exists and
    that the directory is cleaned before each run.
    '''
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            if file.endswith('.png'):
                os.remove(os.path.join(output_dir, file))
        print(f'Removing folder {output_dir=}')
    else:
        os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_plot(
    output_dir: Path,
    epoch: int,
    inputs: ndarray,
    f: ndarray,
    loss: float,
    geometry_mask: ndarray,
    analytical: Callable
) -> None:

    inputs = inputs[geometry_mask]
    f = f[geometry_mask]

    plt.figure()
    plt.plot(inputs, f, label='NN Prediction')
    plt.plot(inputs, analytical(inputs), label='Analytical Solution')
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
    # Ensure data size matches x1 and x2
    if data.size != x1.size:
        raise ValueError(
            f"Size mismatch: 'data' has {data.size} elements, "
            f"but 'x1' and 'x2' have {x1.size} elements."
        )

    mesh = ax.scatter(
        x1, x2, c=data,
        cmap='viridis',
    )
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(mesh, ax=ax)


def save_airfoil_plot(
    output_dir: Path,
    epoch: int,
    inputs: ndarray,
    f: tuple[ndarray, ndarray, ndarray],
    loss: float,
    geometry_mask: Union[ndarray, None],
    analytical: None = None,
) -> None:
    """
    Save a 2D plot of the model output (u, v, p) using triangulation.
    """
    if analytical is not None:
        raise NotImplementedError(
            "No analytical solution for flow around airfoil."
        )

    # Extract x1 and x2 from inputs
    x1, x2 = inputs[:, 0], inputs[:, 1]
    u, v, p = (np.asarray(component).flatten() for component in f)

    # Create a triangulation
    triang = Triangulation(x1, x2)
    triangles = triang.triangles
    if geometry_mask is not None:
        if hasattr(geometry_mask, "cpu"):
            geometry_mask = geometry_mask.cpu().numpy()
        triangle_mask = np.any(geometry_mask[triangles], axis=1)
        triang.set_mask(triangle_mask)

    # Set up the colormap and normalization
    cmap = plt.cm.RdBu_r
    # Use a fixed or symmetric range for better contrast:
    if u.min() < 0.0:
        norm_u = TwoSlopeNorm(vmin=u.min(), vcenter=0.0, vmax=u.max())
    else:
        norm_u = Normalize(vmin=u.min(), vmax=u.max())
    v_abs_max = np.nanmax(np.abs(v))
    norm_v = TwoSlopeNorm(vmin=-v_abs_max, vcenter=0.0, vmax=v_abs_max)
    norm_p = Normalize(vmin=-1.0, vmax=1.0)

    # Create a vertical layout for the plots
    # 3 rows, 1 column
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    # Plot u
    contour_u = axes[0].tricontourf(
        triang,
        u,
        cmap=cmap,
        levels=100,
        norm=norm_u
    )
    cbar_u = fig.colorbar(contour_u, ax=axes[0])
    cbar_u.ax.tick_params(labelsize=12)
    axes[0].set_title('u Component', fontsize=15)
    axes[0].set_xlabel('x', fontsize=12)
    axes[0].set_ylabel('y', fontsize=12)
    axes[0].triplot(triang, color='grey', lw=0.0)

    # Plot v
    contour_v = axes[1].tricontourf(
        triang,
        v,
        cmap=cmap,
        levels=100,
        norm=norm_v
    )
    cbar_v = fig.colorbar(contour_v, ax=axes[1])
    cbar_v.ax.tick_params(labelsize=12)
    axes[1].set_title('v Component', fontsize=15)
    axes[1].set_xlabel('x', fontsize=12)
    axes[1].set_ylabel('y', fontsize=12)
    axes[1].triplot(triang, color='grey', lw=0.0)

    # Plot p
    contour_p = axes[2].tricontourf(
        triang,
        p,
        cmap=cmap,
        levels=100,
        norm=norm_p
    )
    cbar_p = fig.colorbar(contour_p, ax=axes[2])
    cbar_p.ax.tick_params(labelsize=12)
    axes[2].set_title('p Component', fontsize=15)
    axes[2].set_xlabel('x', fontsize=12)
    axes[2].set_ylabel('y', fontsize=12)
    axes[2].triplot(triang, color='grey', lw=0.0)

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
    geometry_mask: None,
    analytical: Union[Callable, None],
) -> None:
    if geometry_mask is not None:
        raise ValueError(
            "geometry_mask is not None for grid plot"
        )

    # Extract x1 and x2 from inputs
    x1, x2 = inputs[:, 0], inputs[:, 1]

    # Reshape x1, x2, and f into 2D grids
    n = int(len(x1)**0.5)
    x1_grid = x2.reshape(n, n).T
    x2_grid = x1.reshape(n, n).T
    f_grid = f.reshape(n, n).T

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


def plot_mesh(xy: Tensor, mesh_masks: dict[str, Tensor]) -> None:
    geometry_mask = mesh_masks['airfoil']
    n_x, n_y = compute_normals(xy, geometry_mask)
    plt.figure(figsize=(8, 6))
    colors = itertools.cycle(['blue', 'red', 'green', 'orange', 'purple'])
    for name, mesh in mesh_masks.items():
        plt.scatter(
            xy[mesh, 0],
            xy[mesh, 1],
            s=10,
            label=name,
            color=next(colors),
            alpha=0.8
        )
    plt.quiver(
        xy[geometry_mask, 0], xy[geometry_mask, 1],
        n_x[geometry_mask], n_y[geometry_mask],
        color="blue", scale=100, width=0.003, label="Normals"
    )
    plt.legend()
    plt.axis("equal")
    plt.title("Mesh with Cross-Sections")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
