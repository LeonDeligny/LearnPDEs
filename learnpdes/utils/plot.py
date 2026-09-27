"""
Plot functions.
"""
#  ======= Imports =======

import os
import itertools
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from learnpdes.utils.utility import compute_normals

from torch import Tensor
from pathlib import Path
from functools import partial
from numpy import ndarray
from matplotlib.collections import LineCollection
from matplotlib.tri import Triangulation
from matplotlib.colors import Normalize
from typing import Callable

from learnpdes import (
    EXPONENTIAL_SCENARIO,
    COSINUS_SCENARIO,
    LAPLACE_SCENARIO,
)

# ======= Functions =======


def get_plot_func(scenario: str) -> Callable:
    """Create a plotter with color limits shared across all of its frames."""
    if scenario in [EXPONENTIAL_SCENARIO, COSINUS_SCENARIO]:
        return save_plot
    elif scenario == LAPLACE_SCENARIO:
        return partial(save_2d_plot, color_norms={})
    else:
        return partial(save_airfoil_plot, color_norms={})


def create_gif(
    output_path: Path = './gifs/training_process.gif',
    input_folder: Path = './gifs/epochs',
    duration_ms: int = 100,
    final_hold_ms: int = 2000,
) -> None:
    """Write looping GIFs with explicit millisecond delays, including a final hold."""
    for delay in (duration_ms, final_hold_ms):
        if not isinstance(delay, int) or not 10 <= delay <= 655350 or delay % 10:
            raise ValueError(
                'GIF delays must be multiples of 10 ms between 10 and 655350.'
            )
    files = sorted(
        Path(input_folder).glob('epoch_*.png'),
        key=lambda path: int(path.stem.split('_')[1]),
    )
    if not files:
        raise ValueError(f'No epoch frames found in {input_folder}.')
    images = []
    try:
        for file in files:
            with Image.open(file) as frame:
                images.append(frame.convert('RGB'))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=[duration_ms] * (len(images) - 1) + [final_hold_ms],
            loop=0,
            disposal=2,
        )
    finally:
        for image in images:
            image.close()


def ensure_directory_exists(
    output_dir: str = './gifs/epochs',
) -> Path:
    """
    Ensure directory exists and
    that the directory is cleaned before each run.
    """
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            if file.endswith('.png'):
                os.remove(os.path.join(output_dir, file))
        print(f'Removing folder {output_dir=}')
    else:
        os.makedirs(output_dir, exist_ok=True)
    return output_dir


PLOT_STYLE = {
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.titlesize': 15,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'savefig.facecolor': 'white',
}


def error_metrics(prediction: ndarray, reference: ndarray) -> dict[str, float]:
    """Discrete errors on the supplied evaluation samples, not the training loss."""
    prediction, reference = np.asarray(prediction), np.asarray(reference)
    if prediction.shape != reference.shape:
        raise ValueError('Prediction and reference shapes must match.')
    difference = prediction - reference
    denominator = np.linalg.norm(reference.ravel())
    numerator = np.linalg.norm(difference.ravel())
    relative = (
        numerator / denominator if denominator else (0.0 if numerator == 0 else np.inf)
    )
    return {
        'relative_l2': float(relative),
        'max_abs': float(np.max(np.abs(difference))),
    }


def publication_figure(epoch, loss, loss_history, total_epochs, metrics=None):
    """Use fixed axes rectangles so panel and colorbar positions never jump."""
    fig = plt.figure(figsize=(16, 9), dpi=100)
    axes = [fig.add_axes([0.06 + i * 0.32, 0.40, 0.215, 0.43]) for i in range(3)]
    fig.suptitle(
        f'Training step {epoch:,}  |  Loss {float(loss):.2e}', y=0.965, fontsize=20
    )
    if metrics is not None:
        subtitle = (
            f'Evaluation grid: relative L₂ = {metrics["relative_l2"]:.2e}'
            f'   |   maximum absolute error = {metrics["max_abs"]:.2e}'
        )
    else:
        subtitle = (
            'No reference solution available; prediction errors are not reported.'
        )
    fig.text(0.5, 0.905, subtitle, ha='center', fontsize=13)
    convergence = fig.add_axes([0.075, 0.12, 0.86, 0.145])
    history = loss_history if loss_history is not None else [(epoch, float(loss))]
    steps, losses = np.asarray(history).T
    convergence.semilogy(
        steps, np.maximum(losses, np.finfo(float).tiny), color='#315b87', lw=1.5
    )
    convergence.set_xlim(0, max(total_epochs or epoch, 1))
    convergence.set_xlabel('Training step')
    convergence.set_ylabel('Objective')
    convergence.set_title('Convergence', loc='left', fontsize=13)
    convergence.grid(alpha=0.2)
    fig.text(
        0.5,
        0.025,
        'Visualization samples are separate from training samples; finer rendering does not establish accuracy.',
        ha='center',
        fontsize=11,
        color='#555555',
    )
    return fig, axes


def save_figure(fig, output_dir, epoch):
    # Do not use bbox_inches="tight": it changes frame sizes and axes positions.
    fig.savefig(Path(output_dir) / f'epoch_{epoch}.png', dpi=100)
    plt.close(fig)


def save_plot(
    output_dir: Path,
    epoch: int,
    inputs: ndarray,
    f: ndarray,
    loss: float,
    geometry_mask: ndarray | None,
    analytical: Callable | None,
    loss_history=None,
    total_epochs=None,
) -> None:
    x, prediction = np.asarray(inputs).ravel(), np.asarray(f).ravel()
    if geometry_mask is not None:
        mask = np.asarray(geometry_mask).ravel()
        x, prediction = x[mask], prediction[mask]
    order = np.argsort(x)
    x, prediction = x[order], prediction[order]
    reference = np.asarray(analytical(x)) if analytical is not None else None
    metrics = error_metrics(prediction, reference) if reference is not None else None
    with plt.rc_context(PLOT_STYLE):
        fig, axes = publication_figure(epoch, loss, loss_history, total_epochs, metrics)
        axes[0].plot(x, prediction, color='#315b87', lw=2)
        axes[0].set_title('Prediction')
        axes[0].set_ylabel('f(x)')
        if reference is not None:
            axes[1].plot(x, reference, color='#27816b', lw=2)
            axes[1].set_title('Reference')
            axes[1].set_ylabel('f(x)')
            lo = min(prediction.min(), reference.min())
            hi = max(prediction.max(), reference.max())
            pad = max(0.05 * (hi - lo), 1e-6)
            for ax in axes[:2]:
                ax.set_ylim(lo - pad, hi + pad)
            axes[2].plot(x, prediction - reference, color='#ad493b', lw=1.8)
            axes[2].axhline(0, color='#666666', lw=0.8)
            axes[2].set_title('Signed error')
            axes[2].set_ylabel('Prediction − reference')
            axes[2].ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
        else:
            for ax, title in zip(axes[1:], ['Reference', 'Signed error']):
                ax.set_title(title)
                ax.text(0.5, 0.5, 'Unavailable', transform=ax.transAxes, ha='center')
        for ax in axes:
            ax.set_xlabel('x')
            ax.set_xlim(x.min(), x.max())
            ax.grid(alpha=0.2)
        save_figure(fig, output_dir, epoch)


def create_plot(
    x1, x2, fig, ax, data, title, label='f(x, y)', norm=None, cmap='viridis'
):
    mesh = ax.pcolormesh(
        x1, x2, data, cmap=cmap, norm=norm, shading='nearest', rasterized=True
    )
    ax.set_xlim(np.min(x1), np.max(x1))
    ax.set_ylim(np.min(x2), np.max(x2))
    finish_field(fig, ax, mesh, title, label)


def finish_field(fig, ax, mesh, title, label):
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # Keep the panel rectangle fixed while retaining equal physical axis scales.
    ax.set_aspect('equal', adjustable='box')
    slot = ax.get_position()
    cax = fig.add_axes([slot.x1 + 0.008, slot.y0, 0.011, slot.height])
    colorbar = fig.colorbar(mesh, cax=cax, extend='both')
    colorbar.set_label(label)
    colorbar.formatter.set_powerlimits((-2, 2))
    colorbar.update_ticks()


def value_norm(*fields, symmetric=False):
    lo = min(float(np.min(field)) for field in fields)
    hi = max(float(np.max(field)) for field in fields)
    if symmetric:
        bound = max(abs(lo), abs(hi)) or 1.0
        return Normalize(-bound, bound)
    if lo == hi:
        pad = abs(lo) * 0.01 or 1.0
        lo, hi = lo - pad, hi + pad
    return Normalize(lo, hi)


def fixed_norm(color_norms, name, *fields, symmetric=False):
    """Initialize limits from the first frame and keep them for this plotter."""
    if name not in color_norms:
        color_norms[name] = value_norm(*fields, symmetric=symmetric)
    return color_norms[name]


def save_airfoil_plot(
    output_dir: Path,
    epoch: int,
    inputs: ndarray,
    f: tuple[ndarray, ndarray, ndarray],
    loss: float,
    geometry_mask: ndarray | None,
    analytical: None = None,
    triangulation: Triangulation | None = None,
    boundary_edges: ndarray | None = None,
    loss_history=None,
    total_epochs=None,
    pressure_label='Pressure p (model units)',
    *,
    color_norms: dict[str, Normalize] | None = None,
) -> None:
    """Save flow fields with fixed scales when color_norms is reused across calls.

    get_plot_func owns this state for an animation. Callers can also supply
    explicit Normalize objects under 'u', 'v', and 'p' to choose the ranges.
    """
    if analytical is not None:
        raise NotImplementedError('No analytical solution for flow around airfoil.')
    triang = triangulation
    if triang is None:
        triang = Triangulation(inputs[:, 0], inputs[:, 1])
        if geometry_mask is not None:
            mask = np.asarray(geometry_mask)
            triang.set_mask(np.any(mask[triang.triangles], axis=1))
    if color_norms is None:
        color_norms = {}
    visible_nodes = np.unique(triang.get_masked_triangles())
    # The unit inlet speed prevents tiny initial velocities setting tiny limits.
    velocity_range = np.array([-1.0, 1.0])
    with plt.rc_context(PLOT_STYLE):
        fig, axes = publication_figure(epoch, loss, loss_history, total_epochs)
        titles = ['Horizontal velocity', 'Vertical velocity', 'Pressure']
        labels = [
            'Velocity u (model units)',
            'Velocity v (model units)',
            pressure_label,
        ]
        for index, (ax, field, title, label) in enumerate(zip(axes, f, titles, labels)):
            values = np.asarray(field).ravel()
            fields = (values[visible_nodes],)
            if index < 2:
                fields += (velocity_range,)
            norm = fixed_norm(
                color_norms, ('u', 'v', 'p')[index], *fields, symmetric=index < 2
            )
            mesh = ax.tripcolor(
                triang,
                values,
                shading='gouraud',
                cmap='RdBu_r' if index < 2 else 'viridis',
                norm=norm,
                rasterized=True,
            )
            if boundary_edges is not None:
                ax.add_collection(
                    LineCollection(boundary_edges, colors='#222222', lw=0.8)
                )
            finish_field(fig, ax, mesh, title, label)
        save_figure(fig, output_dir, epoch)


def save_2d_plot(
    output_dir: Path,
    epoch: int,
    inputs: ndarray,
    f: ndarray,
    loss: float,
    geometry_mask: None,
    analytical: Callable | None,
    loss_history=None,
    total_epochs=None,
    *,
    color_norms: dict[str, Normalize] | None = None,
) -> None:
    """Save prediction/reference and signed error with shared, fixed color limits.

    Reuse color_norms across calls or obtain a stateful plotter from get_plot_func.
    Supply explicit norms under 'solution' and 'error' to choose the ranges.
    """
    if geometry_mask is not None:
        raise ValueError('geometry_mask is not None for grid plot')
    # Accept rectangular grids and arbitrary input ordering; rows are y, columns x.
    x, y = np.unique(inputs[:, 0]), np.unique(inputs[:, 1])
    if len(x) * len(y) != len(inputs):
        raise ValueError('Expected a complete rectangular visualization grid.')
    order = np.lexsort((inputs[:, 0], inputs[:, 1]))
    x_grid, y_grid = np.meshgrid(x, y)
    prediction = np.asarray(f).ravel()[order].reshape(len(y), len(x))
    reference = (
        np.asarray(analytical(x_grid, y_grid)) if analytical is not None else None
    )
    metrics = error_metrics(prediction, reference) if reference is not None else None
    if color_norms is None:
        color_norms = {}
    with plt.rc_context(PLOT_STYLE):
        fig, axes = publication_figure(epoch, loss, loss_history, total_epochs, metrics)
        fields = (prediction,) if reference is None else (prediction, reference)
        norm = fixed_norm(color_norms, 'solution', *fields)
        create_plot(x_grid, y_grid, fig, axes[0], prediction, 'Prediction', norm=norm)
        if reference is not None:
            create_plot(x_grid, y_grid, fig, axes[1], reference, 'Reference', norm=norm)
            difference = prediction - reference
            create_plot(
                x_grid,
                y_grid,
                fig,
                axes[2],
                difference,
                'Signed error',
                label='Prediction − reference',
                norm=fixed_norm(color_norms, 'error', difference, symmetric=True),
                cmap='RdBu_r',
            )
        else:
            for ax, title in zip(axes[1:], ['Reference', 'Signed error']):
                ax.set_title(title)
                ax.text(0.5, 0.5, 'Unavailable', transform=ax.transAxes, ha='center')
        save_figure(fig, output_dir, epoch)


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
            xy[mesh, 0], xy[mesh, 1], s=10, label=name, color=next(colors), alpha=0.8
        )
    plt.quiver(
        xy[geometry_mask, 0],
        xy[geometry_mask, 1],
        n_x[geometry_mask],
        n_y[geometry_mask],
        color='blue',
        scale=100,
        width=0.003,
        label='Normals',
    )
    plt.legend()
    plt.axis('equal')
    plt.title('Mesh with Cross-Sections')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
