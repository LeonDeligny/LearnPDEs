"""Shared contracts for numerical functions and training checkpoints."""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, NotRequired, TypedDict

from numpy.typing import ArrayLike, NDArray
from torch import Tensor

# Mesh indices, masks and floating fields deliberately retain their own dtypes.
type Array = NDArray[Any]
type AnalyticalValue = Array | Tensor
type Analytical = Callable[..., AnalyticalValue | tuple[AnalyticalValue, ...]]
type TensorFunction = Callable[[Tensor], Tensor]
type LossResult = tuple[Tensor, Tensor, Tensor | tuple[Tensor, ...], Tensor | None]
type LossFunction = Callable[[], LossResult]
type LossHistory = list[tuple[int, float]]
type CollocationData = tuple[
    Tensor,
    dict[str, Tensor],
    int,
    Analytical | None,
    TensorFunction,
    TensorFunction,
    TensorFunction,
]


class PointGroup(TypedDict):
    kind: str
    coordinates: Tensor | ArrayLike


class RecordedPointGroup(TypedDict):
    kind: str
    coordinates: list[list[float]]


type PointGroups = Mapping[str, PointGroup | RecordedPointGroup]
type RecordedPoints = dict[str, RecordedPointGroup]


class Checkpoint(TypedDict):
    step: int
    loss: float
    prediction: Array
    collocation: RecordedPoints | None
    evaluation_mse: dict[str, float]


class FieldEvaluation(TypedDict):
    inputs: Array
    f: Array | tuple[Array, ...]
    geometry_mask: Array | None
    triangles: NotRequired[Array]
    boundary_edges: NotRequired[Array | None]
    pressure_label: NotRequired[str]


class TrainingParams(TypedDict):
    learning_rate: float
    epochs: int
    lbfgs_steps: NotRequired[int]
    resample_every: NotRequired[int]


class ExportPaths(TypedDict, total=False):
    output_dir: str | Path
    frame_dir: str | Path
    html_path: str | Path | None
    gif_path: str | Path | None


class PlotOptions(ExportPaths):
    plot_func: Callable[..., None]
    evaluate: NotRequired[Callable[[], FieldEvaluation]]
    max_frames: NotRequired[int]
    duration_ms: NotRequired[int]
    final_hold_ms: NotRequired[int]
