"""Plotly views of scalar and flow PINN training, with optional image export."""

from __future__ import annotations

from collections.abc import Callable
from html import escape
from pathlib import Path
from typing import Any, cast

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

from learnpdes import COSINUS_SCENARIO
from learnpdes.scenarios import cosinus
from learnpdes.scenarios.registry import get_scenario
from learnpdes.types import Analytical, Array, Checkpoint, LossHistory, PointGroups
from learnpdes.utils.collocation import add_overlay, point_traces, snapshot, summary
from learnpdes.utils.mesh_raster import MeshRaster
from learnpdes.utils.plot import error_metrics, value_range
from learnpdes.utils.plot_style import (
    BLUE,
    COLORS,
    GREEN,
    RED,
    align_field_panels,
    colorbar,
    scientific_style,
)
from learnpdes.visualization.grids import rectangular_triangles


def cosinus_axes(
    fig: go.Figure, rows: tuple[int, ...] = (1, 2), cols: tuple[int, ...] = (1,)
) -> None:
    """Mark the training interval and display pi symbolically and numerically."""
    for row, col in ((row, col) for row in rows for col in cols):
        fig.update_xaxes(
            title_text='x (π ≈ 3.141593)',
            tickmode='array',
            tickvals=[multiple * np.pi for multiple in range(-3, 4)],
            ticktext=[
                '−3π<br>−9.425',
                '−2π<br>−6.283',
                '−π<br>−3.142',
                '0',
                'π<br>3.142',
                '2π<br>6.283',
                '3π<br>9.425',
            ],
            row=row,
            col=col,
        )
        cast(Callable[..., go.Figure], fig.add_vrect)(
            x0=-np.pi,
            x1=np.pi,
            fillcolor='#0f766e',
            opacity=0.07,
            line_width=0,
            layer='below',
            row=row,
            col=col,
        )
        for bound in cosinus.TRAINING_BOUNDS:
            cast(Callable[..., go.Figure], fig.add_vline)(
                x=bound, line_dash='dot', line_color='#64748b', row=row, col=col
            )


class InteractivePlot:
    """Capture Trainer checkpoints and build a figure after training finishes.

    Scalar fields use lines or heatmaps. Flow fields use the supplied fluid
    triangles, retaining the airfoil hole and original boundary geometry.
    """

    def __init__(
        self,
        scenario: str,
        *,
        color_limits: dict[str, tuple[float, float]] | None = None,
        cosinus_order: int = 2,
    ) -> None:
        """Configure scenario fields, reference requirements, and plot scales."""
        case = get_scenario(scenario)
        self.scenario = scenario
        cosinus.validate_order(cosinus_order)
        self.cosinus_order = cosinus_order
        self.flow = case.output_kind != 'scalar'
        self.requires_reference = case.reference_type == 'analytical' or not self.flow
        self.color_limits = dict(color_limits or {})
        for limits in self.color_limits.values():
            if (
                len(limits) != 2
                or not np.isfinite(limits).all()
                or limits[0] >= limits[1]
            ):
                raise ValueError('Colour limits must be finite increasing pairs.')
        self.reset()

    def reset(self) -> None:
        self.checkpoints: list[Checkpoint] = []
        self.coordinates: Array | None = None
        self.reference: Array | None = None
        self.loss_history: LossHistory = []
        self.triangles: Array | None = None
        self.boundary_edges: Array | None = None
        self.pressure_label = 'Pressure p (model units)'

    def __call__(
        self,
        folder: str | Path,
        *,
        epoch: int,
        inputs: Array,
        f: Array | tuple[Array, ...],
        loss: float,
        analytical: Analytical | None,
        loss_history: LossHistory,
        triangles: Array | None = None,
        boundary_edges: Array | None = None,
        geometry_mask: Array | None = None,
        pressure_label: str = 'Pressure p (model units)',
        collocation: PointGroups | None = None,
        **kwargs: object,
    ) -> None:
        coordinates = np.asarray(inputs).reshape(len(inputs), -1)
        prediction = (
            np.asarray(f).reshape(3, -1) if self.flow else np.asarray(f).ravel()
        )
        if analytical is None and self.requires_reference:
            raise ValueError('This scenario requires a reference solution.')
        if self.coordinates is None:
            self._initialize_grid(
                coordinates,
                analytical,
                triangles,
                boundary_edges,
                geometry_mask,
                pressure_label,
            )
        elif not np.array_equal(coordinates, self.coordinates):
            raise ValueError('All checkpoints must use the same visualization grid.')
        expected = (3, len(coordinates)) if self.flow else (len(coordinates),)
        if self.reference is not None and self.reference.shape != expected:
            raise ValueError('Prediction and reference shapes must match.')
        self._validate_checkpoint(epoch, prediction, expected, loss)
        frame: Checkpoint = {
            'step': epoch,
            'loss': float(loss),
            'prediction': prediction.copy(),
            'collocation': snapshot(collocation),
            'evaluation_mse': {},
        }
        if self.scenario == COSINUS_SCENARIO:
            frame['evaluation_mse'] = cosinus.region_mse(coordinates, prediction)
        self.checkpoints.append(frame)
        self.loss_history = list(loss_history)

    def _validate_checkpoint(
        self, epoch: int, prediction: Array, expected: tuple[int, ...], loss: float
    ) -> None:
        if prediction.shape != expected:
            raise ValueError('Prediction and reference shapes must match.')
        if not np.isfinite(prediction).all() or not np.isfinite(loss):
            raise ValueError('Cannot plot non-finite predictions or loss.')
        if self.checkpoints and epoch <= self.checkpoints[-1]['step']:
            raise ValueError('Checkpoint steps must be strictly increasing.')

    def _initialize_grid(
        self,
        coordinates: Array,
        analytical: Analytical | None,
        triangles: Array | None,
        boundary_edges: Array | None,
        geometry_mask: Array | None,
        pressure_label: str,
    ) -> None:
        self.coordinates = coordinates.copy()
        if self.flow:
            self.triangles = self._flow_cells(coordinates, triangles, geometry_mask)
            self.boundary_edges = (
                None if boundary_edges is None else boundary_edges.copy()
            )
            self.pressure_label = pressure_label
        if analytical is not None and self.requires_reference:
            reference = np.asarray(analytical(*coordinates.T))
            if self.flow and reference.shape != (3, len(coordinates)):
                raise ValueError('Expected reference fields (u, v, p).')
            self.reference = reference if self.flow else reference.ravel()

    @staticmethod
    def _flow_cells(
        coordinates: Array,
        triangles: Array | None,
        geometry_mask: Array | None,
    ) -> Array:
        cells = (
            rectangular_triangles(coordinates)
            if triangles is None
            else np.asarray(triangles)
        )
        if (
            cells.ndim != 2
            or cells.shape[1] != 3
            or not len(cells)
            or not np.issubdtype(cells.dtype, np.integer)
            or cells.min() < 0
            or cells.max() >= len(coordinates)
        ):
            raise ValueError('Invalid fluid triangle indices.')
        if geometry_mask is not None:
            cells = cells[~np.any(np.asarray(geometry_mask)[cells], axis=1)]
        if not len(cells):
            raise ValueError('The flow mesh contains no visible fluid cells.')
        return cells.copy()

    def figure(self) -> go.Figure:
        if not self.checkpoints:
            raise ValueError('Capture at least one checkpoint before exporting.')
        if self.flow:
            return self._flow_figure()
        return self._scalar_figure()

    def _scalar_figure(self) -> go.Figure:
        if self.coordinates is None or self.reference is None:
            raise ValueError('Capture a complete scalar checkpoint before exporting.')
        coordinates = self.coordinates
        reference = self.reference
        field = coordinates.shape[1] == 2
        predictions = [frame['prediction'] for frame in self.checkpoints]
        solution = self.color_limits.get(
            'solution', value_range(reference, *predictions)
        )
        errors = [prediction - reference for prediction in predictions]
        error = self.color_limits.get('error', value_range(*errors, symmetric=True))
        trace = self._scalar_trace(coordinates)
        fig, extrapolation, titles = self._scalar_subplots()
        for index, (values, name) in enumerate(
            zip((predictions[0], reference, errors[0]), titles)
        ):
            fig.add_trace(trace(values, name, index), row=1, col=index + 1)
        self._add_convergence(fig, row=2)
        if extrapolation:
            self._add_scalar_diagnostics(fig)
        self._configure_scalar_axes(fig, coordinates, solution, error)
        fig.frames = self._scalar_frames(trace, errors)
        immediate, controls = self._playback_controls()
        self._finish_figure(fig, controls, immediate)
        if field:
            self._scalar_colors(fig, solution, error)
        self._scalar_layout(fig, field=field, extrapolation=extrapolation)
        self._add_collocation(fig, rug=not field)
        return fig

    def _scalar_frames(
        self,
        trace: Callable[[Array, str, int], go.Heatmap | go.Scatter],
        errors: list[Array],
    ) -> list[go.Frame]:
        return [
            go.Frame(
                name=str(frame['step']),
                traces=[0, 2, 4],
                data=[
                    trace(frame['prediction'], 'Prediction', 0),
                    trace(difference, 'Signed error', 2),
                    self._cursor(frame),
                ],
                layout={'title': {'text': self._title(frame)}},
            )
            for frame, difference in zip(self.checkpoints, errors)
        ]

    def _scalar_subplots(self) -> tuple[go.Figure, bool, list[str]]:
        extrapolation = self.scenario == COSINUS_SCENARIO and all(
            region in self.checkpoints[0]['evaluation_mse']
            for region in ('inside', 'outside', 'full')
        )
        specs = [[{}, {}, {}], [{'colspan': 3}, None, None]]
        if extrapolation:
            specs.append([{'colspan': 3}, None, None])
        titles = ['Prediction', 'Reference', 'Signed error', 'Convergence']
        if extrapolation:
            titles.append('Evaluation MSE · excluded from training')
        fig = make_subplots(rows=len(specs), cols=3, specs=specs, subplot_titles=titles)
        return fig, extrapolation, titles

    @staticmethod
    def _scalar_colors(
        fig: go.Figure, solution: tuple[float, float], error: tuple[float, float]
    ) -> None:
        for axis, limits, scale in [
            ('coloraxis', solution, 'Viridis'),
            ('coloraxis3', solution, 'Viridis'),
            ('coloraxis2', error, 'RdBu_r'),
        ]:
            fig.update_layout(
                {axis: {'cmin': limits[0], 'cmax': limits[1], 'colorscale': scale}}
            )
        fig.update_xaxes(showgrid=False, row=1)
        fig.update_yaxes(showgrid=False, row=1)

    @staticmethod
    def _scalar_trace(
        coordinates: Array,
    ) -> Callable[[Array, str, int], go.Heatmap | go.Scatter]:
        field = coordinates.shape[1] == 2
        xy = coordinates
        if field:
            x, y = np.unique(xy[:, 0]), np.unique(xy[:, 1])
            if len(x) * len(y) != len(xy) or len(np.unique(xy, axis=0)) != len(xy):
                raise ValueError('Expected a complete rectangular visualization grid.')
            order = np.lexsort((xy[:, 0], xy[:, 1]))
        else:
            y = np.empty(0)
            order = np.argsort(xy[:, 0])
            x = xy[order, 0]

        def trace(values: np.ndarray, name: str, index: int) -> go.Heatmap | go.Scatter:
            values = np.asarray(values[order], dtype=np.float32)
            if field:
                return go.Heatmap(
                    x=x,
                    y=y,
                    z=values.reshape(len(y), len(x)),
                    coloraxis=('coloraxis', 'coloraxis3', 'coloraxis2')[index],
                    name=name,
                    hoverongaps=False,
                    hovertemplate='x %{x:.3f} · y %{y:.3f}<br>%{z:.5g}<extra>%{fullData.name}</extra>',
                )
            return go.Scatter(
                x=x,
                y=values,
                mode='lines',
                name=name,
                showlegend=False,
                line={'color': (BLUE, GREEN, RED)[index], 'width': 2.3},
                hovertemplate='x %{x:.3f}<br>%{y:.5g}<extra>%{fullData.name}</extra>',
            )

        return trace

    def _add_scalar_diagnostics(self, fig: go.Figure) -> None:
        for index, region in enumerate(self.checkpoints[0].get('evaluation_mse', {})):
            label = cosinus.REGION_LABELS[region]
            shade = COLORS[index % len(COLORS)]
            fig.add_trace(
                go.Scatter(
                    x=[frame['step'] for frame in self.checkpoints],
                    y=[
                        max(
                            frame.get('evaluation_mse', {})[region],
                            np.finfo(np.float32).tiny,
                        )
                        for frame in self.checkpoints
                    ],
                    mode='lines',
                    name=cosinus.REGION_TITLES[region],
                    line={'color': shade, 'width': 2},
                    hovertemplate=f'Step %{{x:,.0f}}<br>MSE %{{y:.3e}}<extra>{label}</extra>',
                ),
                row=3,
                col=1,
            )
        fig.update_yaxes(title_text='MSE', type='log', row=3, col=1)
        fig.update_xaxes(
            title_text='Training step',
            range=[0, max(1, self.loss_history[-1][0])],
            row=3,
            col=1,
        )

    def _configure_scalar_axes(
        self,
        fig: go.Figure,
        coordinates: Array,
        solution: tuple[float, float],
        error: tuple[float, float],
    ) -> None:
        field = coordinates.shape[1] == 2
        x = coordinates[:, 0]
        y = coordinates[:, 1] if field else np.empty(0)
        for col in (1, 2, 3):
            fig.update_xaxes(
                title_text='x', range=[float(x.min()), float(x.max())], row=1, col=col
            )
            if field:
                fig.update_yaxes(
                    title_text='y',
                    range=[float(y.min()), float(y.max())],
                    scaleanchor='x' + (str(col) if col > 1 else ''),
                    scaleratio=1,
                    constrain='domain',
                    row=1,
                    col=col,
                )
                fig.update_xaxes(constrain='domain', row=1, col=col)
            else:
                limits = error if col == 3 else solution
                pad = 0.04 * (limits[1] - limits[0])
                fig.update_yaxes(
                    title_text='Error' if col == 3 else 'f(x)',
                    range=[limits[0] - pad, limits[1] + pad],
                    row=1,
                    col=col,
                )
        if not field:
            fig.update_yaxes(
                zeroline=True, zerolinecolor='#888888', zerolinewidth=1, row=1, col=3
            )
        if self.scenario == COSINUS_SCENARIO:
            cosinus_axes(fig, rows=(1,), cols=(1, 2, 3))

    def _add_collocation(self, fig: go.Figure, *, rug: bool = False) -> None:
        """The first field carries the shared points; all frames retain exact sets."""
        groups = self.checkpoints[0]['collocation']
        indices = add_overlay(fig, groups, rug=rug)
        if not indices:
            return
        # This figure has a custom queued playback handler. Keep controls in
        # that queue so hiding points cannot race a seek or a field selection.
        for button in cast(Any, fig.layout).updatemenus[-1].buttons:
            button.execute = False
        if any(frame['collocation'] != groups for frame in self.checkpoints):
            prototype = fig.data[indices[0]]
            for frame, recorded in zip(fig.frames, self.checkpoints):
                frame.traces = tuple(frame.traces) + tuple(indices)
                frame.data = tuple(frame.data) + tuple(
                    point_traces(
                        recorded['collocation'],
                        xaxis=prototype.xaxis,
                        yaxis=prototype.yaxis,
                        rug=rug,
                    )
                )

    def _scalar_layout(
        self, fig: go.Figure, *, field: bool, extrapolation: bool
    ) -> None:
        """The same comparison panels arranged in columns or stacked on a phone."""
        desktop = self._scalar_layout_changes(
            False, field=field, extrapolation=extrapolation
        )
        compact = self._scalar_layout_changes(
            True, field=field, extrapolation=extrapolation
        )
        # Plotly.relayout uses dotted keys; update_layout accepts nested objects.
        for key, value in desktop.items():
            if key.startswith(('sliders[', 'updatemenus[')):
                continue
            parts = key.split('.')
            if len(parts) == 2:
                cast(Any, fig.layout)[parts[0]][parts[1]] = value
            else:
                cast(Any, fig.layout)[key] = value
        fig.layout.meta = {
            'responsive_layout': {'desktop': desktop, 'compact': compact}
        }

    @staticmethod
    def _scalar_domains(
        compact: bool, extrapolation: bool
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        top: list[tuple[float, float]] = [(0.42, 1.0)] * 3
        if compact:
            # Reserve a fourth row for convergence and a fifth for diagnostics.
            top = (
                [(0.82, 1.0), (0.53, 0.71), (0.24, 0.42)]
                if not extrapolation
                else [(0.85, 1.0), (0.59, 0.74), (0.33, 0.48)]
            )
        elif extrapolation:
            top = [(0.59, 1.0)] * 3
        horizontal: list[tuple[float, float]] = (
            [(0.0, 0.90)] * 3
            if compact
            else [(0.0, 0.245), (0.355, 0.60), (0.71, 0.955)]
        )
        return top, horizontal

    def _scalar_layout_changes(
        self, compact: bool, *, field: bool, extrapolation: bool
    ) -> dict[str, Any]:
        top, horizontal = self._scalar_domains(compact, extrapolation)
        changes: dict[str, Any] = {
            'height': (1700 if extrapolation else 1450)
            if compact
            else (1100 if extrapolation else 900)
        }
        annotations: list[dict[str, Any]] = []
        for index, ((left, right), (bottom, upper)) in enumerate(
            zip(horizontal, top), 1
        ):
            suffix = str(index) if index > 1 else ''
            changes[f'xaxis{suffix}.domain'] = [left, right]
            changes[f'yaxis{suffix}.domain'] = [bottom, upper]
            annotations.append(
                dict(
                    text=('Prediction', 'Reference', 'Signed error')[index - 1],
                    x=(left + right) / 2,
                    y=upper,
                    xref='paper',
                    yref='paper',
                    xanchor='center',
                    yanchor='bottom',
                    yshift=14,
                    showarrow=False,
                    font={'size': 15},
                )
            )
            if field:
                axis = ('coloraxis', 'coloraxis3', 'coloraxis2')[index - 1]
                changes[f'{axis}.colorbar'] = colorbar((bottom, upper), x=right + 0.006)
        convergence = (0, 0.11) if compact else (0, 0.20)
        if extrapolation:
            convergence = (0.15, 0.23) if compact else (0.28, 0.42)
        changes['xaxis4.domain'] = [0, 0.955]
        changes['yaxis4.domain'] = list(convergence)
        annotations.append(
            dict(
                text='Convergence',
                x=0,
                y=convergence[1],
                xref='paper',
                yref='paper',
                xanchor='left',
                yanchor='bottom',
                yshift=10,
                showarrow=False,
                font={'size': 15},
            )
        )
        if extrapolation:
            self._diagnostic_layout(compact, changes, annotations)
        annotations.append(self._footnote(compact))
        changes.update(self._scalar_chrome(compact, extrapolation, annotations))
        return changes

    @staticmethod
    def _diagnostic_layout(
        compact: bool, changes: dict[str, Any], annotations: list[dict[str, Any]]
    ) -> None:
        changes['xaxis5.domain'] = [0, 0.955]
        changes['yaxis5.domain'] = [0, 0.065 if compact else 0.12]
        annotations.append(
            dict(
                text='Evaluation MSE · excluded from training',
                x=0,
                y=0.065 if compact else 0.12,
                xref='paper',
                yref='paper',
                xanchor='left',
                yanchor='bottom',
                yshift=10,
                showarrow=False,
                font={'size': 13 if compact else 15},
            )
        )

    def _scalar_chrome(
        self, compact: bool, extrapolation: bool, annotations: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return {
            'annotations': annotations,
            'title.font.size': 15 if compact else 22,
            'margin.l': 60 if compact else 80,
            'margin.r': 50 if compact else 70,
            **self._responsive_controls(compact),
            'legend': {
                'orientation': 'h',
                'x': 0,
                'y': (-0.055 if compact else -0.08) if extrapolation else 0,
                'yanchor': 'top',
                'yref': 'paper',
                'font': {'size': 10},
                'tracegroupgap': 0,
                'entrywidth': 145,
                'entrywidthmode': 'pixels',
            },
        }

    def _responsive_controls(self, compact: bool) -> dict[str, int | float]:
        cosine = self.scenario == COSINUS_SCENARIO
        top = 185 if cosine else 160
        if self.flow and self.requires_reference:
            top += 60
        if not compact:
            return {
                'margin.b': 220,
                'margin.t': top,
                'title.y': 0.98,
                'sliders[0].x': 0.23,
                'sliders[0].len': 0.77,
                'sliders[0].pad.t': 120,
                'updatemenus[0].pad.t': 132,
            }
        return {
            'margin.b': 370 if cosine else 280,
            'margin.t': top + 40,
            'title.y': 0.96,
            'sliders[0].x': 0,
            'sliders[0].len': 1,
            'sliders[0].pad.t': 280 if cosine else 185,
            'updatemenus[0].pad.t': 230 if cosine else 132,
        }

    def _title(self, frame: Checkpoint) -> str:
        title = f'Training step {frame["step"]:,} | Loss {frame["loss"]:.2e}'
        if self.reference is not None:
            metrics = error_metrics(frame['prediction'], self.reference)
            title += (
                f'<br><span style="font-size:12px">Evaluation grid: relative L₂ = {metrics["relative_l2"]:.2e}'
                f'<br>Maximum absolute error = {metrics["max_abs"]:.2e}</span>'
            )
        elif self.flow:
            title += '<br><span style="font-size:12px">Fixed field scales across all training steps</span>'
        if self.scenario == COSINUS_SCENARIO:
            title += f'<br><span style="font-size:11px">Order {self.cosinus_order} · shaded: training [−π, π]</span>'
        title += (
            f'<br><span style="font-size:10px">{summary(frame["collocation"])}</span>'
        )
        return title

    @staticmethod
    def _cursor(frame: Checkpoint) -> go.Scatter:
        return go.Scatter(
            x=[frame['step']],
            y=[max(frame['loss'], np.finfo(np.float32).tiny)],
            mode='markers',
            marker={'color': BLUE, 'size': 7},
            name='Selected step',
            showlegend=False,
            hovertemplate='Step %{x:,.0f}<br>Loss %{y:.3e}<extra></extra>',
        )

    def _add_convergence(self, fig: go.Figure, *, row: int) -> None:
        if not self.loss_history:
            raise ValueError('Capture at least one checkpoint before exporting.')
        steps, losses = np.asarray(self.loss_history, dtype=float).T
        losses = np.maximum(losses, np.finfo(np.float32).tiny)
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=losses,
                mode='lines',
                name='Loss',
                line={'color': BLUE, 'width': 1.7},
                showlegend=False,
                hovertemplate='Step %{x:,.0f}<br>Loss %{y:.3e}<extra></extra>',
            ),
            row=row,
            col=1,
        )
        fig.add_trace(self._cursor(self.checkpoints[0]), row=row, col=1)
        lo, hi = np.log10(losses.min()), np.log10(losses.max())
        fig.update_xaxes(
            title_text='Training step',
            range=[0, max(1, float(steps[-1]))],
            row=row,
            col=1,
        )
        fig.update_yaxes(
            title_text='Objective',
            type='log',
            range=[lo - 0.2, max(hi, lo + 1) + 0.2],
            row=row,
            col=1,
        )

    def _footnote(self, compact: bool = False) -> dict[str, Any]:
        recorded = self.checkpoints and self.checkpoints[0]['collocation'] is not None
        return dict(
            text=(
                (
                    'First panel: navy PDE points · orange boundaries'
                    + ('<br>' if compact else ' · ')
                    + 'red anchors · purple flux quadrature.'
                    if recorded
                    else 'Training coordinates were not recorded for these checkpoints.'
                )
                + ('<br>' if compact else ' ')
                + (
                    '1D marks show x locations only. '
                    if (
                        recorded
                        and self.coordinates is not None
                        and self.coordinates.shape[1] == 1
                    )
                    else ''
                )
                + 'Field rendering uses a separate grid.'
            ),
            x=0.5,
            y=0,
            xref='paper',
            yref='paper',
            xanchor='center',
            yanchor='top',
            yshift=-(180 if compact else 105)
            if self.scenario == COSINUS_SCENARIO
            else -88,
            showarrow=False,
            font={'size': 9 if compact else 10, 'color': '#666666'},
        )

    @staticmethod
    def _playback_controls() -> tuple[dict[str, Any], list[dict[str, Any]]]:
        immediate = {
            'mode': 'immediate',
            'frame': {'duration': 0, 'redraw': True},
            'transition': {'duration': 0},
        }
        controls = [
            {
                'type': 'buttons',
                'direction': 'left',
                'showactive': False,
                'x': 0,
                'y': -0.13,
                'xanchor': 'left',
                'yanchor': 'top',
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'execute': False,
                        'args': [
                            None,
                            {
                                'fromcurrent': True,
                                'mode': 'immediate',
                                'frame': {'duration': 180, 'redraw': True},
                                'transition': {'duration': 0},
                            },
                        ],
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'execute': False,
                        'args': [[None], immediate],
                    },
                ],
            }
        ]
        return immediate, controls

    def _finish_figure(
        self, fig: go.Figure, controls: list[dict[str, Any]], immediate: dict[str, Any]
    ) -> go.Figure:
        scientific_style(fig)
        controls[0].update(y=0, pad={'t': 132}, yanchor='top')
        fig.update_layout(
            height=1200 if self.flow else 900,
            autosize=True,
            title={
                'text': self._title(self.checkpoints[0]),
                'y': 0.98,
                'yanchor': 'top',
            },
            margin={
                'l': 80,
                'r': 85,
                't': self._responsive_controls(False)['margin.t'],
                'b': 220,
            },
            hovermode='closest',
            uirevision=self.scenario,
            updatemenus=controls,
            sliders=[
                {
                    'active': 0,
                    'x': 0.23,
                    'len': 0.77,
                    'y': 0,
                    'yanchor': 'top',
                    'pad': {'t': 120},
                    'currentvalue': {'prefix': 'Training step: ', 'font': {'size': 12}},
                    'steps': [
                        {
                            'label': str(frame['step']),
                            'method': 'animate',
                            'execute': False,
                            'args': [[str(frame['step'])], immediate],
                        }
                        for frame in self.checkpoints
                    ],
                }
            ],
        )
        fig.add_annotation(self._footnote())
        return fig

    def _add_flow_boundaries(self, fig: go.Figure) -> None:
        if self.boundary_edges is not None:
            segments = np.full((len(self.boundary_edges), 3, 2), np.nan)
            segments[:, :2] = self.boundary_edges
            segments = segments.reshape(-1, 2)
            for row in range(1, 4):
                fig.add_trace(
                    go.Scatter(
                        x=segments[:, 0],
                        y=segments[:, 1],
                        mode='lines',
                        line={'color': '#333333', 'width': 1},
                        hoverinfo='skip',
                        showlegend=False,
                    ),
                    row=row,
                    col=1,
                )

    def _add_flow_fields(
        self, fig: go.Figure, coordinates: Array, triangles: Array
    ) -> tuple[MeshRaster, Array]:
        xy = coordinates
        raster = MeshRaster(xy, triangles)
        visible = np.unique(triangles)
        for index, name in enumerate(('u', 'v', 'p')):
            fields = [frame['prediction'][index, visible] for frame in self.checkpoints]
            if self.reference is not None:
                fields.append(self.reference[index, visible])
            if index < 2:
                fields.append(np.array([-1.0, 1.0]))
            lo, hi = self.color_limits.get(
                name, value_range(*fields, symmetric=index < 2)
            )
            suffix = str(index + 1) if index else ''
            domain = cast(Any, fig.layout)[f'yaxis{suffix}'].domain
            fig.add_trace(
                go.Heatmap(
                    x=raster.x,
                    y=raster.y,
                    z=raster.sample(self.checkpoints[0]['prediction'][index]),
                    zmin=lo,
                    zmax=hi,
                    zauto=False,
                    colorscale='RdBu_r' if index < 2 else 'Viridis',
                    showscale=True,
                    colorbar=colorbar(domain, title=name),
                    name=name,
                    hoverongaps=False,
                    hovertemplate='x %{x:.3f} · y %{y:.3f}<br>%{z:.5g}<extra>%{fullData.name}</extra>',
                ),
                row=index + 1,
                col=1,
            )
            fig.update_xaxes(
                title_text='x',
                range=[float(xy[:, 0].min()), float(xy[:, 0].max())],
                constrain='domain',
                row=index + 1,
                col=1,
            )
            fig.update_yaxes(
                title_text='y',
                range=[float(xy[:, 1].min()), float(xy[:, 1].max())],
                scaleanchor='x' + suffix,
                scaleratio=1,
                constrain='domain',
                row=index + 1,
                col=1,
            )
        return raster, visible

    def _add_flow_reference(
        self, fig: go.Figure, raster: MeshRaster, visible: Array
    ) -> tuple[list[int], list[int]]:
        reference_indices, error_indices = [], []
        if self.reference is not None:
            for index, name in enumerate(('u', 'v', 'p')):
                reference = go.Heatmap(fig.data[index].to_plotly_json())
                reference.update(
                    z=raster.sample(self.reference[index]),
                    name=f'{name} reference',
                    visible=False,
                )
                reference_indices.append(len(fig.data))
                fig.add_trace(reference, row=index + 1, col=1)
                differences = [
                    frame['prediction'][index] - self.reference[index]
                    for frame in self.checkpoints
                ]
                lo, hi = self.color_limits.get(
                    f'{name}_error',
                    value_range(
                        *(field[visible] for field in differences), symmetric=True
                    ),
                )
                error = go.Heatmap(reference.to_plotly_json())
                error.update(
                    z=raster.sample(differences[0]),
                    name=f'{name} signed error',
                    zmin=lo,
                    zmax=hi,
                    colorscale='RdBu_r',
                    colorbar_title_text=f'{name} error',
                )
                error_indices.append(len(fig.data))
                fig.add_trace(error, row=index + 1, col=1)
        return reference_indices, error_indices

    def _flow_frames(
        self, raster: MeshRaster, error_indices: list[int]
    ) -> list[go.Frame]:
        return [
            go.Frame(
                name=str(frame['step']),
                traces=[0, 1, 2, 4, *error_indices],
                data=[
                    go.Heatmap(z=raster.sample(values))
                    for values in frame['prediction']
                ]
                + [self._cursor(frame)]
                + (
                    [
                        go.Heatmap(z=raster.sample(values))
                        for values in frame['prediction'] - self.reference
                    ]
                    if self.reference is not None
                    else []
                ),
                layout={'title': {'text': self._title(frame)}},
            )
            for frame in self.checkpoints
        ]

    def _add_field_controls(
        self,
        controls: list[dict[str, Any]],
        reference_indices: list[int],
        error_indices: list[int],
    ) -> None:
        if self.reference is not None:
            field_indices = {0, 1, 2, *reference_indices, *error_indices}
            controls.append(
                {
                    'type': 'dropdown',
                    'x': 1,
                    'y': 1,
                    'xanchor': 'right',
                    'yanchor': 'bottom',
                    'pad': {'b': 95},
                    'buttons': [
                        {
                            'label': label,
                            'method': 'update',
                            'execute': False,
                            'args': [
                                {
                                    'visible': [
                                        i in selected for i in sorted(field_indices)
                                    ]
                                },
                                {},
                                sorted(field_indices),
                            ],
                        }
                        for label, selected in [
                            ('Prediction', [0, 1, 2]),
                            ('Reference', reference_indices),
                            ('Signed error', error_indices),
                        ]
                    ],
                },
            )

    def _flow_layout(self, fig: go.Figure) -> None:
        for row in range(1, 4):
            fig.update_xaxes(showgrid=False, row=row, col=1)
            fig.update_yaxes(showgrid=False, row=row, col=1)
        # Flat axes retain physical proportions when the figure is resized.
        fig.layout.meta = {
            'responsive_layout': {
                'desktop': {
                    'height': 1200,
                    'title.font.size': 22,
                    'margin.l': 80,
                    'margin.r': 85,
                    **self._responsive_controls(False),
                    'annotations': [
                        a.to_plotly_json() for a in cast(Any, fig.layout).annotations
                    ],
                },
                'compact': {
                    'height': 1200,
                    'title.font.size': 15,
                    'margin.l': 60,
                    'margin.r': 70,
                    **self._responsive_controls(True),
                    'annotations': [
                        a.to_plotly_json()
                        for a in cast(Any, fig.layout).annotations[:-1]
                    ]
                    + [self._footnote(True)],
                },
            }
        }

    def _flow_figure(self) -> go.Figure:
        if self.coordinates is None or self.triangles is None:
            raise ValueError('Capture a complete flow checkpoint before exporting.')
        coordinates = self.coordinates
        triangles = self.triangles
        fig = make_subplots(
            rows=4,
            cols=1,
            row_heights=[0.26, 0.26, 0.26, 0.12],
            vertical_spacing=0.09,
            subplot_titles=[
                'Horizontal velocity u',
                'Vertical velocity v',
                self.pressure_label,
                'Convergence',
            ],
        )
        raster, visible = self._add_flow_fields(fig, coordinates, triangles)
        self._add_convergence(fig, row=4)
        self._add_flow_boundaries(fig)
        reference_indices, error_indices = self._add_flow_reference(
            fig, raster, visible
        )
        fig.frames = self._flow_frames(raster, error_indices)
        immediate, controls = self._playback_controls()
        self._add_field_controls(controls, reference_indices, error_indices)
        self._finish_figure(fig, controls, immediate)
        self._flow_layout(fig)
        self._add_collocation(fig)
        return fig

    def write_frames(self, folder: str | Path) -> None:
        """Render checkpoint PNGs through Plotly/Kaleido for optional GIF export."""
        from importlib.util import find_spec

        if find_spec('kaleido') is None:
            raise RuntimeError(
                'PNG/GIF export requires Kaleido: run uv sync --group export, and install Chrome.'
            )
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        fig = self.figure()
        snapshot = go.Figure(data=fig.data, layout=fig.layout)
        # Assign empty tuples directly: update_layout merges empty lists and can
        # otherwise leave playback controls embedded in static PNG/GIF frames.
        snapshot.layout.updatemenus = ()
        snapshot.layout.sliders = ()
        height = 1200 if self.flow or self.scenario == COSINUS_SCENARIO else 900
        snapshot.update_layout(
            updatemenus=[],
            sliders=[],
            margin_b=140,
            title_y=0.95,
            width=1600,
            height=height,
        )
        align_field_panels(snapshot, 1600, height)
        snapshot.layout.meta = None
        snapshots, paths = [], []
        for frame in fig.frames:
            for index, data in zip(frame.traces, frame.data):
                snapshot.data[index].update(data.to_plotly_json())
            snapshot.update_layout(frame.layout.to_plotly_json())
            snapshots.append(snapshot.to_plotly_json())
            paths.append(folder / f'epoch_{frame.name}.png')
        # Only clear files owned by a previous export, after constructing all
        # snapshots successfully. A later encoder failure leaves the PNGs here.
        for previous in folder.glob('epoch_*.png'):
            previous.unlink()
        pio.write_images(
            snapshots,
            paths,
            format='png',
            width=1600,
            height=height,
        )

    def write_html(self, path: str | Path) -> None:
        """Write a portable HTML file and a shared, local Plotly runtime."""
        path = Path(path)
        figure = self.figure()
        path.parent.mkdir(parents=True, exist_ok=True)
        # Always refresh the shared runtime so regenerating after an upgrade
        # cannot pair new figure data with an old Plotly.js bundle.
        (path.parent / 'plotly.min.js').write_text(get_plotlyjs(), encoding='utf-8')
        body = pio.to_html(
            figure,
            full_html=False,
            include_plotlyjs=cast(Any, 'directory'),
            auto_play=False,
            div_id=f'learnpdes-{self.scenario.replace(" ", "-")}',
            config={
                'responsive': False,
                'displaylogo': False,
                'scrollZoom': False,
                'modeBarButtons': [['toImage', 'zoom2d', 'pan2d', 'resetScale2d']],
            },
            post_script=Path(__file__)
            .with_name('training_playback.js')
            .read_text(encoding='utf-8'),
        )
        label = escape(self.scenario.capitalize())
        path.write_text(
            '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{label} · Interactive PINN training</title>'
            '<meta name="robots" content="noindex">'
            '<style>body{margin:0;background:#fff;font-family:system-ui,sans-serif}'
            '.hint{margin:8px 16px;color:#475569;font-size:13px;line-height:1.5}'
            '.js-plotly-plot .plotly .modebar{top:0;right:8px}'
            '</style></head><body>'
            '<p class="hint">Hover for values. Drag to zoom; double-click to reset. '
            'Use Play or the training-step slider to explore the run.</p>'
            f'{body}<noscript>Enable JavaScript to explore this training run.</noscript>'
            '</body></html>\n',
            encoding='utf-8',
        )
