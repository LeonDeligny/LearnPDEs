"""Plotly views of scalar and flow PINN training, with optional image export."""

from html import escape
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

from learnpdes import (
    cosinus,
    COSINUS_SCENARIO,
    CYLINDER_SCENARIO,
    EXPONENTIAL_SCENARIO,
    LAPLACE_SCENARIO,
    KOVASZNAY_SCENARIO,
    POISEUILLE_SCENARIO,
    POTENTIAL_FLOW_SCENARIO,
    SOLENOIDAL_FLOW_SCENARIO,
)
from learnpdes.utils.plot import error_metrics, value_range
from learnpdes.utils.mesh_raster import MeshRaster
from learnpdes.utils.plot_style import (
    BLUE,
    COLORS,
    GREEN,
    RED,
    align_field_panels,
    colorbar,
    scientific_style,
)
from learnpdes.utils.visualization import rectangular_triangles


def cosinus_axes(fig, rows=(1, 2), cols=(1,)):
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
        fig.add_vrect(
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
            fig.add_vline(
                x=bound, line_dash='dot', line_color='#64748b', row=row, col=col
            )


class InteractivePlot:
    """Capture Trainer checkpoints and build a figure after training finishes.

    Scalar fields use lines or heatmaps. Flow fields use the supplied fluid
    triangles, retaining the airfoil hole and original boundary geometry.
    """

    def __init__(self, scenario: str, *, color_limits=None, cosinus_order=2):
        if scenario not in (
            CYLINDER_SCENARIO,
            EXPONENTIAL_SCENARIO,
            COSINUS_SCENARIO,
            LAPLACE_SCENARIO,
            KOVASZNAY_SCENARIO,
            POISEUILLE_SCENARIO,
            POTENTIAL_FLOW_SCENARIO,
            SOLENOIDAL_FLOW_SCENARIO,
        ):
            raise ValueError(f'Unsupported interactive scenario: {scenario}')
        self.scenario = scenario
        cosinus.validate_order(cosinus_order)
        self.cosinus_order = cosinus_order
        self.flow = scenario in (
            CYLINDER_SCENARIO,
            KOVASZNAY_SCENARIO,
            POTENTIAL_FLOW_SCENARIO,
            SOLENOIDAL_FLOW_SCENARIO,
            POISEUILLE_SCENARIO,
        )
        self.color_limits = dict(color_limits or {})
        for limits in self.color_limits.values():
            if (
                len(limits) != 2
                or not np.isfinite(limits).all()
                or limits[0] >= limits[1]
            ):
                raise ValueError('Colour limits must be finite increasing pairs.')
        self.reset()

    def reset(self):
        self.checkpoints = []
        self.coordinates = None
        self.reference = None
        self.loss_history = []
        self.triangles = None
        self.boundary_edges = None
        self.pressure_label = 'Pressure p (model units)'

    def __call__(
        self,
        folder,
        *,
        epoch,
        inputs,
        f,
        loss,
        analytical,
        loss_history,
        triangles=None,
        boundary_edges=None,
        geometry_mask=None,
        pressure_label='Pressure p (model units)',
        **kwargs,
    ):
        coordinates = np.asarray(inputs).reshape(len(inputs), -1)
        prediction = (
            np.asarray(f).reshape(3, -1) if self.flow else np.asarray(f).ravel()
        )
        if analytical is None and (
            not self.flow or self.scenario in (POISEUILLE_SCENARIO, KOVASZNAY_SCENARIO)
        ):
            raise ValueError('This scenario requires a reference solution.')
        if self.coordinates is None:
            self.coordinates = coordinates.copy()
            if self.flow:
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
                self.triangles = cells.copy()
                self.boundary_edges = (
                    None
                    if boundary_edges is None
                    else np.asarray(boundary_edges).copy()
                )
                self.pressure_label = pressure_label
                if self.scenario in (POISEUILLE_SCENARIO, KOVASZNAY_SCENARIO):
                    self.reference = np.asarray(analytical(*coordinates.T))
                    if self.reference.shape != (3, len(coordinates)):
                        raise ValueError('Expected reference fields (u, v, p).')
            else:
                self.reference = np.asarray(analytical(*coordinates.T)).ravel()
        elif not np.array_equal(coordinates, self.coordinates):
            raise ValueError('All checkpoints must use the same visualization grid.')
        expected = (3, len(coordinates)) if self.flow else self.reference.shape
        if prediction.shape != expected:
            raise ValueError('Prediction and reference shapes must match.')
        if not np.isfinite(prediction).all() or not np.isfinite(loss):
            raise ValueError('Cannot plot non-finite predictions or loss.')
        if self.checkpoints and epoch <= self.checkpoints[-1]['step']:
            raise ValueError('Checkpoint steps must be strictly increasing.')
        frame = {'step': epoch, 'loss': float(loss), 'prediction': prediction.copy()}
        if self.scenario == COSINUS_SCENARIO:
            frame['evaluation_mse'] = cosinus.region_mse(coordinates, prediction)
        self.checkpoints.append(frame)
        self.loss_history = list(loss_history)

    def figure(self) -> go.Figure:
        if not self.checkpoints:
            raise ValueError('Capture at least one checkpoint before exporting.')
        if self.flow:
            return self._flow_figure()
        field = self.scenario == LAPLACE_SCENARIO
        predictions = [frame['prediction'] for frame in self.checkpoints]
        solution = self.color_limits.get(
            'solution', value_range(self.reference, *predictions)
        )
        errors = [prediction - self.reference for prediction in predictions]
        error = self.color_limits.get('error', value_range(*errors, symmetric=True))
        xy = self.coordinates
        if field:
            x, y = np.unique(xy[:, 0]), np.unique(xy[:, 1])
            if len(x) * len(y) != len(xy) or len(np.unique(xy, axis=0)) != len(xy):
                raise ValueError('Expected a complete rectangular visualization grid.')
            order = np.lexsort((xy[:, 0], xy[:, 1]))
        else:
            order = np.argsort(xy[:, 0])
            x = xy[order, 0]

        def trace(values, name, index):
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
        for index, (values, name) in enumerate(
            zip((predictions[0], self.reference, errors[0]), titles)
        ):
            fig.add_trace(trace(values, name, index), row=1, col=index + 1)
        self._add_convergence(fig, row=2)
        if extrapolation:
            for index, region in enumerate(self.checkpoints[0]['evaluation_mse']):
                label = cosinus.REGION_LABELS[region]
                shade = COLORS[index % len(COLORS)]
                fig.add_trace(
                    go.Scatter(
                        x=[frame['step'] for frame in self.checkpoints],
                        y=[
                            max(
                                frame['evaluation_mse'][region],
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
        fig.frames = [
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
        immediate, controls = self._playback_controls()
        self._finish_figure(fig, controls, immediate)
        if field:
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
        self._scalar_layout(fig, field=field, extrapolation=extrapolation)
        return fig

    def _scalar_layout(self, fig, *, field, extrapolation):
        """The same comparison panels arranged in columns or stacked on a phone."""

        def layout(compact):
            top = [(0.42, 1)] * 3
            if compact:
                # Reserve a fourth row for convergence and a fifth for diagnostics.
                top = (
                    [(0.82, 1), (0.53, 0.71), (0.24, 0.42)]
                    if not extrapolation
                    else [(0.85, 1), (0.59, 0.74), (0.33, 0.48)]
                )
            elif extrapolation:
                top = [(0.59, 1)] * 3
            horizontal = (
                [(0, 0.90)] * 3
                if compact
                else [(0, 0.245), (0.355, 0.60), (0.71, 0.955)]
            )
            changes = {
                'height': (1700 if extrapolation else 1450)
                if compact
                else (1100 if extrapolation else 900)
            }
            annotations = []
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
                    changes[f'{axis}.colorbar'] = colorbar(
                        (bottom, upper), x=right + 0.006
                    )
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
            annotations.append(self._footnote(compact))
            changes.update(
                {
                    'annotations': annotations,
                    'title.font.size': 15 if compact else 22,
                    'margin.l': 60 if compact else 80,
                    'margin.r': 50 if compact else 70,
                    **self._responsive_controls(compact),
                    'legend': {
                        'orientation': 'h',
                        'x': 0,
                        'y': (-0.055 if compact else -0.09) if extrapolation else 0,
                        'yanchor': 'top',
                        'yref': 'paper',
                        'font': {'size': 10},
                        'tracegroupgap': 0,
                    },
                }
            )
            return changes

        desktop, compact = layout(False), layout(True)
        # Plotly.relayout uses dotted keys; update_layout accepts nested objects.
        for key, value in desktop.items():
            if key.startswith('sliders['):
                continue
            parts = key.split('.')
            if len(parts) == 2:
                fig.layout[parts[0]][parts[1]] = value
            else:
                fig.layout[key] = value
        fig.layout.meta = {
            'responsive_layout': {'desktop': desktop, 'compact': compact}
        }

    def _responsive_controls(self, compact):
        return {
            'margin.b': 280 if compact else 220,
            'margin.t': (160 if self.scenario == COSINUS_SCENARIO else 135)
            + (40 if compact else 0),
            'title.y': 0.96 if compact else 0.98,
            'sliders[0].x': 0 if compact else 0.23,
            'sliders[0].len': 1 if compact else 0.77,
            'sliders[0].pad.t': 185 if compact else 120,
        }

    def _title(self, frame):
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
        return title

    @staticmethod
    def _cursor(frame):
        return go.Scatter(
            x=[frame['step']],
            y=[max(frame['loss'], np.finfo(np.float32).tiny)],
            mode='markers',
            marker={'color': BLUE, 'size': 7},
            name='Selected step',
            showlegend=False,
            hovertemplate='Step %{x:,.0f}<br>Loss %{y:.3e}<extra></extra>',
        )

    def _add_convergence(self, fig, *, row):
        steps, losses = np.asarray(self.loss_history).T
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

    @staticmethod
    def _footnote(compact=False):
        return dict(
            text=(
                'Visualization samples are separate from training samples.'
                + ('<br>' if compact else ' ')
                + 'Finer rendering does not establish accuracy.'
            ),
            x=0.5,
            y=0,
            xref='paper',
            yref='paper',
            xanchor='center',
            yanchor='top',
            yshift=-88,
            showarrow=False,
            font={'size': 9 if compact else 10, 'color': '#666666'},
        )

    @staticmethod
    def _playback_controls():
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

    def _finish_figure(self, fig, controls, immediate):
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
                't': 160 if self.scenario == COSINUS_SCENARIO else 135,
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

    def _flow_figure(self):
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
        xy = self.coordinates
        raster = MeshRaster(xy, self.triangles)
        visible = np.unique(self.triangles)
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
            domain = fig.layout[f'yaxis{suffix}'].domain
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
        self._add_convergence(fig, row=4)
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
        fig.frames = [
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
        immediate, controls = self._playback_controls()
        if self.reference is not None:
            field_indices = {0, 1, 2, *reference_indices, *error_indices}
            controls.append(
                {
                    'type': 'dropdown',
                    'x': 1,
                    'y': 1.06,
                    'xanchor': 'right',
                    'yanchor': 'bottom',
                    'buttons': [
                        {
                            'label': label,
                            'method': 'update',
                            'execute': False,
                            'args': [
                                {
                                    'visible': [
                                        i not in field_indices or i in selected
                                        for i in range(len(fig.data))
                                    ]
                                }
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
        self._finish_figure(fig, controls, immediate)
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
                    'annotations': [a.to_plotly_json() for a in fig.layout.annotations],
                },
                'compact': {
                    'height': 1200,
                    'title.font.size': 15,
                    'margin.l': 60,
                    'margin.r': 70,
                    **self._responsive_controls(True),
                    'annotations': [
                        a.to_plotly_json() for a in fig.layout.annotations[:-1]
                    ]
                    + [self._footnote(True)],
                },
            }
        }
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
            include_plotlyjs='directory',
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
