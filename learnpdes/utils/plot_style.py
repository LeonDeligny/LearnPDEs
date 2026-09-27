"""Publication styling shared by every Plotly view."""

BLUE = '#315b87'
GREEN = '#27816b'
RED = '#ad493b'
COLORS = [BLUE, GREEN, RED, '#80649a', '#b18335', '#438b91']
FONT = 'DejaVu Sans, Arial, sans-serif'


def scientific_style(fig):
    """Use the typography, restrained colours and axes of the static figures."""
    fig.update_layout(
        template='none',
        paper_bgcolor='white',
        plot_bgcolor='white',
        font={'family': FONT, 'size': 13, 'color': '#222222'},
        colorway=COLORS,
        title={'x': 0.5, 'xanchor': 'center', 'font': {'size': 22}},
        hoverlabel={'font': {'family': FONT, 'size': 12}},
    )
    fig.update_xaxes(
        showline=True,
        linecolor='#222222',
        linewidth=1,
        mirror=False,
        ticks='outside',
        tickwidth=1,
        ticklen=4,
        tickfont={'size': 11},
        showgrid=True,
        gridcolor='#e5e5e5',
        gridwidth=0.6,
        zeroline=False,
        title_font_size=13,
        title_standoff=12,
        automargin=True,
    )
    fig.update_yaxes(
        showline=True,
        linecolor='#222222',
        linewidth=1,
        mirror=False,
        ticks='outside',
        tickwidth=1,
        ticklen=4,
        tickfont={'size': 11},
        showgrid=True,
        gridcolor='#e5e5e5',
        gridwidth=0.6,
        zeroline=False,
        title_font_size=13,
        title_standoff=10,
        automargin=True,
    )
    fig.update_annotations(font={'family': FONT, 'size': 15, 'color': '#222222'})
    for axis in fig.select_yaxes():
        if axis.type == 'log':
            axis.update(
                dtick=1 if axis.range and axis.range[1] - axis.range[0] >= 1 else None,
                exponentformat='power',
                showexponent='all',
            )
    return fig


def colorbar(domain, *, x=1.015, title=''):
    return {
        'title': {'text': title, 'side': 'right'},
        'x': x,
        'xanchor': 'left',
        'y': sum(domain) / 2,
        'len': domain[1] - domain[0],
        'thickness': 11,
        'xpad': 6,
        'ypad': 0,
        'outlinewidth': 0.6,
        'outlinecolor': '#444444',
        'tickfont': {'size': 10},
        'ticks': 'outside',
        'ticklen': 3,
        'tickwidth': 0.6,
        'exponentformat': 'power',
    }


def align_field_panels(fig, width, height):
    """Place bars beside the actual equal-aspect fields in a static export."""
    margin = fig.layout.margin
    available_width = width - margin.l - margin.r
    available_height = height - margin.t - margin.b
    for index, trace in enumerate(fig.data[:3]):
        if trace.type != 'heatmap':
            continue
        xaxis = fig.layout['xaxis' + trace.xaxis[1:]]
        yaxis = fig.layout['yaxis' + trace.yaxis[1:]]
        left, right = xaxis.domain
        bottom, top = yaxis.domain
        aspect = (xaxis.range[1] - xaxis.range[0]) / (yaxis.range[1] - yaxis.range[0])
        span_x = min(
            right - left, (top - bottom) * available_height * aspect / available_width
        )
        span_y = min(
            top - bottom, (right - left) * available_width / aspect / available_height
        )
        xaxis.domain = [(left + right - span_x) / 2, (left + right + span_x) / 2]
        yaxis.domain = [(bottom + top - span_y) / 2, (bottom + top + span_y) / 2]
        bar = (
            fig.layout[trace.coloraxis].colorbar if trace.coloraxis else trace.colorbar
        )
        bar.update(x=xaxis.domain[1] + 0.01, y=sum(yaxis.domain) / 2, len=span_y)
        if not trace.coloraxis:
            for other in fig.data[3:]:
                if other.type == 'heatmap' and other.xaxis == trace.xaxis:
                    other.colorbar.update(x=bar.x, y=bar.y, len=bar.len)
        fig.layout.annotations[index].update(
            y=yaxis.domain[1], x=sum(xaxis.domain) / 2, yshift=14
        )
