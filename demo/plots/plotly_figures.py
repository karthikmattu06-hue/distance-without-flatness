import numpy as np
import plotly.graph_objects as go
from demo.theme import BG_COLOR, PLOT_BG, GRID_COLOR, TEXT_COLOR, ACCENT_COLOR, PLOTLY_LAYOUT


def convergence_chart(history, ylabel="Log-Likelihood", title="Convergence"):
    fig = go.Figure()
    iters = list(range(1, len(history) + 1))

    fig.add_trace(go.Scatter(
        x=iters, y=history, mode='lines+markers',
        line=dict(color=ACCENT_COLOR, width=3),
        marker=dict(size=7, color=ACCENT_COLOR,
                    line=dict(color='white', width=1.5)),
        fill='tozeroy',
        fillcolor='rgba(42,157,143,0.08)',
        name=ylabel,
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=title,
        xaxis=dict(title="Iteration", gridcolor=GRID_COLOR, linecolor='#adb5bd'),
        yaxis=dict(title=ylabel, gridcolor=GRID_COLOR, linecolor='#adb5bd'),
        height=280,
        showlegend=False,
    )
    return fig


def sphere_wireframe(opacity=0.1):
    """Return Plotly traces for a translucent unit sphere."""
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    return go.Surface(
        x=x, y=y, z=z,
        opacity=opacity,
        colorscale=[[0, '#dbe4ee'], [1, '#c5d3e0']],
        showscale=False,
        hoverinfo='skip',
    )


def torus_surface(R=3, r=1, opacity=0.15):
    """Return Plotly Surface trace for a torus."""
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, 2 * np.pi, 40)
    u, v = np.meshgrid(u, v)
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)

    return go.Surface(
        x=x, y=y, z=z,
        opacity=opacity,
        colorscale=[[0, '#dbe4ee'], [1, '#c5d3e0']],
        showscale=False,
        hoverinfo='skip',
    )


def disk_boundary(n_pts=200):
    """Return Plotly trace for the Poincare disk unit circle."""
    theta = np.linspace(0, 2 * np.pi, n_pts)
    return go.Scatter(
        x=np.cos(theta), y=np.sin(theta),
        mode='lines',
        line=dict(color='#495057', width=2.5),
        hoverinfo='skip',
        showlegend=False,
    )
