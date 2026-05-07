# Vibrant, accessible cluster colors (high contrast on white)
CLUSTER_COLORS = [
    '#e63946', '#2a9d8f', '#264653', '#e9c46a', '#7b2cbf',
    '#0096c7', '#f4722b', '#606c38',
]

BG_COLOR = '#ffffff'
PLOT_BG = '#f8f9fa'
GRID_COLOR = '#dee2e6'
TEXT_COLOR = '#212529'
ACCENT_COLOR = '#2a9d8f'

PLOTLY_LAYOUT = dict(
    paper_bgcolor=BG_COLOR,
    plot_bgcolor=PLOT_BG,
    font=dict(color=TEXT_COLOR, family="Inter, system-ui, sans-serif", size=13),
    title_font=dict(size=18, color=TEXT_COLOR),
    margin=dict(l=50, r=30, t=65, b=45),
    legend=dict(
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='#dee2e6',
        borderwidth=1,
        font=dict(size=11, color=TEXT_COLOR),
    ),
)

PLOTLY_LAYOUT_3D = dict(
    paper_bgcolor=BG_COLOR,
    font=dict(color=TEXT_COLOR, family="Inter, system-ui, sans-serif", size=13),
    title_font=dict(size=18, color=TEXT_COLOR),
    margin=dict(l=20, r=20, t=65, b=20),
    legend=dict(
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='#dee2e6',
        borderwidth=1,
        font=dict(size=11, color=TEXT_COLOR),
    ),
    scene=dict(
        bgcolor='#f0f2f5',
        xaxis=dict(backgroundcolor='#f0f2f5', gridcolor='#d0d4d9',
                    showbackground=True, zerolinecolor='#adb5bd'),
        yaxis=dict(backgroundcolor='#f0f2f5', gridcolor='#d0d4d9',
                    showbackground=True, zerolinecolor='#adb5bd'),
        zaxis=dict(backgroundcolor='#f0f2f5', gridcolor='#d0d4d9',
                    showbackground=True, zerolinecolor='#adb5bd'),
    ),
)


def hex_to_rgb(hex_color):
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def blend_colors(responsibilities, colors=None):
    """Blend cluster colors by responsibility weights for soft assignment."""
    if colors is None:
        colors = CLUSTER_COLORS
    N, K = responsibilities.shape
    rgb_colors = [hex_to_rgb(colors[k % len(colors)]) for k in range(K)]

    result = []
    for n in range(N):
        r, g, b = 0.0, 0.0, 0.0
        for k in range(K):
            r += responsibilities[n, k] * rgb_colors[k][0]
            g += responsibilities[n, k] * rgb_colors[k][1]
            b += responsibilities[n, k] * rgb_colors[k][2]
        result.append(f'rgb({int(r)},{int(g)},{int(b)})')
    return result


CUSTOM_CSS = """
<style>
    /* Light, clean sidebar */
    [data-testid="stSidebar"] {
        background-color: #f0f2f5;
        border-right: 1px solid #dee2e6;
    }
    [data-testid="stSidebar"] .stMarkdown h2 {
        color: #264653;
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }

    /* Main area */
    .stApp {
        background-color: #ffffff;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 12px 16px;
    }
    [data-testid="stMetricLabel"] {
        color: #6c757d;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="stMetricValue"] {
        color: #212529;
        font-weight: 700;
    }

    /* Buttons */
    .stButton > button[kind="primary"] {
        background-color: #2a9d8f;
        border: none;
        font-weight: 600;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #21867a;
    }

    /* Clean up footer / menu */
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}

    /* Plotly chart container — subtle border */
    [data-testid="stPlotlyChart"] {
        border: 1px solid #e9ecef;
        border-radius: 8px;
        overflow: hidden;
    }

    /* Radio buttons horizontal */
    .stRadio > div {
        gap: 0.5rem;
    }
</style>
"""
