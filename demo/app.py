"""
Clustering in Curved Spaces — Interactive Demo
================================================
Run with:  streamlit run demo/app.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np

from demo.theme import CUSTOM_CSS
from demo.geometries import get_registry

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Clustering in Curved Spaces",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Geometry")
    registry = get_registry()
    geo_name = st.selectbox(
        "Choose a space", list(registry.keys()), label_visibility="collapsed")
    geo = registry[geo_name]

    st.markdown("## Algorithm")
    algo = st.radio(
        "algo", ["K-Means", "EM", "Side-by-Side"],
        horizontal=True, label_visibility="collapsed")

    # Distance metric selector (for K-Means)
    dist_opts = geo.distance_options
    if len(dist_opts) > 1:
        st.markdown("## Distance (K-Means)")
        dist_name = st.selectbox(
            "Distance metric", list(dist_opts.keys()),
            label_visibility="collapsed")
        dist_key = dist_opts[dist_name]
    else:
        dist_name = list(dist_opts.keys())[0]
        dist_key = list(dist_opts.values())[0]

    st.markdown("## Data")
    K = st.slider("Clusters (K)", 2, 8, geo.default_K)
    n_points = st.slider("Points", 50, 500, geo.default_n_points, step=50)
    spread = st.slider(
        geo.spread_label,
        geo.spread_range[0], geo.spread_range[1],
        geo.spread_default, step=0.01,
        help=geo.spread_help)
    seed = st.number_input("Seed", 0, 9999, 42)

    generate = st.button(
        "Generate New Data", use_container_width=True, type="primary")

    # ── Step-through controls (always shown) ──
    st.markdown("---")
    st.markdown("## Step-Through")

    show_soft = False
    if algo in ("EM", "Side-by-Side"):
        show_soft = st.checkbox("Soft assignment colors", value=True)

    col_s, col_c = st.columns(2)
    step_btn = col_s.button("Step", use_container_width=True)
    converge_btn = col_c.button("Converge", use_container_width=True)
    reset_btn = st.button("Reset", use_container_width=True)

# ─── Session state ──────────────────────────────────────────────────────────
for _k in ('data', 'em_state', 'km_state', 'prev_params'):
    if _k not in st.session_state:
        st.session_state[_k] = None

current_params = (geo_name, K, n_points, spread, seed, dist_key)
params_changed = (st.session_state.prev_params != current_params)

if generate or st.session_state.data is None or params_changed:
    st.session_state.data = geo.generate_data(K, n_points, spread, seed)
    st.session_state.em_state = None
    st.session_state.km_state = None
    st.session_state.prev_params = current_params

data = st.session_state.data

# ─── K-Means state management ──────────────────────────────────────────────
if algo in ("K-Means", "Side-by-Side"):
    if st.session_state.km_state is None or reset_btn:
        st.session_state.km_state = geo.kmeans_init(
            data['points'], K, seed=seed, data=data, distance=dist_key)

    if step_btn and st.session_state.km_state is not None:
        st.session_state.km_state = geo.kmeans_step(
            data['points'], st.session_state.km_state)

    if converge_btn and st.session_state.km_state is not None:
        state = st.session_state.km_state
        for _ in range(500):
            new_state = geo.kmeans_step(data['points'], state)
            shift = new_state.get('_shift', 0)
            improved = new_state.get('_improved', True)
            state = new_state
            if shift < 1e-6 or not improved:
                break
        st.session_state.km_state = state

# ─── EM state management ───────────────────────────────────────────────────
if algo in ("EM", "Side-by-Side"):
    if st.session_state.em_state is None or reset_btn:
        st.session_state.em_state = geo.em_init(
            data['points'], K, seed=seed, data=data)

    if step_btn and st.session_state.em_state is not None:
        st.session_state.em_state = geo.em_step(
            data['points'], st.session_state.em_state)

    if converge_btn and st.session_state.em_state is not None:
        state = st.session_state.em_state
        for _ in range(500):
            new_state = geo.em_step(data['points'], state)
            if (new_state['iteration'] > 1 and
                abs(new_state['log_likelihood'] - state['log_likelihood']) < 1e-6):
                state = new_state
                break
            state = new_state
        st.session_state.em_state = state

# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown(
    f"<h1 style='margin-bottom:0; color:#264653;'>Clustering in Curved Spaces</h1>"
    f"<p style='color:#6c757d; font-size:1.1rem; margin-top:0;'>"
    f"{geo_name} &mdash; {algo}"
    f"{f' &mdash; {dist_name}' if len(dist_opts) > 1 and algo != 'EM' else ''}</p>",
    unsafe_allow_html=True,
)

# ─── Helpers ────────────────────────────────────────────────────────────────
def _em_centers_info(em):
    info = {'means': em['means']}
    for key in ('covariances', 'kappas', 'sigmas',
                'climate_probs', 'terrain_probs'):
        if key in em:
            info[key] = em[key]
    return info


def _show_km(km_state, col=None):
    """Render K-Means plot + metrics."""
    target = col or st
    if km_state is None:
        return
    target.plotly_chart(
        geo.build_plot(
            data['points'], km_state['labels'], km_state['centroids'],
            title=f"K-Means — iter {km_state['iteration']}",
            true_labels=data['labels']),
        use_container_width=True)


def _show_em(em_state, col=None):
    """Render EM plot + metrics."""
    target = col or st
    if em_state is None:
        return
    labels = np.argmax(em_state['responsibilities'], axis=1)
    target.plotly_chart(
        geo.build_plot(
            data['points'], labels, _em_centers_info(em_state),
            responsibilities=em_state['responsibilities'],
            show_soft=show_soft,
            title=f"EM — iter {em_state['iteration']}",
            true_labels=data['labels']),
        use_container_width=True)


# ─── Display ────────────────────────────────────────────────────────────────
if algo == "K-Means":
    km = st.session_state.km_state
    _show_km(km)

    if km and len(km['cost_history']) > 0:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(
                geo.build_convergence_plot(km['cost_history'], "K-Means Cost"),
                use_container_width=True)
        with col2:
            st.metric("Iteration", km['iteration'])
            st.metric("Cost", f"{km['cost_history'][-1]:.1f}")
            st.metric("Clusters", K)

elif algo == "EM":
    em = st.session_state.em_state
    _show_em(em)

    if em and len(em['ll_history']) > 0:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(
                geo.build_convergence_plot(em['ll_history'], "Log-Likelihood"),
                use_container_width=True)
        with col2:
            st.metric("Iteration", em['iteration'])
            ll_str = f"{em['log_likelihood']:.1f}" if em['log_likelihood'] != 0 else "—"
            st.metric("Log-Likelihood", ll_str)
            st.metric("Clusters", K)

elif algo == "Side-by-Side":
    km = st.session_state.km_state
    em = st.session_state.em_state

    col_left, col_right = st.columns(2)

    with col_left:
        iter_km = km['iteration'] if km else 0
        st.markdown(
            f"<h3 style='color:#264653; text-align:center;'>"
            f"K-Means &mdash; iter {iter_km}</h3>",
            unsafe_allow_html=True)
        _show_km(km, col_left)

    with col_right:
        iter_em = em['iteration'] if em else 0
        st.markdown(
            f"<h3 style='color:#264653; text-align:center;'>"
            f"EM &mdash; iter {iter_em}</h3>",
            unsafe_allow_html=True)
        _show_em(em, col_right)

    # Convergence side by side
    c_left, c_right = st.columns(2)
    with c_left:
        if km and len(km['cost_history']) > 0:
            st.plotly_chart(
                geo.build_convergence_plot(km['cost_history'], "K-Means Cost"),
                use_container_width=True)
    with c_right:
        if em and len(em['ll_history']) > 0:
            st.plotly_chart(
                geo.build_convergence_plot(em['ll_history'], "EM Log-Likelihood"),
                use_container_width=True)

# ─── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#adb5bd; font-size:0.8rem;'>"
    "MAA Seaway Section &middot; Spring 2026 &middot; "
    "St. John Fisher University, Rochester NY</div>",
    unsafe_allow_html=True,
)
