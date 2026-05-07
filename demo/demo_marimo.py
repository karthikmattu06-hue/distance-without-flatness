"""
demo_marimo.py — "The Shape of Closeness"
==========================================
Interactive marimo app for the MAA talk.

Run:  marimo run  demo/demo_marimo.py
Edit: marimo edit demo/demo_marimo.py
"""

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="full")


# ─── Imports & theme ─────────────────────────────────────────────────────────

@app.cell
def _imports():
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    import demo_utils as du
    du.apply_dark_theme()

    import marimo as mo
    return du, mo, np, os, plt


# ─── Sphere data (load once) ─────────────────────────────────────────────────

@app.cell
def _sphere_data(du, np, os, sphere_dataset):
    _DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

    _raw, _label, _synth = du.load_sphere_data(_DATA_DIR, dataset=sphere_dataset.value)

    # Subsample for performance  (K-Medoids PAM is O(N²))
    _MAX_N = 200
    _lats_f = _raw['lats']
    _lons_f = _raw['lons']

    if len(_lats_f) > _MAX_N:
        _rng = np.random.default_rng(42)
        _idx = np.sort(_rng.choice(len(_lats_f), _MAX_N, replace=False))
        _mags_f = _raw['magnitude']
        sphere_data = {
            'lats': _lats_f[_idx],
            'lons': _lons_f[_idx],
            'magnitude': _mags_f[_idx] if _mags_f is not None else None,
        }
        sphere_label = _label + f'  (N = {_MAX_N} sub-sampled)'
    else:
        sphere_data = _raw
        sphere_label = _label

    sphere_is_synthetic = _synth

    # Precompute distance matrices once — used for K-Medoids and validation
    _lats = sphere_data['lats']
    _lons = sphere_data['lons']
    sphere_D = du.compute_haversine_matrix(_lats, _lons)           # Haversine NxN

    _pts_2d = np.stack([_lats, _lons], axis=1)
    sphere_D_eucl = np.sqrt(                                        # Euclidean NxN
        (((_pts_2d[:, None] - _pts_2d[None, :]) ** 2).sum(axis=2))
    )

    return sphere_D, sphere_D_eucl, sphere_data, sphere_is_synthetic, sphere_label


# ─── Sphere UI controls ──────────────────────────────────────────────────────

@app.cell
def _sphere_controls(mo, os, du):
    _DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    _candidates = [
        ('USGS Earthquakes (M2.5+, past 30 days)', 'earthquakes', 'earthquakes.csv'),
        ('NASA Meteorite Landings',                 'meteorites',  'meteorites.csv'),
        ('Global Airports (OpenFlights)',           'airports',    'airports.dat'),
    ]
    _opts = {}
    for _lbl, _key, _fname in _candidates:
        _path = os.path.join(_DATA_DIR, _fname)
        if not os.path.exists(_path):
            try:
                du.download_if_missing(du._SPHERE_URLS[_key], _path)
            except Exception:
                pass
        if os.path.exists(_path):
            _opts[_lbl] = _key
    _opts['Synthetic sphere clusters'] = 'synthetic'

    sphere_dataset = mo.ui.dropdown(
        options=_opts,
        value=next(iter(_opts)),
        label="Dataset",
    )
    sphere_algorithm = mo.ui.radio(
        options=["K-Means", "K-Medoids", "GMM"],
        value="K-Means",
        label="Algorithm",
        inline=True,
    )
    sphere_k = mo.ui.slider(
        start=2, stop=10, value=3,
        label="Number of clusters  K",
    )
    return sphere_dataset, sphere_algorithm, sphere_k


# ─── Sphere: K-sweep (re-runs on algorithm change) ───────────────────────────

@app.cell
def _sphere_sweep(du, mo, np, sphere_algorithm, sphere_D, sphere_data):
    _algo = sphere_algorithm.value
    _lats  = sphere_data['lats']
    _lons  = sphere_data['lons']

    if _algo == "K-Means":
        def _run(k, D):
            return du.spherical_kmeans(_lats, _lons, k=k)

    elif _algo == "K-Medoids":
        def _run(k, D):
            return du.kmedoids_pam(D, k=k)

    else:  # GMM (von Mises-Fisher on S²)
        _x, _y, _z = du.latlon_to_cartesian(_lats, _lons)
        _pts3d = np.stack([_x, _y, _z], axis=1)
        def _run(k, D):
            return du.spherical_gmm(_pts3d, k=k)

    with mo.status.spinner(f"Computing {_algo} sweep  K = 2–10 …"):
        sphere_sweep = du.sweep_k(_run, sphere_D, range(2, 11))

    return (sphere_sweep,)


# ─── Sphere: side-by-side maps (re-runs on algorithm OR k change) ─────────────

@app.cell
def _sphere_maps(
    du, mo, np, plt,
    sphere_algorithm, sphere_k,
    sphere_D, sphere_D_eucl, sphere_data,
):
    _algo = sphere_algorithm.value
    _k    = sphere_k.value
    _lats = sphere_data['lats']
    _lons = sphere_data['lons']
    _pts2 = np.stack([_lats, _lons], axis=1)

    # ── Euclidean clustering ─────────────────────────────────────────────────
    if _algo == "K-Means":
        _el, _ec, _ = du.minkowski_kmeans(_pts2, k=_k, p=2)
        _e_clats, _e_clons = _ec[:, 0], _ec[:, 1]

    elif _algo == "K-Medoids":
        _el, _em, _ = du.kmedoids_pam(sphere_D_eucl, k=_k)
        _e_clats, _e_clons = _lats[_em], _lons[_em]

    else:  # GMM
        _el, _emeans, _ = du.euclidean_gmm(_pts2, k=_k)
        _e_clats, _e_clons = _emeans[:, 0], _emeans[:, 1]

    # ── Haversine clustering ─────────────────────────────────────────────────
    if _algo == "K-Means":
        _hl, _hc, _ = du.spherical_kmeans(_lats, _lons, k=_k)
        _h_clats, _h_clons = _hc[:, 0], _hc[:, 1]

    elif _algo == "K-Medoids":
        _hl, _hm, _ = du.kmedoids_pam(sphere_D, k=_k)
        _h_clats, _h_clons = _lats[_hm], _lons[_hm]

    else:  # GMM  (vMF on S²)
        _x, _y, _z = du.latlon_to_cartesian(_lats, _lons)
        _pts3d = np.stack([_x, _y, _z], axis=1)
        _hl, _hmeans, _ = du.spherical_gmm(_pts3d, k=_k)
        _h_clats = np.array([du.cartesian_to_latlon(m[0], m[1], m[2])[0] for m in _hmeans])
        _h_clons = np.array([du.cartesian_to_latlon(m[0], m[1], m[2])[1] for m in _hmeans])

    # ── Label alignment & disagreement count ────────────────────────────────
    from scipy.optimize import linear_sum_assignment as _lsa
    _conf = np.zeros((_k, _k))
    for _ei, _hi in zip(_el, _hl):
        _conf[_ei, _hi] += 1
    _ri, _ci = _lsa(-_conf)
    _h2e = dict(zip(_ci, _ri))
    _hl_aligned = np.array([_h2e.get(int(l), int(l)) for l in _hl])
    _n_diff = int(np.sum(_el != _hl_aligned))
    _pct    = 100.0 * _n_diff / len(_el)

    # ── Plot ─────────────────────────────────────────────────────────────────
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    _fig.patch.set_facecolor(du.BG_FIGURE)

    du.plot_sphere_map(_ax1, _lats, _lons, _el, _e_clats, _e_clons,
                       f'Euclidean  ({_algo})', metric='euclidean')
    du.plot_sphere_map(_ax2, _lats, _lons, _hl, _h_clats, _h_clons,
                       f'Haversine  ({_algo})', metric='haversine')

    plt.tight_layout()
    sphere_maps_img = mo.image(du.fig_to_png(_fig), width="100%")
    plt.close(_fig)

    _kind = "warn" if _pct > 15 else ("success" if _pct < 5 else "info")
    sphere_disagree_txt = mo.callout(
        mo.md(
            f"**{_n_diff} of {len(_el)} points ({_pct:.1f}%)** assigned to "
            f"different clusters when comparing Euclidean vs Haversine {_algo}."
        ),
        kind=_kind,
    )

    sphere_pct    = _pct
    sphere_n_diff = _n_diff
    return sphere_disagree_txt, sphere_maps_img, sphere_pct, sphere_n_diff


# ─── Sphere: validation curves (re-runs on sweep or k change) ────────────────

@app.cell
def _sphere_val_plots(du, mo, plt, sphere_k, sphere_sweep):
    _k     = sphere_k.value
    _k_vals = sphere_sweep['k_values']
    _pve    = sphere_sweep['pve']
    _sil    = sphere_sweep['silhouette_mean']

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    _fig.patch.set_facecolor(du.BG_FIGURE)
    du.plot_pve_curve(_ax1, _k_vals, _pve, _k)
    du.plot_silhouette_curve(_ax2, _k_vals, _sil, _k)
    plt.tight_layout()
    sphere_val_img = mo.image(du.fig_to_png(_fig), width="100%")
    plt.close(_fig)

    return (sphere_val_img,)


# ─── Sphere: interactive globe (Scattergeo orthographic) ────────────────────

@app.cell
def _sphere_3d(du, mo, np, sphere_k, sphere_data, sphere_algorithm, sphere_D):
    import plotly.graph_objects as _go

    _lats = sphere_data['lats']
    _lons = sphere_data['lons']
    _k    = sphere_k.value
    _algo = sphere_algorithm.value

    if _algo == "K-Means":
        _labels, _, _ = du.spherical_kmeans(_lats, _lons, k=_k)
    elif _algo == "K-Medoids":
        _labels, _, _ = du.kmedoids_pam(sphere_D, k=_k)
    else:
        _x0, _y0, _z0 = du.latlon_to_cartesian(_lats, _lons)
        _labels, _, _ = du.spherical_gmm(np.stack([_x0, _y0, _z0], axis=1), k=_k)

    _lats_deg = np.rad2deg(_lats)
    _lons_deg = np.rad2deg(_lons)

    _fig = _go.Figure()
    for _c in range(_k):
        _m = _labels == _c
        _fig.add_trace(_go.Scattergeo(
            lat=_lats_deg[_m],
            lon=_lons_deg[_m],
            mode='markers',
            marker=dict(
                size=7,
                color=du.CLUSTER_COLORS[_c % len(du.CLUSTER_COLORS)],
                opacity=0.88,
                line=dict(width=0.5, color='white'),
            ),
            name=f'Cluster {_c}',
        ))

    _fig.update_geos(
        projection_type='orthographic',
        showland=True,       landcolor='#D4E6C3',
        showocean=True,      oceancolor='#B3D4F0',
        showlakes=True,      lakecolor='#B3D4F0',
        showrivers=False,
        showcountries=True,  countrycolor='#6B7280', countrywidth=0.5,
        showcoastlines=True, coastlinecolor='#374151', coastlinewidth=0.8,
        showframe=True,      framecolor='#94A3B8',   framewidth=1,
        bgcolor=du.BG_FIGURE,
        lataxis=dict(showgrid=True, gridcolor='rgba(180,200,220,0.5)', gridwidth=0.5),
        lonaxis=dict(showgrid=True, gridcolor='rgba(180,200,220,0.5)', gridwidth=0.5),
    )
    _fig.update_layout(
        title=dict(text=f'Globe — {_algo}  (K = {_k})  ·  drag to rotate',
                   font=dict(size=14, color='#1A2030')),
        paper_bgcolor=du.BG_FIGURE,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(font=dict(color='#1A2030'), bgcolor=du.BG_FIGURE),
        height=650,
    )

    # Embed in an iframe so scripts execute in their own document context
    # and marimo cannot intercept drag-to-rotate events
    import plotly.io as _pio
    import html as _html_mod
    _full = _pio.to_html(_fig, include_plotlyjs=True, full_html=True,
                         config={'displayModeBar': False})
    _escaped = _html_mod.escape(_full, quote=True)
    sphere_3d_plotly = mo.Html(
        f'<iframe srcdoc="{_escaped}" '
        f'style="width:100%;height:670px;border:none;" scrolling="no"></iframe>'
    )
    return (sphere_3d_plotly,)


# ─── Sphere tab assembly ─────────────────────────────────────────────────────

@app.cell
def _sphere_tab(
    mo,
    sphere_dataset, sphere_algorithm, sphere_k,
    sphere_disagree_txt, sphere_is_synthetic, sphere_label,
    sphere_maps_img, sphere_val_img, sphere_3d_plotly,
    sphere_pct, sphere_n_diff,
):
    _header = mo.md(
        f"## 🌍 Sphere — Haversine Distance\n\n"
        f"*When data lives on a globe, straight-line distance cuts through the Earth.*\n\n"
        f"**Data:** {sphere_label}"
    )

    _SPHERE_DESC = {
        'earthquakes': "M2.5+ seismic events recorded by USGS in the past 30 days. Points cluster naturally near **tectonic plate boundaries** — the Pacific Ring of Fire, the Mid-Atlantic Ridge, and the Himalayan collision zone. Haversine distance respects Earth's curvature; Euclidean treats lat/lon as a flat grid.",
        'meteorites':  "45,000+ confirmed meteorite finds and falls from NASA's open catalog. Dense clusters appear near **Antarctica** (systematic grid searches since the 1970s) and in populated regions where observation rates are highest. A striking example of sampling bias shaping apparent clusters.",
        'airports':    "7,000+ commercial airports from the OpenFlights database. Clusters reflect **air-traffic hubs** and population density: North America's eastern seaboard, the European triangle, and South/Southeast Asia's coastal corridor.",
        'synthetic':   "Gaussian clusters placed at random locations on the sphere surface. Use this to explore how the algorithm behaves without any real-world structure.",
    }
    _ds_key = sphere_dataset.value if sphere_dataset.value in _SPHERE_DESC else 'synthetic'
    _data_info = mo.callout(mo.md(_SPHERE_DESC[_ds_key]), kind="info")

    _warning = (
        mo.callout(mo.md(f"⚠️ **Synthetic data** — {sphere_label} could not be loaded; showing random sphere clusters instead."), kind="warn")
        if sphere_is_synthetic
        else mo.md("")
    )

    _SPHERE_INSIGHT = {
        'earthquakes': (
            f"Haversine correctly clusters seismic events along **tectonic plate boundaries** "
            f"(Ring of Fire, Mid-Atlantic Ridge), while Euclidean distance misassigns "
            f"**{sphere_pct:.1f}%** of points ({sphere_n_diff}) by treating lat/lon as a flat grid."
        ),
        'meteorites': (
            f"Haversine correctly groups meteorite finds by true surface proximity, while Euclidean "
            f"misassigns **{sphere_pct:.1f}%** of points ({sphere_n_diff}) — particularly distorting "
            f"polar clusters where Antarctic systematic searches dominate."
        ),
        'airports': (
            f"Haversine clusters airports by actual great-circle proximity (reflecting air-traffic hubs), "
            f"while Euclidean misassigns **{sphere_pct:.1f}%** of points ({sphere_n_diff}) — "
            f"distortions are largest near the poles where lon/lat grid cells compress."
        ),
        'synthetic': (
            f"Even on random sphere data, Euclidean distance misassigns **{sphere_pct:.1f}%** of points "
            f"({sphere_n_diff}) relative to Haversine — curvature error accumulates wherever clusters "
            f"span large angular separations."
        ),
    }
    _insight_txt = _SPHERE_INSIGHT.get(_ds_key, _SPHERE_INSIGHT['synthetic'])
    _insight = mo.callout(mo.md(f"**Key Insight:** {_insight_txt}"), kind="success")

    _formula = mo.md(r"""
---
#### 📐 Distance Formulas

**Haversine (great-circle distance on a sphere of radius $r$):**

$$d = 2r \arcsin\!\sqrt{\sin^2\!\tfrac{\Delta\phi}{2} + \cos\phi_1\cos\phi_2\sin^2\!\tfrac{\Delta\lambda}{2}}$$

where $\phi$ = latitude, $\lambda$ = longitude (both in radians).

**Proportion of Variance Explained (PVE):**

$$\text{PVE}(K) = 1 - \frac{\sum_{i} d(x_i,\, c_{\ell_i})^2}{\sum_{i} d(x_i,\, \bar{x})^2}$$

**Silhouette score for point $i$:**

$$s(i) = \frac{b(i) - a(i)}{\max(a(i),\, b(i))}, \quad s(i) \in [-1, 1]$$

where $a(i)$ = mean intra-cluster distance, $b(i)$ = mean nearest-other-cluster distance.
""")

    _main_row = mo.Html(
        '<div style="display:grid;grid-template-columns:65% 34%;gap:1rem;align-items:start">'
        f'<div>{mo.vstack([sphere_3d_plotly, sphere_disagree_txt], gap="0.4rem").text}</div>'
        f'<div>{mo.vstack([_formula, _data_info, _insight], gap="0.5rem").text}</div>'
        '</div>'
    )
    sphere_content = mo.vstack([
        mo.md(f"## 🌍 Sphere — Haversine Distance &nbsp;·&nbsp; *{sphere_label}*"),
        _warning,
        mo.hstack([sphere_dataset, sphere_algorithm, sphere_k], gap="1.5rem"),
        _main_row,
        mo.Html(
            '<div style="display:grid;grid-template-columns:62% 37%;gap:1rem;align-items:start">'
            f'<div>{mo.vstack([mo.md("**Euclidean vs Haversine**"), sphere_maps_img], gap="0.3rem").text}</div>'
            f'<div>{mo.vstack([mo.md("**Validation Sweep K = 2–10**"), sphere_val_img], gap="0.3rem").text}</div>'
            '</div>'
        ),
    ], gap="0.6rem")

    return (sphere_content,)


# ─── Overview tab ────────────────────────────────────────────────────────────

@app.cell
def _overview(mo):
    overview_content = mo.vstack([
        mo.md("# The Shape of Closeness"),
        mo.md("### *Distance Metrics for Clustering Across Geometric Surfaces*"),
        mo.md(
            """
## Project Map

| Surface | Distance Metric | Algorithms | Real Data |
|:--------|:----------------|:-----------|:----------|
| 🌍 Sphere | Haversine (great-circle) | K-Means, K-Medoids, GMM | USGS Earthquakes |
| 🍩 Torus | Wraparound (flat torus) | K-Means, K-Medoids, GMM | Protein Backbone Angles |
| 🔵 Hyperbolic | Poincaré disk metric | K-Means, K-Medoids, GMM | Animal Taxonomy |
| 📏 Feature Space | Minkowski Lp, Mahalanobis | K-Means, K-Medoids, GMM | Palmer Penguins |
| 🔀 Mixed Type | Gower (Haversine + categorical) | K-Medoids | Earthquakes + Categories |
"""
        ),
        mo.callout(
            mo.md(
                "The choice of distance metric shapes what *closeness* means — "
                "and determines whether clustering succeeds or fails. "
                "Each tab demonstrates this on a different geometric surface, "
                "with validation via **PVE** (Proportion of Variance Explained) "
                "and **Silhouette** scores."
            ),
            kind="info",
        ),
        mo.md(
            """
## Convergence & Metric Properties

| Geometry | Distance | True metric? | WCSS ↓ monotone? | Converges? | Global optimum? |
|:---------|:---------|:---:|:---:|:---:|:---:|
| Euclidean ℝⁿ | $L^2$ | ✓ | ✓ | ✓ | ✗ (local min) |
| Sphere $S^2$ | Haversine | ✓ | ✓ | ✓ | ✗ |
| Flat torus $T^2$ | Wraparound $L^2$ | ✓ | ✓ | ✓* | ✗ |
| Poincaré disk $\\mathbb{D}^2$ | $\\operatorname{arccosh}$ | ✓ | ✓ | ✓ | ✗ |
| Feature space | Minkowski $L^p$ | ✓ ($p \\geq 1$) | ✓ | ✓ | ✗ |
| Mixed type | Gower | ✗ (triangle ineq. fails) | ✓ | ✓ | ✗ |

*\\*Torus circular mean is the Fréchet mean only when clusters don't straddle the $0/2\\pi$ cut — degenerate cases exist.*

> **Lloyd's guarantee:** every assign-then-update cycle **cannot increase** WCSS, so the algorithm converges in finitely many steps. But the limit depends on initialisation and may be a local, not global, minimum — hence the need for multiple restarts or K-Means++.
"""
        ),
    ])
    return (overview_content,)


# ─── Torus data (load once) ──────────────────────────────────────────────────

@app.cell
def _torus_data(du, np, os, torus_seed):
    _DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    _raw, _label, _simulated = du.load_torus_data(_DATA_DIR, seed=torus_seed.value)

    _MAX_N = 200
    _pts_f = _raw['points']
    if len(_pts_f) > _MAX_N:
        _rng = np.random.default_rng(43)
        _idx = np.sort(_rng.choice(len(_pts_f), _MAX_N, replace=False))
        _ss = _raw['secondary_structure']
        torus_data = {
            'points': _pts_f[_idx],
            'phi':    _raw['phi'][_idx],
            'psi':    _raw['psi'][_idx],
            'secondary_structure': _ss[_idx] if _ss is not None else None,
        }
        torus_label = _label + f'  (N = {_MAX_N} sub-sampled)'
    else:
        torus_data = _raw
        torus_label = _label

    torus_is_simulated = _simulated

    _pts = torus_data['points']
    torus_D      = du.compute_torus_matrix(_pts)
    torus_D_eucl = np.sqrt(
        np.sum((_pts[:, np.newaxis] - _pts[np.newaxis, :])**2, axis=-1)
    )

    return torus_D, torus_D_eucl, torus_data, torus_is_simulated, torus_label


# ─── Torus UI controls ───────────────────────────────────────────────────────

@app.cell
def _torus_controls(mo):
    torus_algorithm = mo.ui.radio(
        options=["K-Means", "K-Medoids", "GMM"],
        value="K-Means",
        label="Algorithm",
        inline=True,
    )
    torus_k = mo.ui.slider(
        start=2, stop=10, value=3,
        label="Number of clusters  K",
    )
    torus_seed = mo.ui.slider(start=0, stop=99, value=42, label="Synthetic seed")
    return torus_seed, torus_algorithm, torus_k


# ─── Torus: K-sweep (re-runs on algorithm change) ────────────────────────────

@app.cell
def _torus_sweep(du, mo, np, torus_algorithm, torus_D, torus_data):
    _algo = torus_algorithm.value
    _pts  = torus_data['points']

    if _algo == "K-Means":
        def _run(k, D):
            return du.torus_kmeans(_pts, k=k)
    elif _algo == "K-Medoids":
        def _run(k, D):
            return du.kmedoids_pam(D, k=k)
    else:  # GMM
        def _run(k, D):
            return du.torus_gmm(_pts, k=k)

    with mo.status.spinner(f"Computing {_algo} sweep  K = 2–10 …"):
        torus_sweep = du.sweep_k(_run, torus_D, range(2, 11))

    return (torus_sweep,)


# ─── Torus: side-by-side maps (re-runs on algorithm OR k change) ─────────────

@app.cell
def _torus_maps(
    du, mo, np, plt,
    torus_algorithm, torus_k,
    torus_D, torus_D_eucl, torus_data,
):
    _algo = torus_algorithm.value
    _k    = torus_k.value
    _pts  = torus_data['points']

    # ── Euclidean clustering ─────────────────────────────────────────────────
    if _algo == "K-Means":
        _el, _ec, _ = du.minkowski_kmeans(_pts, k=_k, p=2)
        _e_centers  = _ec
    elif _algo == "K-Medoids":
        _el, _em, _ = du.kmedoids_pam(torus_D_eucl, k=_k)
        _e_centers  = _pts[_em]
    else:
        _el, _em, _ = du.euclidean_gmm(_pts, k=_k)
        _e_centers  = _em

    # ── Toroidal clustering ──────────────────────────────────────────────────
    if _algo == "K-Means":
        _tl, _tc, _ = du.torus_kmeans(_pts, k=_k)
        _t_centers  = _tc
    elif _algo == "K-Medoids":
        _tl, _tm, _ = du.kmedoids_pam(torus_D, k=_k)
        _t_centers  = _pts[_tm]
    else:
        _tl, _tm, _ = du.torus_gmm(_pts, k=_k)
        _t_centers  = _tm

    # ── Label alignment & disagreement count ────────────────────────────────
    from scipy.optimize import linear_sum_assignment as _lsa
    _conf = np.zeros((_k, _k))
    for _ei, _ti in zip(_el, _tl):
        _conf[_ei, _ti] += 1
    _ri, _ci = _lsa(-_conf)
    _t2e = dict(zip(_ci, _ri))
    _tl_aligned = np.array([_t2e.get(int(l), int(l)) for l in _tl])
    _n_diff = int(np.sum(_el != _tl_aligned))
    _pct    = 100.0 * _n_diff / len(_el)

    # % of disagreements near a wraparound boundary (within 0.8 rad)
    _diff_mask = _el != _tl_aligned
    _near = (
        (_pts[:, 0] < 0.8) | (_pts[:, 0] > 2 * np.pi - 0.8) |
        (_pts[:, 1] < 0.8) | (_pts[:, 1] > 2 * np.pi - 0.8)
    )
    _near_pct = 100.0 * int(np.sum(_diff_mask & _near)) / max(_n_diff, 1)

    # ── Plot ─────────────────────────────────────────────────────────────────
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    _fig.patch.set_facecolor(du.BG_FIGURE)

    du.plot_torus_map(_ax1, _pts, _el, _e_centers,
                      f'Euclidean  ({_algo})', metric='euclidean')
    du.plot_torus_map(_ax2, _pts, _tl, _t_centers,
                      f'Toroidal  ({_algo})', metric='toroidal')

    plt.tight_layout()
    torus_maps_img = mo.image(du.fig_to_png(_fig), width="100%")
    plt.close(_fig)

    _kind = "warn" if _pct > 15 else ("success" if _pct < 5 else "info")
    torus_disagree_txt = mo.callout(
        mo.md(
            f"**{_n_diff} of {len(_el)} points ({_pct:.1f}%)** assigned differently "
            f"(Euclidean vs Toroidal {_algo}).  "
            f"**{_near_pct:.0f}%** of disagreements are near a wraparound boundary."
        ),
        kind=_kind,
    )

    torus_pct      = _pct
    torus_near_pct = _near_pct
    return torus_disagree_txt, torus_maps_img, torus_pct, torus_near_pct


# ─── Torus: interactive 3D donut (plotly) ────────────────────────────────────

@app.cell
def _torus_3d(du, mo, np, torus_k, torus_data):
    import plotly.graph_objects as _go

    _pts = torus_data['points']
    _k   = torus_k.value
    _tl, _, _ = du.torus_kmeans(_pts, k=_k)

    # Torus surface mesh
    _u = np.linspace(0, 2 * np.pi, 50)
    _v = np.linspace(0, 2 * np.pi, 50)
    _uu, _vv = np.meshgrid(_u, _v)
    _sx, _sy, _sz = du.torus_to_3d(_uu, _vv)

    _fig = _go.Figure()
    _fig.add_trace(_go.Surface(
        x=_sx, y=_sy, z=_sz,
        opacity=0.55,
        colorscale='Blues',
        cmin=_sz.min(), cmax=_sz.max(),
        showscale=False, hoverinfo='skip', name='torus',
        lighting=dict(ambient=0.6, diffuse=0.8, specular=0.3, roughness=0.5),
        lightposition=dict(x=2, y=2, z=2),
    ))

    for _c in range(_k):
        _mask = _tl == _c
        _cx, _cy, _cz = du.torus_to_3d(_pts[_mask, 0], _pts[_mask, 1])
        _fig.add_trace(_go.Scatter3d(
            x=_cx, y=_cy, z=_cz,
            mode='markers',
            marker=dict(size=3.5, color=du.CLUSTER_COLORS[_c % len(du.CLUSTER_COLORS)], opacity=0.85),
            name=f'Cluster {_c}',
        ))

    _fig.update_layout(
        title=dict(text=f'Torus — Toroidal K-Means  (K = {_k})', font=dict(size=14, color='#1A2030')),
        scene=dict(
            aspectmode='data',
            bgcolor=du.BG_FIGURE,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        paper_bgcolor=du.BG_FIGURE,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(font=dict(color='#1A2030')),
        height=650,
    )
    torus_3d_plotly = mo.ui.plotly(_fig)
    return (torus_3d_plotly,)


# ─── Torus: validation curves (re-runs on sweep or k change) ─────────────────

@app.cell
def _torus_val_plots(du, mo, plt, torus_k, torus_sweep):
    _k      = torus_k.value
    _k_vals = torus_sweep['k_values']
    _pve    = torus_sweep['pve']
    _sil    = torus_sweep['silhouette_mean']

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    _fig.patch.set_facecolor(du.BG_FIGURE)
    du.plot_pve_curve(_ax1, _k_vals, _pve, _k)
    du.plot_silhouette_curve(_ax2, _k_vals, _sil, _k)
    plt.tight_layout()
    torus_val_img = mo.image(du.fig_to_png(_fig), width="100%")
    plt.close(_fig)

    return (torus_val_img,)


# ─── Torus tab assembly ──────────────────────────────────────────────────────

@app.cell
def _torus_tab(
    mo,
    torus_algorithm, torus_k, torus_seed,
    torus_disagree_txt, torus_is_simulated, torus_label,
    torus_maps_img, torus_val_img,
    torus_3d_plotly, torus_pct, torus_near_pct,
):
    _header = mo.md(
        f"## 🍩 Torus — Wraparound Distance\n\n"
        f"*Periodic data wraps around — clustering must respect the edges.*\n\n"
        f"**Data:** {torus_label}"
    )

    _data_info = mo.callout(mo.md(
        "**Protein backbone dihedral angles** (φ, ψ) from ~500 amino acid residues, "
        "generated to match known Ramachandran distributions. "
        "α-helices cluster near **(−60°, −45°)**, β-sheets near **(−120°, 130°)**. "
        "Because angles wrap at ±180°, a pair like (179°, −179°) is only 2° apart — "
        "the torus is the natural space for periodic data like this. "
        "Drag the **seed slider** to generate a new random realisation."
    ), kind="info")

    _notice = (
        mo.callout(mo.md("⚠️ **Synthetic data** — Ramachandran CSV not found."), kind="warn")
        if torus_is_simulated is None
        else mo.md("")
    )

    _insight = mo.callout(
        mo.md(
            f"**Key Insight:** Toroidal distance correctly handles wraparound in φ/ψ angle space. "
            f"Euclidean distance misassigns **{torus_pct:.1f}%** of points; "
            f"**{torus_near_pct:.0f}%** of those errors occur near the 0°/360° boundary where "
            f"Euclidean distance sees two nearby angles as maximally far apart."
        ),
        kind="success",
    )

    _formula = mo.md(r"""
---
#### 📐 Distance Formula

**Flat torus distance (wraparound L2):**

$$d_{\text{torus}}(u, v) = \sqrt{\sum_{j=1}^{2} \min(|u_j - v_j|,\; L - |u_j - v_j|)^2}$$

where $L = 2\pi$ is the period of each angular dimension.
PVE and Silhouette are computed using $d_{\text{torus}}$ as the base distance.
""")

    _main_row = mo.Html(
        '<div style="display:grid;grid-template-columns:65% 34%;gap:1rem;align-items:start">'
        f'<div>{mo.vstack([torus_3d_plotly, torus_disagree_txt], gap="0.4rem").text}</div>'
        f'<div>{mo.vstack([_formula, _data_info, _insight], gap="0.5rem").text}</div>'
        '</div>'
    )
    torus_content = mo.vstack([
        mo.md(f"## 🍩 Torus — Wraparound Distance &nbsp;·&nbsp; *{torus_label}*"),
        _notice,
        mo.hstack([torus_algorithm, torus_k, torus_seed], gap="1.5rem"),
        _main_row,
        mo.Html(
            '<div style="display:grid;grid-template-columns:62% 37%;gap:1rem;align-items:start">'
            f'<div>{mo.vstack([mo.md("**Euclidean vs Toroidal**"), torus_maps_img], gap="0.3rem").text}</div>'
            f'<div>{mo.vstack([mo.md("**Validation Sweep K = 2–10**"), torus_val_img], gap="0.3rem").text}</div>'
            '</div>'
        ),
    ], gap="0.6rem")

    return (torus_content,)


# ─── Hyperbolic data (load once) ─────────────────────────────────────────────

@app.cell
def _hyp_data(du, np, os, hyp_seed):
    _DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    _raw, _label, _is_synth = du.load_hyperbolic_data(_DATA_DIR, seed=hyp_seed.value)

    _MAX_N = 200
    _pts_f = _raw['points']
    if len(_pts_f) > _MAX_N:
        _rng = np.random.default_rng(44)
        _idx = np.sort(_rng.choice(len(_pts_f), _MAX_N, replace=False))
        hyp_data  = {'points': _pts_f[_idx], 'labels': _raw['labels'][_idx]}
        hyp_label = _label + f'  (N = {_MAX_N} sub-sampled)'
    else:
        hyp_data  = _raw
        hyp_label = _label

    hyp_is_synthetic = _is_synth

    _pts      = hyp_data['points']
    hyp_D      = du.compute_poincare_matrix(_pts)
    hyp_D_eucl = np.sqrt(
        np.sum((_pts[:, np.newaxis] - _pts[np.newaxis, :])**2, axis=-1)
    )

    return hyp_D, hyp_D_eucl, hyp_data, hyp_is_synthetic, hyp_label


# ─── Hyperbolic UI controls ───────────────────────────────────────────────────

@app.cell
def _hyp_controls(mo):
    hyp_algorithm = mo.ui.radio(
        options=["K-Means", "K-Medoids", "GMM"],
        value="K-Means",
        label="Algorithm",
        inline=True,
    )
    hyp_k = mo.ui.slider(
        start=2, stop=10, value=3,
        label="Number of clusters  K",
    )
    hyp_distortion_toggle = mo.ui.checkbox(label="Show distance distortion plot")
    hyp_seed = mo.ui.slider(start=0, stop=99, value=42, label="Synthetic seed")
    return hyp_algorithm, hyp_distortion_toggle, hyp_k, hyp_seed


# ─── Hyperbolic: K-sweep (re-runs on algorithm change) ───────────────────────

@app.cell
def _hyp_sweep(du, mo, np, hyp_algorithm, hyp_D, hyp_data):
    _algo = hyp_algorithm.value
    _pts  = hyp_data['points']

    if _algo == "K-Means":
        def _run(k, D):
            return du.hyperbolic_kmeans(_pts, k=k)
    elif _algo == "K-Medoids":
        def _run(k, D):
            return du.kmedoids_pam(D, k=k)
    else:
        def _run(k, D):
            return du.hyperbolic_gmm(_pts, k=k)

    with mo.status.spinner(f"Computing {_algo} sweep  K = 2–10 …"):
        hyp_sweep = du.sweep_k(_run, hyp_D, range(2, 11))

    return (hyp_sweep,)


# ─── Hyperbolic: side-by-side maps (re-runs on algorithm OR k change) ─────────

@app.cell
def _hyp_maps(
    du, mo, np, plt,
    hyp_algorithm, hyp_k,
    hyp_D, hyp_D_eucl, hyp_data,
):
    _algo = hyp_algorithm.value
    _k    = hyp_k.value
    _pts  = hyp_data['points']

    # ── Euclidean clustering ─────────────────────────────────────────────────
    if _algo == "K-Means":
        _el, _ec, _ = du.minkowski_kmeans(_pts, k=_k, p=2)
        _e_centers  = _ec
    elif _algo == "K-Medoids":
        _el, _em, _ = du.kmedoids_pam(hyp_D_eucl, k=_k)
        _e_centers  = _pts[_em]
    else:
        _el, _em, _ = du.euclidean_gmm(_pts, k=_k)
        _e_centers  = _em

    # ── Hyperbolic clustering ────────────────────────────────────────────────
    if _algo == "K-Means":
        _hl, _hc, _ = du.hyperbolic_kmeans(_pts, k=_k)
        _h_centers  = _hc
    elif _algo == "K-Medoids":
        _hl, _hm, _ = du.kmedoids_pam(hyp_D, k=_k)
        _h_centers  = _pts[_hm]
    else:
        _hl, _hm, _ = du.hyperbolic_gmm(_pts, k=_k)
        _h_centers  = _hm

    # ── Label alignment & disagreement ──────────────────────────────────────
    from scipy.optimize import linear_sum_assignment as _lsa
    _conf = np.zeros((_k, _k))
    for _ei, _hi in zip(_el, _hl):
        _conf[_ei, _hi] += 1
    _ri, _ci = _lsa(-_conf)
    _h2e = dict(zip(_ci, _ri))
    _hl_aligned = np.array([_h2e.get(int(l), int(l)) for l in _hl])
    _n_diff = int(np.sum(_el != _hl_aligned))
    _pct    = 100.0 * _n_diff / len(_el)

    # ── Plot ─────────────────────────────────────────────────────────────────
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    _fig.patch.set_facecolor(du.BG_FIGURE)

    du.plot_poincare_disk(_ax1, _pts, _el, _e_centers,
                          f'Euclidean  ({_algo})', metric='euclidean')
    du.plot_poincare_disk(_ax2, _pts, _hl, _h_centers,
                          f'Hyperbolic  ({_algo})', metric='hyperbolic')

    plt.tight_layout()
    hyp_maps_img = mo.image(du.fig_to_png(_fig), width="100%")
    plt.close(_fig)

    _kind = "warn" if _pct > 15 else ("success" if _pct < 5 else "info")
    hyp_disagree_txt = mo.callout(
        mo.md(
            f"**{_n_diff} of {len(_el)} points ({_pct:.1f}%)** assigned to "
            f"different clusters when comparing Euclidean vs Hyperbolic {_algo}."
        ),
        kind=_kind,
    )

    hyp_pct = _pct
    return hyp_disagree_txt, hyp_maps_img, hyp_pct


# ─── Hyperbolic: distance distortion plot (conditional on toggle) ─────────────

@app.cell
def _hyp_distortion(
    du, mo, np, plt,
    hyp_distortion_toggle, hyp_data, hyp_D, hyp_D_eucl,
):
    if not hyp_distortion_toggle.value:
        hyp_distortion_img = mo.md("")
    else:
        _pts = hyp_data['points']
        _n   = len(_pts)

        # Sample random pairs (avoid duplicate indices)
        _rng   = np.random.default_rng(99)
        _i_idx = _rng.integers(0, _n, size=1200)
        _j_idx = _rng.integers(0, _n, size=1200)
        _valid = _i_idx != _j_idx
        _i_idx, _j_idx = _i_idx[_valid][:600], _j_idx[_valid][:600]

        _d_eucl = hyp_D_eucl[_i_idx, _j_idx]
        _d_hyp  = hyp_D[_i_idx, _j_idx]
        _r_avg  = (
            np.linalg.norm(_pts[_i_idx], axis=1) +
            np.linalg.norm(_pts[_j_idx], axis=1)
        ) / 2

        _fig, _ax = plt.subplots(figsize=(11, 3.8))
        _fig.patch.set_facecolor(du.BG_FIGURE)
        _sc = _ax.scatter(_d_eucl, _d_hyp, c=_r_avg, cmap='plasma',
                          s=15, alpha=0.65, linewidths=0)
        _cb = _fig.colorbar(_sc, ax=_ax)
        _cb.set_label('Avg radius of pair', fontsize=11, color='#ccccdd')
        plt.setp(_cb.ax.yaxis.get_ticklabels(), color='#888899')
        _ax.set_xlabel('Euclidean distance', fontsize=13)
        _ax.set_ylabel('Hyperbolic distance', fontsize=13)
        _ax.set_title('Distance Distortion: Euclidean vs Hyperbolic',
                      fontsize=15, fontweight='bold')
        _ax.tick_params(labelsize=11)
        _ax.grid(True, alpha=0.3)
        plt.tight_layout()
        hyp_distortion_img = mo.image(du.fig_to_png(_fig), width="90%")
        plt.close(_fig)

    return (hyp_distortion_img,)


# ─── Hyperbolic: validation curves ───────────────────────────────────────────

@app.cell
def _hyp_val_plots(du, mo, plt, hyp_k, hyp_sweep):
    _k      = hyp_k.value
    _k_vals = hyp_sweep['k_values']
    _pve    = hyp_sweep['pve']
    _sil    = hyp_sweep['silhouette_mean']

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    _fig.patch.set_facecolor(du.BG_FIGURE)
    du.plot_pve_curve(_ax1, _k_vals, _pve, _k)
    du.plot_silhouette_curve(_ax2, _k_vals, _sil, _k)
    plt.tight_layout()
    hyp_val_img = mo.image(du.fig_to_png(_fig), width="100%")
    plt.close(_fig)

    return (hyp_val_img,)


# ─── Hyperbolic tab assembly ──────────────────────────────────────────────────

@app.cell
def _hyp_tab(
    mo,
    hyp_algorithm, hyp_k, hyp_distortion_toggle, hyp_seed,
    hyp_disagree_txt, hyp_is_synthetic, hyp_label,
    hyp_maps_img, hyp_val_img,
    hyp_distortion_img, hyp_pct,
):
    _header = mo.md(
        f"## 🔵 Hyperbolic — Poincaré Disk Metric\n\n"
        f"*Hierarchical data needs space that expands exponentially.*\n\n"
        f"**Data:** {hyp_label}"
    )

    _info = mo.callout(mo.md(
        "**Simulated hierarchical taxonomy** — nodes in a synthetic tree embedded in the Poincaré disk. "
        "**Central points** (near origin) represent high-level categories (kingdom, phylum). "
        "**Boundary points** (near the unit circle edge) are specific instances (species, subspecies). "
        "Hyperbolic space expands exponentially away from the origin, naturally fitting the branching "
        "structure of trees — the same number of nodes at each depth ring, but exponentially more space. "
        "Drag the **seed slider** to generate a different tree topology."
    ), kind="info")

    _dist_table = mo.md("""
### Distance Table: Euclidean Radius → Hyperbolic Distance

| Euclidean radius | Hyperbolic distance from origin |
|:---:|:---:|
| 0.50 | 1.10 |
| 0.80 | 2.20 |
| 0.90 | 2.94 |
| 0.95 | 3.66 |
| 0.99 | 5.29 |
""")

    _insight = mo.callout(
        mo.md(
            f"**Key Insight:** In hyperbolic space, points near the disk boundary are "
            f"*exponentially* farther from the center than Euclidean distance suggests. "
            f"Euclidean clustering misassigns **{hyp_pct:.1f}%** of points, collapsing "
            f"the rich hierarchical structure near the boundary into fewer, larger clusters."
        ),
        kind="success",
    )

    _formula = mo.md(r"""
---
#### 📐 Distance Formula

**Poincaré disk distance:**

$$d_{\mathbb{H}}(u, v) = \mathrm{arccosh}\!\left(1 + \frac{2\,\|u - v\|^2}{(1 - \|u\|^2)(1 - \|v\|^2)}\right)$$

Points $u, v$ lie inside the open unit disk $(\|u\|, \|v\| < 1)$.
As $\|u\| \to 1$ the boundary is approached and distances diverge: $d_{\mathbb{H}}(0, u) = 2\,\mathrm{arctanh}(\|u\|)$.
PVE and Silhouette are computed using $d_{\mathbb{H}}$ as the base distance.
""")

    _main_row = mo.Html(
        '<div style="display:grid;grid-template-columns:65% 34%;gap:1rem;align-items:start">'
        f'<div>{mo.vstack([mo.md("**Euclidean vs Hyperbolic**"), hyp_maps_img, hyp_disagree_txt], gap="0.4rem").text}</div>'
        f'<div>{mo.vstack([_formula, _dist_table, _info], gap="0.5rem").text}</div>'
        '</div>'
    )
    hyp_content = mo.vstack([
        mo.md(f"## 🔵 Hyperbolic — Poincaré Disk Metric &nbsp;·&nbsp; *{hyp_label}*"),
        mo.hstack([hyp_algorithm, hyp_k, hyp_seed], gap="1.5rem"),
        _main_row,
        mo.hstack([
            mo.vstack([hyp_distortion_toggle, hyp_distortion_img]),
            mo.vstack([mo.md("**Validation Sweep K = 2–10**"), hyp_val_img, _insight]),
        ], gap="1rem"),
    ], gap="0.6rem")

    return (hyp_content,)


# ─── Feature Space: data (load once) ─────────────────────────────────────────

@app.cell
def _feat_data(du, np, os, feat_dataset):
    _DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    _raw, _label, _is_synth = du.load_feature_data(_DATA_DIR, dataset=feat_dataset.value)

    _MAX_N = 300
    _pts_f = _raw['points']
    if len(_pts_f) > _MAX_N:
        _rng = np.random.default_rng(45)
        _idx = np.sort(_rng.choice(len(_pts_f), _MAX_N, replace=False))
        _sp  = _raw['species']
        feat_data = {
            'points':       _pts_f[_idx],
            'points_2d':    _raw['points_2d'][_idx],
            'species':      _sp[_idx] if _sp is not None else None,
            'feature_names': _raw['feature_names'],
        }
        feat_label = _label + f'  (N = {_MAX_N} sub-sampled)'
    else:
        feat_data = _raw
        feat_label = _label

    feat_is_synthetic = _is_synth
    return feat_data, feat_is_synthetic, feat_label


# ─── Feature Space: UI controls ───────────────────────────────────────────────

@app.cell
def _feat_controls(mo, os, du):
    _DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    _candidates = [
        ('Palmer Penguins (3 species)', 'penguins', 'penguins.csv'),
        ('Iris (3 species)',            'iris',     'iris.csv'),
    ]
    _opts = {}
    for _lbl, _key, _fname in _candidates:
        _path = os.path.join(_DATA_DIR, _fname)
        if not os.path.exists(_path):
            try:
                du.download_if_missing(du._FEAT_URLS[_key], _path)
            except Exception:
                pass
        if os.path.exists(_path):
            _opts[_lbl] = _key
    _opts['Synthetic'] = 'synthetic'

    feat_dataset = mo.ui.dropdown(
        options=_opts,
        value=next(iter(_opts)),
        label="Dataset",
    )
    feat_algorithm = mo.ui.radio(
        options=["K-Means", "K-Medoids", "GMM"],
        value="K-Means",
        label="Algorithm",
        inline=True,
    )
    feat_k = mo.ui.slider(start=2, stop=10, value=3, label="Number of clusters  K")
    feat_p = mo.ui.slider(start=0.5, stop=20.0, value=2.0, step=0.5,
                          label="Minkowski p value")
    return feat_dataset, feat_algorithm, feat_k, feat_p


# ─── Feature Space: precompute Minkowski matrix (re-runs on p change) ─────────

@app.cell
def _feat_mink_matrix(du, feat_data, feat_p):
    feat_D_mink = du.compute_minkowski_matrix_full(feat_data['points'], feat_p.value)
    return (feat_D_mink,)


# ─── Feature Space: K-sweep (re-runs on algorithm OR p change) ────────────────

@app.cell
def _feat_sweep(du, mo, np, feat_algorithm, feat_p, feat_data, feat_D_mink):
    _algo = feat_algorithm.value
    _pts  = feat_data['points']
    _p    = feat_p.value

    if _algo == "K-Means":
        def _run(k, D):
            return du.minkowski_kmeans(_pts, k=k, p=_p)
    elif _algo == "K-Medoids":
        def _run(k, D):
            return du.kmedoids_pam(D, k=k)
    else:  # GMM
        def _run(k, D):
            return du.euclidean_gmm(_pts, k=k)

    with mo.status.spinner(f"Computing {_algo} sweep  K = 2–10  (p={_p}) …"):
        feat_sweep = du.sweep_k(_run, feat_D_mink, range(2, 11))

    return (feat_sweep,)


# ─── Feature Space: unit ball gallery (4 canonical balls, current p highlighted)

@app.cell
def _feat_unit_ball(du, mo, np, plt, feat_data, feat_p):
    _p = feat_p.value
    _p_disp = '∞' if _p >= 20 else f'{_p:g}'

    _fig, _axs = plt.subplots(1, 4, figsize=(14, 3.2))
    _fig.patch.set_facecolor(du.BG_FIGURE)

    # Which preset matches the current p?
    def _is_active(preset_p):
        if preset_p == 1:   return _p == 1
        if preset_p == 2:   return _p == 2
        if preset_p == 100: return _p >= 20
        return False

    _BALLS = [
        (1,   '#f39c12', 'L1 — Manhattan'),
        (2,   '#2ecc71', 'L2 — Euclidean'),
        (100, '#9b59b6', 'L∞ — Chebyshev'),
    ]

    for _ax, (pp, _col, _lbl) in zip(_axs[:3], _BALLS):
        du.plot_unit_ball(_ax, p=pp, color=_col)
        _active = _is_active(pp)
        _ax.set_title(_lbl, fontsize=11, fontweight='bold', pad=5,
                      color='#C62828' if _active else '#1A2030')
        if _active:
            for _s in _ax.spines.values():
                _s.set_edgecolor('#C62828'); _s.set_linewidth(3)

    # 4th subplot: Mahalanobis unit ellipse {x : xᵀ Σ⁻¹ x ≤ 1}
    _ax4 = _axs[3]
    _pts_raw = feat_data['points']
    _cov = np.cov(_pts_raw[:, :2].T)
    try:
        _cov_inv = np.linalg.inv(_cov + 1e-8 * np.eye(2))
        _th = np.linspace(0, 2 * np.pi, 400)
        _circ = np.stack([np.cos(_th), np.sin(_th)], axis=1)
        _vals = np.einsum('ij,jk,ik->i', _circ, _cov_inv, _circ)
        _scale = 1.0 / np.sqrt(np.maximum(_vals, 1e-12))
        _ell = _circ * _scale[:, None]
        _ax4.fill(_ell[:, 0], _ell[:, 1], color='#e74c3c', alpha=0.28)
        _ax4.plot(_ell[:, 0], _ell[:, 1], color='#e74c3c', lw=2.5)
    except np.linalg.LinAlgError:
        _ax4.text(0, 0, 'singular', ha='center', va='center', fontsize=10)
    _ax4.set_xlim(-1.6, 1.6); _ax4.set_ylim(-1.6, 1.6)
    _ax4.set_aspect('equal')
    _ax4.axhline(0, color='#aaaaaa', lw=0.6)
    _ax4.axvline(0, color='#aaaaaa', lw=0.6)
    _ax4.grid(True, alpha=0.25); _ax4.tick_params(labelsize=8)
    _ax4.set_title('Mahalanobis', fontsize=11, fontweight='bold', pad=5)

    _fig.suptitle(
        f'Unit Balls  ·  current slider: p = {_p_disp}  '
        f'{"← highlighted above" if any(_is_active(pp) for pp, *_ in _BALLS) else "(between presets)"}',
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    feat_unit_ball_img = mo.image(du.fig_to_png(_fig), width="100%")
    plt.close(_fig)
    return (feat_unit_ball_img,)


# ─── Feature Space: Euclidean vs current-p Voronoi comparison ────────────────

@app.cell
def _feat_mink_plot(du, mo, np, plt, feat_algorithm, feat_k, feat_p, feat_data, feat_D_mink):
    from matplotlib.colors import ListedColormap as _LCM

    _algo  = feat_algorithm.value
    _k     = feat_k.value
    _p     = feat_p.value
    _pts   = feat_data['points']
    _pts2  = feat_data['points_2d']
    _p_disp = '∞' if _p >= 20 else f'{_p:g}'

    # ── Consistent PCA basis ──────────────────────────────────────────────────
    _mean_v   = _pts.mean(axis=0)
    _centered = _pts - _mean_v
    _, _, _Vt = np.linalg.svd(_centered, full_matrices=False)

    def _proj(centers):
        return (centers - _mean_v) @ _Vt[:2].T

    # ── Euclidean reference: always K-Means p=2 ───────────────────────────────
    _e_labels, _e_centers, _ = du.minkowski_kmeans(_pts, k=_k, p=2, seed=42)
    _e_c2 = _proj(_e_centers)

    # ── Current algorithm + current p ─────────────────────────────────────────
    if _algo == "K-Means":
        _c_labels, _c_centers, _ = du.minkowski_kmeans(_pts, k=_k, p=_p, seed=42)
        _c_c2 = _proj(_c_centers)
    elif _algo == "K-Medoids":
        _c_labels, _med_idx, _ = du.kmedoids_pam(feat_D_mink, k=_k, seed=42)
        _c_c2 = _proj(_pts[_med_idx])
    else:
        _c_labels, _c_centers, _ = du.euclidean_gmm(_pts, k=_k, seed=42)
        _c_c2 = _proj(_c_centers)

    # ── Voronoi shading helper ─────────────────────────────────────────────────
    _xl = _pts2[:, 0].min() - 0.5;  _xh = _pts2[:, 0].max() + 0.5
    _yl = _pts2[:, 1].min() - 0.5;  _yh = _pts2[:, 1].max() + 0.5
    _res = 180

    def _shade(ax, centers_2d, p_val, labels, title):
        _xx, _yy = np.meshgrid(np.linspace(_xl, _xh, _res),
                                np.linspace(_yl, _yh, _res))
        _grid = np.column_stack([_xx.ravel(), _yy.ravel()])
        _Dg = np.zeros((_grid.shape[0], _k))
        for _j, _c in enumerate(centers_2d):
            _diff = np.abs(_grid - _c)
            _Dg[:, _j] = (_diff.max(axis=1) if p_val >= 20
                          else np.sum(_diff ** p_val, axis=1) ** (1.0 / p_val))
        _rgn = np.argmin(_Dg, axis=1).reshape(_xx.shape)
        _cm = _LCM([du.CLUSTER_COLORS[i % len(du.CLUSTER_COLORS)] for i in range(_k)])
        ax.contourf(_xx, _yy, _rgn, levels=np.arange(-0.5, _k), cmap=_cm, alpha=0.18)
        ax.contour(_xx, _yy, _rgn, levels=np.arange(0.5, _k),
                   colors='#4A5568', linewidths=0.9)
        for _c in range(_k):
            _m = labels == _c
            ax.scatter(_pts2[_m, 0], _pts2[_m, 1],
                       c=du.CLUSTER_COLORS[_c % len(du.CLUSTER_COLORS)],
                       s=22, alpha=0.80, linewidths=0, zorder=3)
        ax.scatter(centers_2d[:, 0], centers_2d[:, 1],
                   c='white', s=150, marker='*',
                   edgecolors='#1A2030', linewidths=0.8, zorder=5)
        ax.set_xlim(_xl, _xh); ax.set_ylim(_yl, _yh)
        ax.set_xlabel('PC 1', fontsize=11); ax.set_ylabel('PC 2', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=6)
        ax.grid(True, alpha=0.22); ax.tick_params(labelsize=9)

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    _fig.patch.set_facecolor(du.BG_FIGURE)
    _shade(_ax1, _e_c2,  2,  _e_labels, 'Euclidean  (p = 2)  ·  K-Means')
    _shade(_ax2, _c_c2, _p, _c_labels, f'Minkowski  (p = {_p_disp})  ·  {_algo}')

    plt.tight_layout()
    feat_mink_img = mo.image(du.fig_to_png(_fig), width="100%")
    plt.close(_fig)
    return (feat_mink_img,)


# ─── Feature Space: Mahalanobis unwarping ─────────────────────────────────────

@app.cell
def _feat_mahal(du, mo, np, plt, feat_k, feat_data):
    _k   = feat_k.value
    _pts = feat_data['points']

    # Mahalanobis clustering on full feature space
    _labels, _centers, _ = du.mahalanobis_kmeans(_pts, k=_k)

    # Project to 2D PCA for the "original" view
    _mean_v = _pts.mean(axis=0)
    _centered = _pts - _mean_v
    _, _, _Vt = np.linalg.svd(_centered, full_matrices=False)
    _pts2 = _centered @ _Vt[:2].T
    _c2   = (_centers - _mean_v) @ _Vt[:2].T

    # Whitening via eigendecomposition of full covariance
    _cov = np.cov(_pts.T)
    _W = np.eye(_pts.shape[1])           # default: identity (no whitening)
    try:
        _eigvals, _eigvecs = np.linalg.eigh(_cov)
        _eigvals = np.maximum(_eigvals, 1e-10)
        _W = _eigvecs @ np.diag(1.0 / np.sqrt(_eigvals)) @ _eigvecs.T
    except np.linalg.LinAlgError:
        pass
    _whitened = _centered @ _W.T

    # Centers in whitened space
    _c_white = (_centers - _mean_v) @ _W.T

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    _fig.patch.set_facecolor(du.BG_FIGURE)

    # LEFT: original PCA space with covariance ellipses
    du.plot_feature_scatter_2d(_ax1, _pts2, _labels, _c2,
                               f'Original Feature Space (PCA 2D, K = {_k})', p=2)
    # Overlay per-cluster covariance ellipses
    import matplotlib.patches as _mp3
    for _c in range(_k):
        _mask = _labels == _c
        if _mask.sum() < 3:
            continue
        _sub = _pts2[_mask]
        _cv  = np.cov(_sub.T)
        if _cv.ndim < 2:
            continue
        _ev, _evec = np.linalg.eigh(_cv)
        _ev  = np.maximum(_ev, 1e-10)
        _ang = np.degrees(np.arctan2(_evec[1, 1], _evec[0, 1]))
        _ell = _mp3.Ellipse(
            xy=_sub.mean(axis=0),
            width=2 * np.sqrt(_ev[1]) * 2,
            height=2 * np.sqrt(_ev[0]) * 2,
            angle=_ang,
            fill=False,
            edgecolor=du.CLUSTER_COLORS[_c % len(du.CLUSTER_COLORS)],
            lw=2, alpha=0.85, zorder=4,
        )
        _ax1.add_patch(_ell)

    # RIGHT: whitened space (only first 2 dims)
    _w2 = _whitened[:, :2]
    _cw2 = _c_white[:, :2]
    du.plot_feature_scatter_2d(_ax2, _w2, _labels, _cw2,
                               f'Mahalanobis-Whitened Space (K = {_k})', p=2)
    _ax2.set_xlabel('Whitened dim 1', fontsize=12)
    _ax2.set_ylabel('Whitened dim 2', fontsize=12)

    plt.tight_layout()
    feat_mahal_img = mo.image(du.fig_to_png(_fig), width="100%")
    plt.close(_fig)
    return (feat_mahal_img,)


# ─── Feature Space: validation curves ────────────────────────────────────────

@app.cell
def _feat_val_plots(du, mo, plt, feat_k, feat_sweep):
    _k      = feat_k.value
    _k_vals = feat_sweep['k_values']
    _pve    = feat_sweep['pve']
    _sil    = feat_sweep['silhouette_mean']

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    _fig.patch.set_facecolor(du.BG_FIGURE)
    du.plot_pve_curve(_ax1, _k_vals, _pve, _k)
    du.plot_silhouette_curve(_ax2, _k_vals, _sil, _k)
    plt.tight_layout()
    feat_val_img = mo.image(du.fig_to_png(_fig), width="100%")
    plt.close(_fig)
    return (feat_val_img,)


# ─── Feature Space: tab assembly ──────────────────────────────────────────────

@app.cell
def _feat_tab(
    mo,
    feat_dataset, feat_algorithm, feat_k, feat_p,
    feat_is_synthetic, feat_label,
    feat_unit_ball_img, feat_mink_img, feat_mahal_img,
    feat_val_img,
):
    _header = mo.md(
        f"## 📏 Feature Space — Minkowski & Mahalanobis\n\n"
        f"*Correlated features need distance that knows the data's shape.*\n\n"
        f"**Data:** {feat_label}"
    )
    _FEAT_DESC = {
        'penguins': "**Palmer Station penguins (2007–2009)** — 344 penguins across 3 species (Adelie, Chinstrap, Gentoo), measured on 4 features: bill length, bill depth, flipper length, body mass. Species overlap in Euclidean space but become cleanly separable once feature correlation is accounted for via Mahalanobis distance.",
        'iris':     "**Fisher's Iris dataset (1936)** — 150 iris flowers from 3 species (setosa, versicolor, virginica), each with 4 measurements. A classic benchmark: setosa is linearly separable from the other two; versicolor and virginica overlap, making them sensitive to choice of distance metric.",
        'synthetic': "**Synthetically generated** — Gaussian clusters in standardised feature space. Useful for exploring how p (the Minkowski exponent) changes cluster shape.",
    }
    _ds_key = feat_dataset.value if feat_dataset.value in _FEAT_DESC else 'synthetic'
    _data_info = mo.callout(mo.md(_FEAT_DESC[_ds_key]), kind="info")

    _warning = (
        mo.callout(mo.md("⚠️ **Synthetic data** — penguins.csv not found."), kind="warn")
        if feat_is_synthetic else mo.md("")
    )

    _insight = mo.callout(
        mo.md(
            "**Key Insight:** Mahalanobis distance accounts for feature correlations by "
            "whitening the space — clusters that appear elongated under Euclidean distance "
            "become compact spheres. Minkowski p controls the 'shape' of closeness: "
            "p=1 is robust to outliers, p=2 is the familiar Euclidean, p=∞ focuses on the "
            "single largest feature difference."
        ),
        kind="success",
    )

    _formula = mo.md(r"""
---
#### 📐 Distance Formulas

**Minkowski $L^p$ distance:**

$$d_p(u, v) = \left(\sum_{j=1}^{D} |u_j - v_j|^p\right)^{1/p}$$

Special cases: $p=1$ Manhattan · $p=2$ Euclidean · $p\to\infty$ Chebyshev $\bigl(\max_j |u_j - v_j|\bigr)$.

**Mahalanobis distance:**

$$d_M(u, v) = \sqrt{(u - v)^\top \Sigma^{-1} (u - v)}$$

where $\Sigma$ is the sample covariance matrix — equivalent to Euclidean after whitening $\tilde{x} = \Sigma^{-1/2}x$.
PVE and Silhouette are computed using the selected Minkowski distance.
""")

    _top_row = mo.Html(
        '<div style="display:grid;grid-template-columns:65% 34%;gap:1rem;align-items:start">'
        f'<div>{mo.vstack([mo.md("**Unit Ball Gallery** — highlighted ball matches current p"), feat_unit_ball_img], gap="0.3rem").text}</div>'
        f'<div>{mo.vstack([_formula, _data_info], gap="0.5rem").text}</div>'
        '</div>'
    )
    feat_content = mo.vstack([
        mo.md(f"## 📏 Feature Space — Minkowski & Mahalanobis &nbsp;·&nbsp; *{feat_label}*"),
        _warning,
        mo.hstack([feat_dataset, feat_algorithm, feat_k, feat_p], gap="1.5rem"),
        _top_row,
        mo.vstack([mo.md("**Euclidean (p=2) vs Minkowski (current p) — same data, same K**"),
                   feat_mink_img], gap="0.3rem"),
        mo.vstack([mo.md("**Mahalanobis Unwarping — original space vs whitened space**"),
                   feat_mahal_img], gap="0.3rem"),
        mo.Html(
            '<div style="display:grid;grid-template-columns:62% 37%;gap:1rem;align-items:start">'
            f'<div>{mo.vstack([mo.md("**Validation Sweep K = 2–10**"), feat_val_img], gap="0.3rem").text}</div>'
            f'<div>{_insight.text}</div>'
            '</div>'
        ),
    ], gap="0.6rem")
    return (feat_content,)


# ─── Mixed Type: data (load once) ────────────────────────────────────────────

@app.cell
def _mixed_data(du, np, os):
    _DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    _raw, _label, _is_synth = du.load_mixed_data(_DATA_DIR)

    _MAX_N = 200
    _n = len(_raw['lats'])
    if _n > _MAX_N:
        _rng = np.random.default_rng(46)
        _idx = np.sort(_rng.choice(_n, _MAX_N, replace=False))
        mixed_data = {
            'lats':            _raw['lats'][_idx],
            'lons':            _raw['lons'][_idx],
            'magnitude':       _raw['magnitude'][_idx] if _raw['magnitude'] is not None else None,
            'magnitude_class': _raw['magnitude_class'][_idx],
            'depth_class':     _raw['depth_class'][_idx],
            'true_labels':     None,
        }
        mixed_label = _label + f'  (N = {_MAX_N} sub-sampled)'
    else:
        mixed_data  = _raw
        mixed_label = _label

    mixed_is_synthetic = _is_synth
    return mixed_data, mixed_is_synthetic, mixed_label


# ─── Mixed Type: outlier state ────────────────────────────────────────────────

@app.cell
def _mixed_state(mo):
    outlier_state, set_outlier_state = mo.state(False)
    return outlier_state, set_outlier_state


# ─── Mixed Type: UI controls ──────────────────────────────────────────────────

@app.cell
def _mixed_controls(mo, set_outlier_state):
    mixed_k = mo.ui.slider(start=2, stop=10, value=3, label="Number of clusters  K")
    mixed_geo_weight = mo.ui.slider(
        start=0.0, stop=5.0, value=1.0, step=0.1,
        label="Geographic weight  (w_geo)",
    )
    mixed_add_outliers = mo.ui.button(
        label="💥 Add 15 Outliers",
        on_change=lambda _: set_outlier_state(True),
    )
    mixed_reset_outliers = mo.ui.button(
        label="🔄 Reset",
        on_change=lambda _: set_outlier_state(False),
    )
    return mixed_add_outliers, mixed_geo_weight, mixed_k, mixed_reset_outliers


# ─── Mixed Type: Gower distance matrices (re-runs on geo_weight change) ────────

@app.cell
def _mixed_gower_matrices(du, mixed_data, mixed_geo_weight):
    _w = mixed_geo_weight.value
    mixed_D_geo  = du.gower_distance_matrix_v2(mixed_data, w_geo=1.0, w_cat=0.0)
    mixed_D_cat  = du.gower_distance_matrix_v2(mixed_data, w_geo=0.0, w_cat=1.0)
    mixed_D_blend = du.gower_distance_matrix_v2(mixed_data, w_geo=_w,  w_cat=1.0)
    return mixed_D_blend, mixed_D_cat, mixed_D_geo


# ─── Mixed Type: Gower K-sweep (re-runs on geo_weight or k change) ────────────

@app.cell
def _mixed_sweep(du, mo, mixed_k, mixed_D_blend):
    def _run(k, D):
        return du.kmedoids_pam(D, k=k)

    with mo.status.spinner("Computing K-Medoids sweep (Gower)  K = 2–10 …"):
        mixed_sweep = du.sweep_k(_run, mixed_D_blend, range(2, 11))

    return (mixed_sweep,)


# ─── Mixed Type: Gower ablation maps ─────────────────────────────────────────

@app.cell
def _mixed_ablation(
    du, mo, np, plt,
    mixed_k, mixed_geo_weight, mixed_data,
    mixed_D_geo, mixed_D_cat, mixed_D_blend,
):
    _k   = mixed_k.value
    _w   = mixed_geo_weight.value
    _lats = mixed_data['lats']
    _lons = mixed_data['lons']
    _dep_cl = mixed_data['depth_class']

    _cat_markers = {'shallow': 'o', 'intermediate': '^', 'deep': 's'}

    _lg, _gmed, _ = du.kmedoids_pam(mixed_D_geo,   k=_k)
    _lc, _cmed, _ = du.kmedoids_pam(mixed_D_cat,   k=_k)
    _lb, _bmed, _ = du.kmedoids_pam(mixed_D_blend,  k=_k)

    _fig, (_a1, _a2, _a3) = plt.subplots(1, 3, figsize=(14, 4))
    _fig.patch.set_facecolor(du.BG_FIGURE)

    du.plot_geo_map(_a1, _lats, _lons, _lg, _gmed,
                    f'Geography only  (w_geo=1, w_cat=0)  K = {_k}',
                    cat_values=_dep_cl, cat_markers=_cat_markers)
    du.plot_geo_map(_a2, _lats, _lons, _lc, _cmed,
                    f'Categories only  (w_geo=0, w_cat=1)  K = {_k}',
                    cat_values=_dep_cl, cat_markers=_cat_markers)
    du.plot_geo_map(_a3, _lats, _lons, _lb, _bmed,
                    f'Blended Gower  (w_geo={_w:.1f}, w_cat=1)  K = {_k}',
                    cat_values=_dep_cl, cat_markers=_cat_markers)

    from matplotlib.lines import Line2D as _L2D
    _leg = [_L2D([0],[0], marker=m, color='k', markerfacecolor='#5C6BC0',
                 markersize=9, linestyle='None', label=c)
            for c, m in _cat_markers.items()]
    _a3.legend(handles=_leg, fontsize=10, title='Depth class',
               title_fontsize=10, loc='lower right')

    plt.tight_layout()
    mixed_ablation_img = mo.image(du.fig_to_png(_fig), width="100%")
    plt.close(_fig)
    return (mixed_ablation_img,)


# ─── Mixed Type: outlier robustness (K-Means vs K-Medoids on Haversine) ────────

@app.cell
def _mixed_outlier_robustness(du, mo, np, plt, mixed_k, mixed_data, outlier_state):
    _k    = mixed_k.value
    _lats = mixed_data['lats'].copy()
    _lons = mixed_data['lons'].copy()
    _n    = len(_lats)
    _with_outliers = outlier_state()

    _outlier_mask = np.zeros(_n + 15 if _with_outliers else _n, dtype=bool)

    if _with_outliers:
        _rng = np.random.default_rng(2025)
        _o_lats = _rng.uniform(-np.pi / 2, np.pi / 2, 15)
        _o_lons = _rng.uniform(-np.pi, np.pi, 15)
        _lats_aug = np.concatenate([_lats, _o_lats])
        _lons_aug = np.concatenate([_lons, _o_lons])
        _outlier_mask[-15:] = True
    else:
        _lats_aug = _lats
        _lons_aug = _lons

    _n_aug = len(_lats_aug)

    # K-Means on sphere (Haversine)
    _km_labels, _km_centers, _ = du.spherical_kmeans(_lats_aug, _lons_aug, k=_k)
    _km_clats, _km_clons = _km_centers[:, 0], _km_centers[:, 1]

    # K-Medoids on sphere (Haversine)
    _hD = du.compute_haversine_matrix(_lats_aug, _lons_aug)
    _med_labels, _med_idx, _ = du.kmedoids_pam(_hD, k=_k)
    _med_clats = _lats_aug[_med_idx]
    _med_clons = _lons_aug[_med_idx]

    # Centroid shift (vs clean run)
    _shift_txt = ""
    if _with_outliers:
        _km_clean, _km_cc, _ = du.spherical_kmeans(_lats, _lons, k=_k)
        _hD_clean = du.compute_haversine_matrix(_lats, _lons)
        _med_clean, _mc_idx, _ = du.kmedoids_pam(_hD_clean, k=_k)
        _mc_clats = _lats[_mc_idx]; _mc_clons = _lons[_mc_idx]

        def _mean_shift(c_clean, c_aug):
            _cl, _cr = c_clean[:, 0], c_clean[:, 1]
            _al, _ar = c_aug[:, 0], c_aug[:, 1]
            _shifts = []
            for _j in range(_k):
                _d = du.haversine(_cl[_j], _cr[_j], _al, _ar)
                _shifts.append(float(np.min(_d)))
            return float(np.mean(_shifts))

        _km_sh  = np.degrees(_mean_shift(_km_cc, _km_centers))
        _med_sh = np.degrees(_mean_shift(
            np.stack([_mc_clats, _mc_clons], axis=1),
            np.stack([_med_clats, _med_clons], axis=1),
        ))
        _shift_txt = (
            f"K-Means centroids shifted **{_km_sh:.1f}°** on average — "
            f"K-Medoids shifted **{_med_sh:.1f}°**"
        )

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    _fig.patch.set_facecolor(du.BG_FIGURE)
    _om = _outlier_mask if _with_outliers else None
    du.plot_sphere_map(_ax1, _lats_aug, _lons_aug, _km_labels,
                       _km_clats, _km_clons,
                       f'K-Means (Haversine, K = {_k})',
                       metric='haversine')
    if _om is not None:
        _ax1.scatter(
            np.rad2deg(_lons_aug[_om]), np.rad2deg(_lats_aug[_om]),
            c='#e74c3c', marker='x', s=80, linewidths=2.0, zorder=7, label='Outliers',
        )
        _ax1.legend(fontsize=11)

    du.plot_sphere_map(_ax2, _lats_aug, _lons_aug, _med_labels,
                       _med_clats, _med_clons,
                       f'K-Medoids (Haversine, K = {_k})',
                       metric='haversine')
    if _om is not None:
        _ax2.scatter(
            np.rad2deg(_lons_aug[_om]), np.rad2deg(_lats_aug[_om]),
            c='#e74c3c', marker='x', s=80, linewidths=2.0, zorder=7, label='Outliers',
        )
        _ax2.legend(fontsize=11)

    plt.tight_layout()
    mixed_outlier_img = mo.image(du.fig_to_png(_fig), width="100%")
    plt.close(_fig)

    mixed_shift_txt = mo.callout(
        mo.md(_shift_txt), kind="warn"
    ) if _shift_txt else mo.md("")

    return mixed_outlier_img, mixed_shift_txt


# ─── Mixed Type: validation curves ───────────────────────────────────────────

@app.cell
def _mixed_val_plots(du, mo, plt, mixed_k, mixed_sweep):
    _k      = mixed_k.value
    _k_vals = mixed_sweep['k_values']
    _pve    = mixed_sweep['pve']
    _sil    = mixed_sweep['silhouette_mean']

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    _fig.patch.set_facecolor(du.BG_FIGURE)
    du.plot_pve_curve(_ax1, _k_vals, _pve, _k)
    du.plot_silhouette_curve(_ax2, _k_vals, _sil, _k)
    plt.tight_layout()
    mixed_val_img = mo.image(du.fig_to_png(_fig), width="100%")
    plt.close(_fig)
    return (mixed_val_img,)


# ─── Mixed Type: tab assembly ─────────────────────────────────────────────────

@app.cell
def _mixed_tab(
    mo,
    mixed_k, mixed_geo_weight, mixed_add_outliers, mixed_reset_outliers,
    mixed_is_synthetic, mixed_label,
    mixed_ablation_img, mixed_outlier_img, mixed_shift_txt,
    mixed_val_img,
):
    _header = mo.md(
        f"## 🔀 Mixed Type — Gower Distance\n\n"
        f"*Real data mixes coordinates with categories — Gower unifies them.*\n\n"
        f"**Data:** {mixed_label}"
    )
    _warning = (
        mo.callout(mo.md("⚠️ **Synthetic data** — earthquakes.csv not found."), kind="warn")
        if mixed_is_synthetic else mo.md("")
    )

    _insight = mo.callout(
        mo.md(
            "**Key Insight:** Gower distance blends geographic proximity (Haversine) with "
            "categorical similarity (magnitude class, depth class) into a single normalized "
            "distance matrix. Adjusting the geographic weight shifts cluster boundaries between "
            "'nearby but different' and 'similar but far' groupings. K-Medoids uses actual "
            "data points as cluster representatives, making it more robust to outliers than "
            "K-Means centroids (which can be pulled far from the data by extreme values)."
        ),
        kind="success",
    )

    _formula = mo.md(r"""
---
#### 📐 Distance Formulas

**Gower distance (weighted average over feature types):**

$$d_{\text{Gower}}(i, j) = \frac{w_{\text{geo}} \cdot d_{\text{Haversine}}(i,j)/\pi \;+\; w_{\text{cat}} \sum_k \mathbf{1}[c_k(i) \neq c_k(j)]}{w_{\text{geo}} + n_{\text{cat}} \cdot w_{\text{cat}}}$$

Each term is normalised to $[0,1]$: Haversine divided by $\pi$ (maximum great-circle), categorical as a 0/1 mismatch indicator.

**K-Medoids (PAM):** selects cluster representatives from actual data points,
minimising $\sum_i d(x_i,\, m_{\ell_i})$ over all medoid assignments $m \in \mathcal{X}$.
Unlike K-Means, no centroid computation is required — the algorithm only needs the distance matrix.
""")

    _top_row = mo.Html(
        '<div style="display:grid;grid-template-columns:65% 34%;gap:1rem;align-items:start">'
        f'<div>{mo.vstack([mo.hstack([mixed_k, mixed_geo_weight], gap="1rem"), mo.md("> **K-Medoids** only — Gower produces a distance matrix, so K-Means (which needs centroids) does not apply."), mixed_ablation_img], gap="0.4rem").text}</div>'
        f'<div>{mo.vstack([_formula, _insight], gap="0.5rem").text}</div>'
        '</div>'
    )
    mixed_content = mo.vstack([
        mo.md(f"## 🔀 Mixed Type — Gower Distance &nbsp;·&nbsp; *{mixed_label}*"),
        _warning,
        _top_row,
        mo.hstack([mixed_add_outliers, mixed_reset_outliers], gap="0.5rem"),
        mo.hstack([
            mo.vstack([mo.md("**Outlier Robustness: K-Means vs K-Medoids**"), mixed_outlier_img]),
            mo.vstack([mo.md("**Validation — Gower, K = 2–10**"), mixed_val_img]),
        ], gap="1rem"),
        mixed_shift_txt,
    ], gap="0.6rem")
    return (mixed_content,)


# ─── Lloyd's Algorithm controls ──────────────────────────────────────────────

@app.cell
def _lloyd_controls(mo):
    lloyd_surface = mo.ui.dropdown(
        options={
            'Euclidean (ℝ²)':          'euclidean',
            'Spherical (Haversine)':   'spherical',
            'Torus (Wraparound L²)':   'torus',
            'Hyperbolic (Poincaré)':   'hyperbolic',
        },
        value='Euclidean (ℝ²)',
        label="Surface",
    )
    lloyd_k    = mo.ui.slider(start=2, stop=6, value=3, label="K  (clusters)",  show_value=True)
    lloyd_seed = mo.ui.slider(start=0, stop=9, value=0, label="Seed",            show_value=True)
    return lloyd_k, lloyd_seed, lloyd_surface


# ─── Lloyd's Algorithm animation ─────────────────────────────────────────────

@app.cell
def _lloyd_animation(du, lloyd_k, lloyd_seed, lloyd_surface, mo, np):
    import plotly.graph_objects as _go
    from plotly.subplots import make_subplots as _msp

    _geom = lloyd_surface.value
    _k    = lloyd_k.value
    _seed = lloyd_seed.value
    _N    = 84           # total points (divisible by k up to 6)
    _rng  = np.random.default_rng(_seed * 37 + 17)
    _per  = _N // _k

    # ── Generate geometry-specific synthetic data ─────────────────────────────
    _pts_list = []

    if _geom == 'euclidean':
        for _c in range(_k):
            _cx, _cy = _rng.uniform(-2.0, 2.0, 2)
            _pts_list.append(np.column_stack([
                _cx + _rng.normal(0, 0.42, _per),
                _cy + _rng.normal(0, 0.42, _per),
            ]))

    elif _geom == 'spherical':
        for _c in range(_k):
            _clat = _rng.uniform(-1.0, 1.0)
            _clon = _rng.uniform(-2.6, 2.6)
            _lats = np.clip(_clat + _rng.normal(0, 0.18, _per), -1.52, 1.52)
            _lons = np.clip(_clon + _rng.normal(0, 0.18, _per), -3.08, 3.08)
            _pts_list.append(np.column_stack([_lats, _lons]))

    elif _geom == 'torus':
        _TP = 2 * np.pi
        for _c in range(_k):
            _cphi = _rng.uniform(0, _TP)
            _cpsi = _rng.uniform(0, _TP)
            _phi  = (_cphi + _rng.normal(0, 0.35, _per)) % _TP
            _psi  = (_cpsi + _rng.normal(0, 0.35, _per)) % _TP
            _pts_list.append(np.column_stack([_phi, _psi]))

    elif _geom == 'hyperbolic':
        for _c in range(_k):
            _th = _rng.uniform(0, 2 * np.pi)
            _r  = _rng.uniform(0.15, 0.55)
            _cx2, _cy2 = _r * np.cos(_th), _r * np.sin(_th)
            _px = _cx2 + _rng.normal(0, 0.10, _per)
            _py = _cy2 + _rng.normal(0, 0.10, _per)
            _nrm = np.sqrt(_px**2 + _py**2)
            _far = _nrm >= 0.92
            _px[_far] *= 0.88 / _nrm[_far]
            _py[_far] *= 0.88 / _nrm[_far]
            _pts_list.append(np.column_stack([_px, _py]))

    _pts = np.vstack(_pts_list)

    # ── Run Lloyd's ──────────────────────────────────────────────────────────
    _steps = du.lloyd_steps(_pts, _k, geometry=_geom, seed=_seed, max_iter=14)

    # ── Color helpers ─────────────────────────────────────────────────────────
    _COLS = du.CLUSTER_COLORS
    _GRAY = '#888888'

    def _pt_colors(labels):
        if labels[0] == -1:
            return [_GRAY] * len(labels)
        return [_COLS[int(l) % len(_COLS)] for l in labels]

    def _ctr_colors():
        return [_COLS[c % len(_COLS)] for c in range(_k)]

    # ── Display-coordinate transforms ─────────────────────────────────────────
    if _geom == 'spherical':
        _px_d = np.rad2deg(_pts[:, 1])   # lon → x
        _py_d = np.rad2deg(_pts[:, 0])   # lat → y
        def _cx_d(c): return np.rad2deg(c[:, 1])
        def _cy_d(c): return np.rad2deg(c[:, 0])
        _xlab, _ylab = 'Longitude (°)', 'Latitude (°)'
    else:
        _px_d = _pts[:, 0]
        _py_d = _pts[:, 1]
        def _cx_d(c): return c[:, 0]
        def _cy_d(c): return c[:, 1]
        _xlab = {'euclidean': 'Feature 1', 'torus': 'φ  (rad)', 'hyperbolic': 'x'}[_geom]
        _ylab = {'euclidean': 'Feature 2', 'torus': 'ψ  (rad)', 'hyperbolic': 'y'}[_geom]

    _title_map = {
        'euclidean':  "Euclidean  ℝ²",
        'spherical':  "Sphere  (Haversine)",
        'torus':      "Flat Torus  (Wraparound)",
        'hyperbolic': "Poincaré Disk  (Hyperbolic)",
    }

    # ── Precompute WCSS range for stable axes ─────────────────────────────────
    _all_wcss  = [s['wcss']      for s in _steps if s['phase'] == 'assign']
    _all_iters = [s['iteration'] for s in _steps if s['phase'] == 'assign']

    # ── Build Plotly figure ───────────────────────────────────────────────────
    _fig = _msp(
        rows=1, cols=2,
        column_widths=[0.60, 0.40],
        subplot_titles=[
            f"Lloyd's K-Means  ·  {_title_map[_geom]}",
            "WCSS Convergence",
        ],
        horizontal_spacing=0.10,
    )

    _s0 = _steps[0]

    # Trace 0 — data points
    _fig.add_trace(_go.Scatter(
        x=_px_d, y=_py_d,
        mode='markers',
        marker=dict(color=_pt_colors(_s0['labels']), size=8, opacity=0.85,
                    line=dict(width=0)),
        name='Points', showlegend=False,
    ), row=1, col=1)

    # Trace 1 — centroids
    _fig.add_trace(_go.Scatter(
        x=_cx_d(_s0['centers']), y=_cy_d(_s0['centers']),
        mode='markers',
        marker=dict(symbol='star', size=22, color=_ctr_colors(),
                    line=dict(color='white', width=1.5)),
        name='Centroids', showlegend=False,
    ), row=1, col=1)

    # Trace 2 — WCSS progress line (starts empty)
    _fig.add_trace(_go.Scatter(
        x=[], y=[],
        mode='lines+markers',
        line=dict(color='#e74c3c', width=2.5),
        marker=dict(size=7, color='#e74c3c'),
        showlegend=False,
    ), row=1, col=2)

    # ── Capture subplot title annotations (to re-use in each frame) ───────────
    _base_anns = list(_fig.layout.annotations)

    # ── Build animation frames ────────────────────────────────────────────────
    _frames     = []
    _sldr_steps = []
    _wcss_x, _wcss_y = [], []

    _PHASE_LABEL = {
        'init':   '🎲 Initialisation — K centroids seeded at random points',
        'assign': '📍 Assignment — each point → nearest centroid',
        'update': '🔄 Update — each centroid → mean of its cluster',
    }

    for _i, _s in enumerate(_steps):
        _phase = _s['phase']
        _it    = _s['iteration']

        # Accumulate WCSS only on assign phases
        _wx, _wy = list(_wcss_x), list(_wcss_y)
        if _phase == 'assign' and not np.isinf(_s['wcss']):
            _wx = _wcss_x + [float(_it)]
            _wy = _wcss_y + [float(_s['wcss'])]
            _wcss_x, _wcss_y = _wx, _wy

        # Slider label
        if _phase == 'init':
            _lbl = 'Init'
        elif _phase == 'assign':
            _lbl = f'A{_it}'
        else:
            _lbl = f'U{_it}'

        # Phase description annotation
        _desc_ann = dict(
            text=f"<i>{_PHASE_LABEL[_phase]}</i>",
            xref='paper', yref='paper',
            x=0.30, y=-0.11,
            showarrow=False,
            font=dict(size=12, color='#c8d0e0'),
            align='center',
        )
        _frame_anns = _base_anns + [_desc_ann]

        _frame = _go.Frame(
            data=[
                # trace 0 — points (only color changes)
                _go.Scatter(
                    x=_px_d, y=_py_d,
                    mode='markers',
                    marker=dict(color=_pt_colors(_s['labels']), size=8,
                                opacity=0.85, line=dict(width=0)),
                ),
                # trace 1 — centroids (position changes)
                _go.Scatter(
                    x=_cx_d(_s['centers']), y=_cy_d(_s['centers']),
                    mode='markers',
                    marker=dict(symbol='star', size=22, color=_ctr_colors(),
                                line=dict(color='white', width=1.5)),
                ),
                # trace 2 — WCSS line (accumulates)
                _go.Scatter(
                    x=_wx, y=_wy,
                    mode='lines+markers',
                    line=dict(color='#e74c3c', width=2.5),
                    marker=dict(size=7, color='#e74c3c'),
                ),
            ],
            traces=[0, 1, 2],
            name=str(_i),
            layout=_go.Layout(annotations=_frame_anns),
        )
        _frames.append(_frame)

        _sldr_steps.append(dict(
            method='animate',
            args=[[str(_i)], {'mode': 'immediate',
                              'frame': {'duration': 0, 'redraw': True}}],
            label=_lbl,
        ))

    _fig.frames = _frames

    # ── Dark theme + axis styling ─────────────────────────────────────────────
    _BG   = '#1a1a2e'
    _PBGC = '#16213e'
    _GRD  = 'rgba(255,255,255,0.08)'
    _TXT  = '#d8dce8'

    _fig.update_layout(
        height=490,
        paper_bgcolor=_BG,
        plot_bgcolor=_PBGC,
        font=dict(color=_TXT, size=12),
        margin=dict(l=50, r=25, t=70, b=105),

        xaxis  =dict(title=_xlab, gridcolor=_GRD, zeroline=False, title_font_size=13),
        yaxis  =dict(title=_ylab, gridcolor=_GRD, zeroline=False, title_font_size=13),
        xaxis2 =dict(title='Iteration', gridcolor=_GRD, zeroline=False,
                     title_font_size=13, dtick=1,
                     range=[0.5, (max(_all_iters) + 1) if _all_iters else 2]),
        yaxis2 =dict(title='WCSS', gridcolor=_GRD, zeroline=False,
                     title_font_size=13,
                     range=[(min(_all_wcss) * 0.88) if _all_wcss else 0,
                            (max(_all_wcss) * 1.10) if _all_wcss else 1]),

        updatemenus=[dict(
            type='buttons', showactive=False,
            x=0.01, y=-0.18, xanchor='left', yanchor='top',
            direction='left',
            buttons=[
                dict(label='▶  Play', method='animate',
                     args=[None, {'frame': {'duration': 850, 'redraw': True},
                                  'fromcurrent': True,
                                  'transition': {'duration': 200, 'easing': 'linear'}}]),
                dict(label='⏸  Pause', method='animate',
                     args=[[None], {'frame': {'duration': 0}, 'mode': 'immediate'}]),
            ],
            bgcolor=_PBGC, bordercolor=_GRD,
            font=dict(color=_TXT, size=13),
        )],

        sliders=[dict(
            active=0,
            steps=_sldr_steps,
            x=0.12, len=0.82,
            y=-0.06,
            currentvalue=dict(prefix='Step: ', font=dict(size=12, color=_TXT),
                              visible=True, xanchor='center'),
            pad=dict(t=8),
            bgcolor=_PBGC, bordercolor=_GRD,
            tickcolor=_TXT,
            font=dict(color=_TXT, size=9),
        )],
    )

    # Fix subplot title font color
    for _ann in _fig.layout.annotations:
        _ann.font.color = '#a0a8b8'
        _ann.font.size  = 14

    # Hyperbolic: draw Poincaré disk boundary + lock aspect ratio
    if _geom == 'hyperbolic':
        _th_c = np.linspace(0, 2 * np.pi, 300)
        _fig.add_trace(_go.Scatter(
            x=np.cos(_th_c), y=np.sin(_th_c),
            mode='lines',
            line=dict(color='rgba(255,255,255,0.25)', width=1.5, dash='dot'),
            showlegend=False,
        ), row=1, col=1)
        _fig.update_xaxes(range=[-1.18, 1.18], row=1, col=1)
        _fig.update_yaxes(range=[-1.18, 1.18], row=1, col=1,
                          scaleanchor='x', scaleratio=1)

    # Torus: mark the 2π wrap lines for context
    if _geom == 'torus':
        _fig.update_xaxes(range=[0, 2 * np.pi], row=1, col=1)
        _fig.update_yaxes(range=[0, 2 * np.pi], row=1, col=1)

    # ── Render via mo.ui.plotly (no plotly.js bundle needed — marimo loads it) ──
    lloyd_anim = mo.ui.plotly(_fig)

    # ── Formula callout (geometry-specific) ───────────────────────────────────
    _MATH = {
        'euclidean': r"""**Distance** &nbsp; $d(x,y) = \|x - y\|_2$

**Assignment** &nbsp; $\hat c(x) = \operatorname{argmin}_c\, d(x, \mu_c)$

**Centroid update** &nbsp; $\mu_c \leftarrow \dfrac{1}{|C_c|}\displaystyle\sum_{x \in C_c} x$

Each iteration monotonically decreases WCSS; convergence is guaranteed but the limit may be a local minimum.""",

        'spherical': r"""**Distance** &nbsp; $d_{\text{Hav}}(p,q) = 2R\arcsin\!\sqrt{\sin^2\!\tfrac{\Delta\phi}{2} + \cos\phi_p\cos\phi_q\sin^2\!\tfrac{\Delta\lambda}{2}}$

**Centroid update** — embed in $\mathbb{R}^3$ via $(\cos\phi\cos\lambda,\,\cos\phi\sin\lambda,\,\sin\phi)$,
take the arithmetic mean, then **project back** onto the sphere (renormalise to unit length).

Arithmetic mean in lat/lon is geometrically incorrect — it distorts near the poles and across the antimeridian.""",

        'torus': r"""**Distance** &nbsp; $d(p,q) = \sqrt{\displaystyle\sum_i \min(|\delta_i|,\,2\pi - |\delta_i|)^2}$ &nbsp; (wraparound $L^2$)

**Centroid update** — circular mean for each angle:
$\bar\theta = \operatorname{atan2}\!\Bigl(\tfrac{1}{n}\textstyle\sum\sin\theta_i,\;\tfrac{1}{n}\textstyle\sum\cos\theta_i\Bigr)$

Plain arithmetic mean fails at the $0\,/\,2\pi$ boundary — a cluster straddling the seam collapses to the wrong side.""",

        'hyperbolic': r"""**Distance** &nbsp; $d(x,y) = \operatorname{arccosh}\!\!\left(1 + \dfrac{2\,\|x-y\|^2}{(1-\|x\|^2)(1-\|y\|^2)}\right)$

**Centroid update** — *Einstein (Lorentz-weighted) midpoint*:
$\bar x = \dfrac{\displaystyle\sum_i \gamma_i\, x_i}{\displaystyle\sum_i \gamma_i}$, &nbsp; $\gamma_i = \bigl(1 - \|x_i\|^2\bigr)^{-1/2}$

Points near the disk boundary are exponentially far apart; arithmetic means drift toward $\mathbf{0}$ — only the weighted midpoint respects the geometry.""",
    }

    lloyd_math = mo.callout(mo.md(_MATH[_geom]), kind="info")

    return lloyd_anim, lloyd_math


# ─── Lloyd's tab ─────────────────────────────────────────────────────────────

@app.cell
def _lloyd_tab(lloyd_anim, lloyd_k, lloyd_math, lloyd_seed, lloyd_surface, mo):
    _ctrl_row = mo.hstack(
        [lloyd_surface, lloyd_k, lloyd_seed], gap="1.5rem"
    )

    _how_it_works = mo.md(
        "**Lloyd's Algorithm** alternates two steps until the cluster assignments stop changing:  \n"
        "1. **Assign** — label every point with the index of its nearest centroid (using the geometry's distance).  \n"
        "2. **Update** — move each centroid to the *mean* of its assigned points (respecting the geometry).  \n\n"
        "Use the slider or ▶ Play to step through each iteration. "
        "Watch how WCSS (within-cluster sum of squares) drops — the steepest fall happens in the first few iterations."
    )

    lloyd_content = mo.vstack([
        mo.md("## 🔄 Lloyd's Algorithm — Step-by-Step Animation"),
        _how_it_works,
        _ctrl_row,
        lloyd_anim,
        lloyd_math,
    ], gap="0.6rem")

    return (lloyd_content,)


# ─── Algorithm Comparison: controls ──────────────────────────────────────────

@app.cell
def _algo_controls(mo):
    algo_k    = mo.ui.slider(start=2, stop=6, value=3, label="K", show_value=True)
    algo_seed = mo.ui.slider(start=0, stop=9, value=0, label="Seed", show_value=True)
    return algo_k, algo_seed


# ─── Algorithm Comparison: side-by-side ──────────────────────────────────────

@app.cell
def _algo_comparison(du, algo_k, algo_seed, mo, np, plt):
    _k    = algo_k.value
    _seed = algo_seed.value
    _N    = 120

    _pts, _ = du.generate_euclidean_clusters(n_points=_N, k=_k, seed=_seed)
    _pts2 = _pts[:, :2]

    # Distance matrix (Euclidean)
    _D = np.sqrt(((_pts[:, None] - _pts[None, :]) ** 2).sum(-1))

    # Run all three algorithms
    _km_labels, _km_centers, _ = du.minkowski_kmeans(_pts, k=_k, p=2, seed=_seed)
    _kmed_labels, _kmed_medoids, _ = du.kmedoids_pam(_D, k=_k, seed=_seed)
    _gmm_labels, _, _ = du.euclidean_gmm(_pts, k=_k, seed=_seed)

    # Align labels (greedy matching to K-means as reference)
    def _align(ref, other, k):
        from itertools import permutations as _perms
        best_perm, best_score = None, -1
        for perm in _perms(range(k)):
            mapped = np.array([perm[l] for l in other])
            score = np.sum(mapped == ref)
            if score > best_score:
                best_perm, best_score = perm, score
        return np.array([best_perm[l] for l in other])

    _kmed_aligned = _align(_km_labels, _kmed_labels, _k)
    _gmm_aligned  = _align(_km_labels, _gmm_labels, _k)

    _disagree_med = int(np.sum(_km_labels != _kmed_aligned))
    _disagree_gmm = int(np.sum(_km_labels != _gmm_aligned))

    # ── Plot 1×3 ──────────────────────────────────────────────────────────────
    _fig, _axes = plt.subplots(1, 3, figsize=(14, 4.2))
    _fig.patch.set_facecolor(du.BG_FIGURE)

    _configs = [
        ('K-Means (L²)', _km_labels, None),
        (f'K-Medoids  ({_disagree_med} differ)', _kmed_aligned, _km_labels != _kmed_aligned),
        (f'GMM  ({_disagree_gmm} differ)', _gmm_aligned, _km_labels != _gmm_aligned),
    ]

    for _ax, (_title, _labels, _diff_mask) in zip(_axes, _configs):
        for _c in range(_k):
            _m = _labels == _c
            _ax.scatter(_pts2[_m, 0], _pts2[_m, 1],
                        c=du.CLUSTER_COLORS[_c % len(du.CLUSTER_COLORS)],
                        s=28, alpha=0.75, linewidths=0, zorder=3)
        if _diff_mask is not None and _diff_mask.any():
            _ax.scatter(_pts2[_diff_mask, 0], _pts2[_diff_mask, 1],
                        facecolors='none', edgecolors='#e74c3c',
                        s=110, linewidths=1.8, zorder=6, label='Disagree')
            _ax.legend(fontsize=9, loc='upper right')
        _ax.set_title(_title, fontsize=13, fontweight='bold', pad=6)
        _ax.set_xlabel('Feature 1', fontsize=11)
        _ax.set_ylabel('Feature 2', fontsize=11)
        _ax.grid(True, alpha=0.25)
        _ax.tick_params(labelsize=9)

    plt.tight_layout()
    algo_compare_img = mo.image(du.fig_to_png(_fig), width="100%")
    plt.close(_fig)
    return (algo_compare_img,)


# ─── Algorithm Comparison: GMM covariance demo ───────────────────────────────

@app.cell
def _algo_covariance(du, algo_seed, mo, np, plt):
    _seed = algo_seed.value
    _rng  = np.random.default_rng(_seed * 11 + 5)

    # Elongated, tilted Gaussian clusters — K-means will fail
    _n = 80
    _angle1 = np.pi / 4    # 45° tilt
    _angle2 = -np.pi / 6   # -30° tilt
    _angle3 = np.pi / 2.5

    def _tilted_blob(n, cx, cy, angle, sx, sy, rng):
        _rot = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle),  np.cos(angle)]])
        _raw = rng.normal(size=(n, 2)) * np.array([sx, sy])
        return (_raw @ _rot.T) + np.array([cx, cy])

    _c1 = _tilted_blob(_n, -2, 0, _angle1, 2.0, 0.3, _rng)
    _c2 = _tilted_blob(_n, 2, 0, _angle2, 1.8, 0.25, _rng)
    _c3 = _tilted_blob(_n, 0, 2.5, _angle3, 1.5, 0.3, _rng)

    _pts = np.vstack([_c1, _c2, _c3])
    _k = 3

    _D = np.sqrt(((_pts[:, None] - _pts[None, :]) ** 2).sum(-1))

    _km_labels, _, _ = du.minkowski_kmeans(_pts, k=_k, p=2, seed=_seed)
    _gmm_labels, _, _ = du.euclidean_gmm(_pts, k=_k, seed=_seed)

    _gmm_aligned = np.copy(_gmm_labels)
    from itertools import permutations as _perms
    _best_perm, _best_score = None, -1
    for _perm in _perms(range(_k)):
        _mapped = np.array([_perm[l] for l in _gmm_labels])
        _score = np.sum(_mapped == _km_labels)
        if _score > _best_score:
            _best_perm, _best_score = _perm, _score
    _gmm_aligned = np.array([_best_perm[l] for l in _gmm_labels])

    _disagree = int(np.sum(_km_labels != _gmm_aligned))

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    _fig.patch.set_facecolor(du.BG_FIGURE)

    for _ax, _labels, _title in [
        (_ax1, _km_labels, 'K-Means (spherical assumption)'),
        (_ax2, _gmm_aligned, 'GMM (full covariance)'),
    ]:
        for _c in range(_k):
            _m = _labels == _c
            _ax.scatter(_pts[_m, 0], _pts[_m, 1],
                        c=du.CLUSTER_COLORS[_c % len(du.CLUSTER_COLORS)],
                        s=26, alpha=0.8, linewidths=0, zorder=3)

        # Draw covariance ellipses for GMM panel
        if _ax is _ax2:
            for _c in range(_k):
                _m = _labels == _c
                if _m.sum() < 3:
                    continue
                _sub = _pts[_m]
                _cov = np.cov(_sub.T)
                _eigvals, _eigvecs = np.linalg.eigh(_cov)
                _eigvals = np.maximum(_eigvals, 1e-6)
                _ang = np.degrees(np.arctan2(_eigvecs[1, 1], _eigvecs[0, 1]))
                from matplotlib.patches import Ellipse as _Ellipse
                _ell = _Ellipse(
                    xy=_sub.mean(0), width=2 * 2 * np.sqrt(_eigvals[1]),
                    height=2 * 2 * np.sqrt(_eigvals[0]), angle=_ang,
                    facecolor='none',
                    edgecolor=du.CLUSTER_COLORS[_c % len(du.CLUSTER_COLORS)],
                    lw=2, ls='--', zorder=4,
                )
                _ax.add_patch(_ell)

        _ax.set_title(_title, fontsize=13, fontweight='bold', pad=6)
        _ax.set_xlabel('x₁', fontsize=12)
        _ax.set_ylabel('x₂', fontsize=12)
        _ax.grid(True, alpha=0.25)
        _ax.tick_params(labelsize=9)
        _ax.set_aspect('equal')

    plt.tight_layout()
    algo_cov_img = mo.image(du.fig_to_png(_fig), width="100%")
    plt.close(_fig)

    algo_cov_txt = mo.callout(
        mo.md(
            f"**{_disagree} of {len(_pts)} points** assigned differently. "
            "K-means assumes spherical clusters — elongated, tilted clusters violate this. "
            "GMM models each cluster's full covariance matrix $\\Sigma_c$, "
            "so elliptical clusters are recovered correctly. "
            "Dashed ellipses show $2\\sigma$ contours of the fitted Gaussians."
        ),
        kind="success",
    )
    return algo_cov_img, algo_cov_txt


# ─── Algorithm Comparison: tab assembly ──────────────────────────────────────

@app.cell
def _algo_tab(algo_compare_img, algo_cov_img, algo_cov_txt, algo_k, algo_seed, mo):
    algo_content = mo.vstack([
        mo.md("## ⚖️ Algorithm Comparison — K-Means vs K-Medoids vs GMM"),
        mo.md(
            "Same data, same K, three algorithms. "
            "Red rings mark points where the algorithm **disagrees** with K-means."
        ),
        mo.hstack([algo_k, algo_seed], gap="1.5rem"),
        algo_compare_img,
        mo.md("---"),
        mo.md("### When Covariance Matters"),
        mo.md(
            "K-means partitions space with **spherical Voronoi cells** — it cannot model "
            "elongated or tilted clusters. GMM fits a full **covariance matrix** per cluster."
        ),
        algo_cov_img,
        algo_cov_txt,
    ], gap="0.6rem")
    return (algo_content,)


# ─── Pitfalls: controls ──────────────────────────────────────────────────────

@app.cell
def _pitfalls_controls(mo):
    pitfalls_seed = mo.ui.slider(start=0, stop=9, value=0, label="Seed", show_value=True)
    return (pitfalls_seed,)


# ─── Pitfalls: K-means failure modes ─────────────────────────────────────────

@app.cell
def _pitfalls_failures(du, mo, np, pitfalls_seed, plt):
    _seed = pitfalls_seed.value
    _rng  = np.random.default_rng(_seed * 31 + 7)

    # ── Dataset 1: two moons (non-convex) ─────────────────────────────────────
    _n = 150
    _t1 = np.linspace(0, np.pi, _n)
    _moon1 = np.column_stack([np.cos(_t1),  np.sin(_t1)]) + _rng.normal(0, 0.08, (_n, 2))
    _moon2 = np.column_stack([np.cos(_t1) + 0.5, -np.sin(_t1) + 0.4]) + _rng.normal(0, 0.08, (_n, 2))
    _moons = np.vstack([_moon1, _moon2])
    _moons_true = np.array([0] * _n + [1] * _n)

    # ── Dataset 2: unequal sizes ──────────────────────────────────────────────
    _big   = _rng.normal(loc=[0, 0], scale=1.5, size=(200, 2))
    _small = _rng.normal(loc=[5, 5], scale=0.3, size=(30, 2))
    _sizes = np.vstack([_big, _small])
    _sizes_true = np.array([0] * 200 + [1] * 30)

    # ── Dataset 3: unequal density ────────────────────────────────────────────
    _dense  = _rng.normal(loc=[0, 0], scale=0.3, size=(150, 2))
    _sparse = _rng.normal(loc=[3, 0], scale=1.2, size=(150, 2))
    _dens   = np.vstack([_dense, _sparse])
    _dens_true = np.array([0] * 150 + [1] * 150)

    _datasets = [
        ('Non-convex (moons)', _moons, _moons_true, 2),
        ('Unequal sizes', _sizes, _sizes_true, 2),
        ('Unequal density', _dens, _dens_true, 2),
    ]

    _fig, _axes = plt.subplots(2, 3, figsize=(14, 7))
    _fig.patch.set_facecolor(du.BG_FIGURE)

    for _col, (_name, _pts, _true, _k) in enumerate(_datasets):
        # Top row: ground truth
        _ax_t = _axes[0, _col]
        for _c in range(_k):
            _m = _true == _c
            _ax_t.scatter(_pts[_m, 0], _pts[_m, 1],
                          c=du.CLUSTER_COLORS[_c], s=16, alpha=0.7, linewidths=0)
        _ax_t.set_title(f'{_name}\nGround Truth', fontsize=12, fontweight='bold', pad=6)
        _ax_t.grid(True, alpha=0.25)
        _ax_t.tick_params(labelsize=8)
        _ax_t.set_aspect('equal')

        # Bottom row: K-means result
        _ax_b = _axes[1, _col]
        _km_labels, _km_centers, _ = du.minkowski_kmeans(_pts, k=_k, p=2, seed=_seed)
        for _c in range(_k):
            _m = _km_labels == _c
            _ax_b.scatter(_pts[_m, 0], _pts[_m, 1],
                          c=du.CLUSTER_COLORS[_c], s=16, alpha=0.7, linewidths=0)
        _ax_b.scatter(_km_centers[:, 0], _km_centers[:, 1],
                      c='white', s=160, marker='*', edgecolors='#1A2030',
                      linewidths=0.8, zorder=5)
        _ax_b.set_title('K-Means Result', fontsize=12, fontweight='bold', pad=6)
        _ax_b.grid(True, alpha=0.25)
        _ax_b.tick_params(labelsize=8)
        _ax_b.set_aspect('equal')

    plt.tight_layout()
    pitfalls_fail_img = mo.image(du.fig_to_png(_fig), width="100%")
    plt.close(_fig)
    return (pitfalls_fail_img,)


# ─── Pitfalls: initialisation sensitivity ─────────────────────────────────────

@app.cell
def _pitfalls_init(du, mo, np, pitfalls_seed, plt):
    _base_seed = pitfalls_seed.value
    _rng = np.random.default_rng(_base_seed * 13 + 3)
    _pts, _ = du.generate_euclidean_clusters(n_points=120, k=3, seed=_base_seed)
    _pts2 = _pts[:, :2]
    _k = 3
    _n_runs = 6

    _fig, _axes = plt.subplots(2, 3, figsize=(14, 7))
    _fig.patch.set_facecolor(du.BG_FIGURE)
    _wcss_vals = []

    for _i, _ax in enumerate(_axes.flat):
        _s = _base_seed * 100 + _i * 7 + 1
        _labels, _centers, _ = du.minkowski_kmeans(_pts, k=_k, p=2, seed=_s)
        _wcss = float(sum(
            np.sum((_pts[_labels == c] - _centers[c]) ** 2)
            for c in range(_k)
        ))
        _wcss_vals.append(_wcss)

        for _c in range(_k):
            _m = _labels == _c
            _ax.scatter(_pts2[_m, 0], _pts2[_m, 1],
                        c=du.CLUSTER_COLORS[_c % len(du.CLUSTER_COLORS)],
                        s=20, alpha=0.7, linewidths=0, zorder=3)
        _ax.scatter(_centers[:, 0], _centers[:, 1],
                    c='white', s=140, marker='*', edgecolors='#1A2030',
                    linewidths=0.8, zorder=5)
        _ax.set_title(f'Seed {_s}  ·  WCSS = {_wcss:.1f}', fontsize=11, pad=4)
        _ax.grid(True, alpha=0.25)
        _ax.tick_params(labelsize=8)

    _best_i = int(np.argmin(_wcss_vals))
    _worst_i = int(np.argmax(_wcss_vals))
    for _i, _ax in enumerate(_axes.flat):
        if _i == _best_i:
            for spine in _ax.spines.values():
                spine.set_edgecolor('#2E7D32'); spine.set_linewidth(3)
        elif _i == _worst_i:
            for spine in _ax.spines.values():
                spine.set_edgecolor('#C62828'); spine.set_linewidth(3)

    plt.tight_layout()
    pitfalls_init_img = mo.image(du.fig_to_png(_fig), width="100%")
    plt.close(_fig)

    _spread = max(_wcss_vals) - min(_wcss_vals)
    pitfalls_init_txt = mo.callout(
        mo.md(
            f"**Same data, same K={_k}, six random starts.** "
            f"WCSS ranges from **{min(_wcss_vals):.1f}** "
            f"(green border) to **{max(_wcss_vals):.1f}** "
            f"(red border) — a spread of {_spread:.1f}. "
            "Each run converges to a **local** minimum; the global optimum is not guaranteed. "
            "In practice, run K-means 10–50 times and keep the best."
        ),
        kind="warn",
    )
    return pitfalls_init_img, pitfalls_init_txt


# ─── Pitfalls: tab assembly ──────────────────────────────────────────────────

@app.cell
def _pitfalls_tab(mo, pitfalls_fail_img, pitfalls_init_img, pitfalls_init_txt, pitfalls_seed):
    pitfalls_content = mo.vstack([
        mo.md("## ⚠️ Pitfalls — When K-Means Breaks Down"),
        pitfalls_seed,
        mo.md("### Failure Modes"),
        mo.md(
            "K-means assumes clusters are **convex**, **similarly sized**, and **similarly dense**. "
            "When any assumption is violated, the algorithm confidently returns the wrong answer."
        ),
        pitfalls_fail_img,
        mo.callout(
            mo.md(
                "**Non-convex:** K-means bisects moons with a straight line — it can only carve "
                "convex regions. **Unequal sizes:** the large cluster 'donates' points to the small "
                "one to equalise WCSS. **Unequal density:** the boundary shifts toward the dense cluster."
            ),
            kind="info",
        ),
        mo.md("---"),
        mo.md("### Initialisation Sensitivity"),
        mo.md(
            "Lloyd's algorithm is **deterministic given the initial centroids** — but those centroids "
            "are chosen randomly, so different seeds yield different partitions."
        ),
        pitfalls_init_img,
        pitfalls_init_txt,
    ], gap="0.6rem")
    return (pitfalls_content,)


# ─── Main layout ─────────────────────────────────────────────────────────────

@app.cell
def _layout(
    mo,
    algo_content, feat_content, hyp_content, lloyd_content,
    mixed_content, overview_content, pitfalls_content,
    sphere_content, torus_content,
):
    tabs = mo.ui.tabs({
        "📋 Overview":    overview_content,
        "🌍 Sphere":      sphere_content,
        "🍩 Torus":       torus_content,
        "🔵 Hyperbolic":  hyp_content,
        "📏 Features":    feat_content,
        "🔀 Mixed":       mixed_content,
        "🔄 Lloyd's":     lloyd_content,
        "⚖️ Algorithms":  algo_content,
        "⚠️ Pitfalls":    pitfalls_content,
    })

    layout = mo.vstack([
        tabs,
        mo.md(
            "---\n"
            "*MAA Seaway Section · Spring 2026 · St. John Fisher University, Rochester NY*"
        ),
    ])

    return layout, tabs


@app.cell
def _show(layout):
    layout  # marimo renders the last expression as cell output


if __name__ == "__main__":
    app.run()
