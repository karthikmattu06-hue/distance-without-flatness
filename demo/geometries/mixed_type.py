import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .base import GeometryBase
from demo.theme import CLUSTER_COLORS, PLOTLY_LAYOUT, blend_colors, hex_to_rgb, BG_COLOR, GRID_COLOR
from demo.plots.plotly_figures import convergence_chart


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLIMATE_NAMES = ['tropical', 'arid', 'temperate', 'polar']
TERRAIN_NAMES = ['mountain', 'plains', 'forest', 'coastal']
N_CLIMATES = len(CLIMATE_NAMES)
N_TERRAINS = len(TERRAIN_NAMES)

# Plotly marker symbols for the four climate categories
CLIMATE_MARKERS = {0: 'circle', 1: 'square', 2: 'diamond', 3: 'triangle-up'}
TERRAIN_MARKERS = {0: 'circle', 1: 'square', 2: 'diamond', 3: 'triangle-up'}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_gaussian_pdf(X, mu, cov, reg=1e-6):
    """
    Log N(x | mu, cov) for each row of X via Cholesky decomposition.

    Parameters
    ----------
    X   : (N, D) data points
    mu  : (D,)   mean vector
    cov : (D, D) covariance matrix
    reg : float  diagonal regularisation

    Returns
    -------
    log_probs : (N,) log-densities
    """
    D = X.shape[1]
    cov_reg = cov + reg * np.eye(D)
    try:
        L = np.linalg.cholesky(cov_reg)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(cov_reg + 1e-4 * np.eye(D))

    diff = X - mu                              # (N, D)
    z = np.linalg.solve(L, diff.T)             # (D, N)
    mahal = np.sum(z ** 2, axis=0)             # (N,)
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (D * np.log(2 * np.pi) + log_det + mahal)


def _log_sum_exp(log_vals, axis=None):
    """Numerically stable log-sum-exp."""
    max_val = np.max(log_vals, axis=axis, keepdims=True)
    return np.squeeze(max_val, axis=axis) + np.log(
        np.sum(np.exp(log_vals - max_val), axis=axis)
    )


def _haversine_distance(lat1, lon1, lat2, lon2):
    """Haversine distance on a unit sphere (inputs in radians)."""
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _gower_distance_matrix(points, climate, terrain,
                           w_geo=1.0, w_climate=1.0, w_terrain=1.0):
    """
    Pairwise Gower distance matrix.

    Geographic component: haversine / pi  (normalised to [0, 1]).
    Categorical components: simple matching (0 if same, 1 if different).
    Final distance: weighted average.
    """
    lats, lons = points[:, 0], points[:, 1]
    N = len(lats)
    D = np.zeros((N, N))
    w_total = w_geo + w_climate + w_terrain

    for i in range(N):
        for j in range(i + 1, N):
            d_geo = _haversine_distance(lats[i], lons[i], lats[j], lons[j]) / np.pi
            d_climate = 0.0 if climate[i] == climate[j] else 1.0
            d_terrain = 0.0 if terrain[i] == terrain[j] else 1.0
            val = (w_geo * d_geo + w_climate * d_climate + w_terrain * d_terrain) / w_total
            D[i, j] = val
            D[j, i] = val

    return D


def _kmedoids_simplified(D, k, max_iter=100, seed=123):
    """
    K-Medoids via simplified PAM (random-swap variant).

    Parameters
    ----------
    D        : (N, N) precomputed distance matrix
    k        : number of clusters
    max_iter : maximum iterations
    seed     : random seed

    Returns
    -------
    medoids      : (k,) indices of medoid points
    labels       : (N,) cluster assignments
    cost_history : list of total-cost per iteration
    """
    rng = np.random.default_rng(seed)
    N = D.shape[0]

    medoids = rng.choice(N, size=k, replace=False)
    cost_history = []

    for _ in range(max_iter):
        dists_to_med = D[:, medoids]                    # (N, k)
        labels = np.argmin(dists_to_med, axis=1)        # (N,)
        cost = sum(D[i, medoids[labels[i]]] for i in range(N))
        cost_history.append(cost)

        improved = False
        for m_idx in range(k):
            non_medoids = np.array([i for i in range(N) if i not in medoids])
            n_cands = min(30, len(non_medoids))
            candidates = rng.choice(non_medoids, size=n_cands, replace=False)
            for cand in candidates:
                new_medoids = medoids.copy()
                new_medoids[m_idx] = cand
                new_dists = D[:, new_medoids]
                new_labels = np.argmin(new_dists, axis=1)
                new_cost = sum(D[i, new_medoids[new_labels[i]]] for i in range(N))
                if new_cost < cost:
                    medoids = new_medoids
                    labels = new_labels
                    cost = new_cost
                    improved = True

        if not improved:
            break

    return medoids, labels, cost_history


def _ellipse_points(mean, cov, n_std=2, n_points=100):
    """Points on the boundary of a covariance ellipse."""
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    angle = np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1])
    t = np.linspace(0, 2 * np.pi, n_points)
    ellipse = np.array([n_std * np.sqrt(max(eigenvalues[1], 0)) * np.cos(t),
                         n_std * np.sqrt(max(eigenvalues[0], 0)) * np.sin(t)])
    R = np.array([[np.cos(angle), -np.sin(angle)],
                   [np.sin(angle),  np.cos(angle)]])
    rotated = R @ ellipse
    return mean[0] + rotated[0], mean[1] + rotated[1]


# ---------------------------------------------------------------------------
# Geometry class
# ---------------------------------------------------------------------------

class MixedTypeGeometry(GeometryBase):
    name = "Mixed-Type"
    plot_type = "2d"
    default_K = 4
    default_n_points = 250
    spread_range = (0.1, 1.5)
    spread_label = "Geographic Spread"
    spread_default = 0.3
    spread_help = "Controls geographic spread of clusters. Categorical features are separate."
    distance_options = {
        "Gower (Geo + Categorical)": "gower",
        "Geographic Only": "geo_only",
        "Categorical Only": "cat_only",
    }

    # ------------------------------------------------------------------
    # Data generation
    # ------------------------------------------------------------------
    def generate_data(self, K, n_points, spread, seed):
        """
        Generate K clusters with continuous geographic (lat, lon) features
        and two categorical features (climate, terrain).

        Two clusters are placed geographically close but with very different
        categorical profiles so that pure geographic clustering fails.
        """
        rng = np.random.default_rng(seed)

        # Lay out cluster centres roughly on a grid, then make two close
        cols = int(np.ceil(np.sqrt(K)))
        base_centers = []
        for i in range(K):
            cx = (i % cols) * 1.2 - (cols - 1) * 0.6
            cy = (i // cols) * 1.2 - (cols - 1) * 0.6
            base_centers.append(np.array([cy, cx]))       # (lat, lon)

        # Make the last two clusters geographically close
        if K >= 2:
            base_centers[-1] = base_centers[-2] + rng.uniform(-0.15, 0.15, size=2)

        # Build per-cluster categorical distributions
        def _dominant_probs(n_cat, dominant_idx, dominant_p=0.70):
            """Return a probability vector with one dominant category."""
            probs = np.full(n_cat, (1.0 - dominant_p) / (n_cat - 1))
            probs[dominant_idx] = dominant_p
            return probs

        pts_per_cluster = n_points // K
        all_points, all_labels = [], []
        all_climate, all_terrain = [], []

        for i in range(K):
            n_i = pts_per_cluster if i < K - 1 else n_points - pts_per_cluster * (K - 1)

            # Random covariance scaled by *spread*
            A = rng.standard_normal((2, 2))
            cov = spread * (A @ A.T) / 4.0 + 0.01 * np.eye(2)

            pts = rng.multivariate_normal(base_centers[i], cov, size=n_i)
            all_points.append(pts)
            all_labels.append(np.full(n_i, i))

            # Categorical distributions — each cluster has a dominant category
            clim_probs = _dominant_probs(N_CLIMATES, i % N_CLIMATES)
            terr_probs = _dominant_probs(N_TERRAINS, (i + 1) % N_TERRAINS)

            # For the close pair, make categories maximally different
            if K >= 2 and i == K - 1:
                clim_probs = _dominant_probs(N_CLIMATES, (K - 2 + 2) % N_CLIMATES)
                terr_probs = _dominant_probs(N_TERRAINS, (K - 2 + 3) % N_TERRAINS)

            all_climate.append(rng.choice(N_CLIMATES, size=n_i, p=clim_probs))
            all_terrain.append(rng.choice(N_TERRAINS, size=n_i, p=terr_probs))

        points = np.vstack(all_points)
        labels = np.concatenate(all_labels)
        climate = np.concatenate(all_climate).astype(int)
        terrain = np.concatenate(all_terrain).astype(int)

        # Shuffle
        perm = rng.permutation(len(points))
        points = points[perm]
        labels = labels[perm]
        climate = climate[perm]
        terrain = terrain[perm]

        return {
            'points': points,
            'labels': labels,
            'params': {'centers': base_centers},
            'climate': climate,
            'terrain': terrain,
            'climate_names': list(CLIMATE_NAMES),
            'terrain_names': list(TERRAIN_NAMES),
        }

    # ------------------------------------------------------------------
    # K-Medoids with Gower distance
    # ------------------------------------------------------------------
    def kmeans_init(self, points, K, seed=123, **kwargs):
        """Initialise K-Medoids: pick K random medoid indices and compute distance matrix."""
        climate = kwargs.get('climate', np.zeros(len(points), dtype=int))
        terrain = kwargs.get('terrain', np.zeros(len(points), dtype=int))
        distance = kwargs.get('distance', 'gower')

        rng = np.random.default_rng(seed)
        N = len(points)

        if distance == 'geo_only':
            D = _gower_distance_matrix(points, climate, terrain,
                                        w_geo=1.0, w_climate=0.0, w_terrain=0.0)
        elif distance == 'cat_only':
            D = _gower_distance_matrix(points, climate, terrain,
                                        w_geo=0.0, w_climate=1.0, w_terrain=1.0)
        else:
            D = _gower_distance_matrix(points, climate, terrain)
        medoids = rng.choice(N, size=K, replace=False)

        # Initial assignment
        dists_to_med = D[:, medoids]
        labels = np.argmin(dists_to_med, axis=1)
        cost = float(sum(D[i, medoids[labels[i]]] for i in range(N)))

        return {
            'centroids': points[medoids],
            'labels': labels,
            'cost_history': [cost],
            'iteration': 0,
            '_medoids': medoids,
            '_D': D,
        }

    def kmeans_step(self, points, state):
        """One K-Medoids (PAM) step: try swapping each medoid and keep improvements."""
        D = state['_D']
        medoids = state['_medoids'].copy()
        N = D.shape[0]
        K = len(medoids)

        dists_to_med = D[:, medoids]
        labels = np.argmin(dists_to_med, axis=1)
        cost = float(sum(D[i, medoids[labels[i]]] for i in range(N)))

        rng = np.random.default_rng(state['iteration'])
        improved = False
        for m_idx in range(K):
            non_medoids = np.array([i for i in range(N) if i not in medoids])
            n_cands = min(30, len(non_medoids))
            candidates = rng.choice(non_medoids, size=n_cands, replace=False)
            for cand in candidates:
                new_medoids = medoids.copy()
                new_medoids[m_idx] = cand
                new_dists = D[:, new_medoids]
                new_labels = np.argmin(new_dists, axis=1)
                new_cost = float(sum(D[i, new_medoids[new_labels[i]]] for i in range(N)))
                if new_cost < cost:
                    medoids = new_medoids
                    labels = new_labels
                    cost = new_cost
                    improved = True

        cost_history = state['cost_history'] + [cost]

        return {
            'centroids': points[medoids],
            'labels': labels,
            'cost_history': cost_history,
            'iteration': state['iteration'] + 1,
            '_medoids': medoids,
            '_D': D,
            '_improved': improved,
        }

    def kmeans_full(self, points, K, max_iter=100, seed=123, **kwargs):
        """
        K-Medoids using Gower distance (simplified PAM).

        Expects *points* to be the (N, 2) geographic array.  The categorical
        arrays ``climate`` and ``terrain`` must be passed via **kwargs** or
        will be treated as all-zeros.
        """
        state = self.kmeans_init(points, K, seed=seed, **kwargs)

        for _ in range(max_iter):
            state = self.kmeans_step(points, state)
            if not state['_improved']:
                break

        return {'labels': state['labels'], 'centroids': state['centroids'], 'cost_history': state['cost_history']}

    # ------------------------------------------------------------------
    # EM – Mixed-type Gaussian + Categorical mixture
    # ------------------------------------------------------------------
    def em_init(self, points, K, seed=123, **kwargs):
        """
        Initialise a Gaussian + Categorical mixture model.

        State dictionary
        ----------------
        means            : (K, 2)
        covariances      : list of (2, 2)
        weights          : (K,)
        climate_probs    : (K, 4)
        terrain_probs    : (K, 4)
        responsibilities : (N, K)
        log_likelihood   : float
        iteration        : int
        ll_history       : list[float]
        """
        climate = kwargs.get('climate', np.zeros(len(points), dtype=int))
        terrain = kwargs.get('terrain', np.zeros(len(points), dtype=int))

        rng = np.random.default_rng(seed)
        N, D = points.shape

        init_idx = rng.choice(N, size=K, replace=False)
        means = points[init_idx].copy()
        covariances = [np.cov(points.T) * 0.5 + 1e-6 * np.eye(D) for _ in range(K)]
        weights = np.ones(K) / K

        # Categorical: uniform + small perturbation
        climate_probs = np.ones((K, N_CLIMATES)) / N_CLIMATES + rng.uniform(0, 0.1, size=(K, N_CLIMATES))
        climate_probs /= climate_probs.sum(axis=1, keepdims=True)
        terrain_probs = np.ones((K, N_TERRAINS)) / N_TERRAINS + rng.uniform(0, 0.1, size=(K, N_TERRAINS))
        terrain_probs /= terrain_probs.sum(axis=1, keepdims=True)

        responsibilities = np.full((N, K), 1.0 / K)

        # Initial log-likelihood
        log_resp = np.zeros((N, K))
        for k in range(K):
            log_gauss = _log_gaussian_pdf(points, means[k], covariances[k])
            log_cat = (np.log(climate_probs[k, climate] + 1e-300)
                       + np.log(terrain_probs[k, terrain] + 1e-300))
            log_resp[:, k] = np.log(weights[k] + 1e-300) + log_gauss + log_cat
        ll = float(np.sum(_log_sum_exp(log_resp, axis=1)))

        return {
            'means': means,
            'covariances': covariances,
            'weights': weights,
            'climate_probs': climate_probs,
            'terrain_probs': terrain_probs,
            'responsibilities': responsibilities,
            'log_likelihood': ll,
            'iteration': 0,
            'll_history': [],
            # Carry categorical arrays through for subsequent steps
            '_climate': climate,
            '_terrain': terrain,
        }

    def em_step(self, points, state):
        """
        One EM iteration for the Gaussian + Categorical mixture.

        E-step
        ------
        log p_{nk} = log pi_k + log N(x_n | mu_k, Sigma_k)
                      + sum_f log phi_{k,f,c_{nf}}

        Normalise via log-sum-exp to get responsibilities.

        M-step
        ------
        Gaussian: weighted mean and covariance (+ 1e-6*I regularisation).
        Categorical: weighted frequency counts + Laplace smoothing (alpha=0.01).
        """
        means = state['means'].copy()
        covariances = [c.copy() for c in state['covariances']]
        weights = state['weights'].copy()
        climate_probs = state['climate_probs'].copy()
        terrain_probs = state['terrain_probs'].copy()
        climate = state['_climate']
        terrain = state['_terrain']

        N, D = points.shape
        K = len(means)
        reg = 1e-6
        alpha = 0.01

        # ===================== E-STEP =====================
        log_resp = np.zeros((N, K))
        for k in range(K):
            log_gauss = _log_gaussian_pdf(points, means[k], covariances[k], reg=reg)
            log_cat = (np.log(climate_probs[k, climate] + 1e-300)
                       + np.log(terrain_probs[k, terrain] + 1e-300))
            log_resp[:, k] = np.log(weights[k] + 1e-300) + log_gauss + log_cat

        log_resp_norm = _log_sum_exp(log_resp, axis=1)          # (N,)
        responsibilities = np.exp(log_resp - log_resp_norm[:, np.newaxis])
        ll = float(np.sum(log_resp_norm))

        # ===================== M-STEP =====================
        N_k = responsibilities.sum(axis=0)                      # (K,)

        for k in range(K):
            r_k = responsibilities[:, k]
            if N_k[k] < 1e-10:
                continue

            # Gaussian mean
            means[k] = (r_k[:, np.newaxis] * points).sum(axis=0) / N_k[k]
            # Gaussian covariance
            diff = points - means[k]
            covariances[k] = (diff.T @ (diff * r_k[:, np.newaxis])) / N_k[k]
            covariances[k] += reg * np.eye(D)

            # Mixing weight
            weights[k] = N_k[k] / N

            # Climate categorical probs (Laplace smoothing)
            for v in range(N_CLIMATES):
                mask_v = (climate == v)
                climate_probs[k, v] = (np.sum(r_k[mask_v]) + alpha) / (N_k[k] + alpha * N_CLIMATES)

            # Terrain categorical probs (Laplace smoothing)
            for v in range(N_TERRAINS):
                mask_v = (terrain == v)
                terrain_probs[k, v] = (np.sum(r_k[mask_v]) + alpha) / (N_k[k] + alpha * N_TERRAINS)

        iteration = state['iteration'] + 1
        ll_history = state['ll_history'] + [ll]

        return {
            'means': means,
            'covariances': covariances,
            'weights': weights,
            'climate_probs': climate_probs,
            'terrain_probs': terrain_probs,
            'responsibilities': responsibilities,
            'log_likelihood': ll,
            'iteration': iteration,
            'll_history': ll_history,
            '_climate': climate,
            '_terrain': terrain,
        }

    def em_full(self, points, K, max_iter=200, seed=123, **kwargs):
        """Run EM to convergence."""
        state = self.em_init(points, K, seed=seed, **kwargs)
        tol = 1e-6

        for _ in range(max_iter):
            prev_ll = state['log_likelihood']
            state = self.em_step(points, state)
            if state['iteration'] > 1 and abs(state['log_likelihood'] - prev_ll) < tol:
                break

        return state

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def build_plot(self, points, labels, centers, responsibilities=None,
                   show_soft=False, title="", true_labels=None, **kwargs):
        """
        2D scatter of geographic features.

        Marker *symbol* encodes climate type; colour encodes cluster.
        Hover text shows climate + terrain names.
        Centers are shown as black stars; covariance ellipses are drawn
        when the EM state provides them.
        """
        climate = kwargs.get('climate', None)
        terrain = kwargs.get('terrain', None)

        fig = go.Figure()
        N = len(points)
        K = int(labels.max()) + 1 if len(labels) > 0 else 0

        # Determine center coordinates and optional covariances
        if isinstance(centers, dict):
            center_pts = np.asarray(centers['means'])
            covariances = centers.get('covariances', None)
        else:
            center_pts = np.asarray(centers)
            covariances = None
        has_covs = covariances is not None

        # Per-point colours
        if show_soft and responsibilities is not None:
            marker_colors = blend_colors(responsibilities)
        else:
            marker_colors = [CLUSTER_COLORS[int(l) % len(CLUSTER_COLORS)] for l in labels]

        # --- scatter with climate-based markers ---
        if climate is not None:
            for c_idx in range(N_CLIMATES):
                mask_c = (climate == c_idx)
                if not np.any(mask_c):
                    continue
                idxs = np.where(mask_c)[0]
                colors_sub = [marker_colors[i] for i in idxs]

                hover_texts = []
                for i in idxs:
                    parts = [f"lat: {points[i, 0]:.3f}",
                             f"lon: {points[i, 1]:.3f}"]
                    parts.append(f"climate: {CLIMATE_NAMES[climate[i]]}")
                    if terrain is not None:
                        parts.append(f"terrain: {TERRAIN_NAMES[terrain[i]]}")
                    parts.append(f"cluster: {labels[i]}")
                    hover_texts.append("<br>".join(parts))

                fig.add_trace(go.Scatter(
                    x=points[idxs, 1],          # lon on x-axis
                    y=points[idxs, 0],          # lat on y-axis
                    mode='markers',
                    marker=dict(
                        size=7,
                        color=colors_sub,
                        symbol=CLIMATE_MARKERS[c_idx],
                        opacity=0.75,
                        line=dict(width=0.4, color='rgba(255,255,255,0.4)'),
                    ),
                    name=f"{CLIMATE_NAMES[c_idx]}",
                    hovertext=hover_texts,
                    hoverinfo='text',
                ))
        else:
            # Fallback: plain scatter without marker-shape encoding
            fig.add_trace(go.Scatter(
                x=points[:, 1], y=points[:, 0],
                mode='markers',
                marker=dict(size=6, color=marker_colors, opacity=0.75,
                            line=dict(width=0.3, color='rgba(255,255,255,0.3)')),
                showlegend=False,
                hovertemplate='lat: %{y:.3f}<br>lon: %{x:.3f}<extra></extra>',
            ))

        # Cluster boundaries
        if covariances is not None:
            # EM mode: draw learned covariance ellipses
            for j in range(len(covariances)):
                color = CLUSTER_COLORS[j % len(CLUSTER_COLORS)]
                for n_std in (1, 2):
                    ey, ex = _ellipse_points(center_pts[j], covariances[j],
                                             n_std=n_std, n_points=120)
                    fig.add_trace(go.Scatter(
                        x=ex, y=ey, mode='lines',
                        line=dict(color=color, width=2 if n_std == 1 else 1),
                        opacity=0.8 if n_std == 1 else 0.4,
                        showlegend=False, hoverinfo='skip',
                    ))
        else:
            # K-Medoids mode: circles (assumes equal spherical spread)
            for j in range(K):
                members = points[labels == j]
                if len(members) < 2:
                    continue
                radius = float(np.mean(np.linalg.norm(members - center_pts[j], axis=1)))
                color = CLUSTER_COLORS[j % len(CLUSTER_COLORS)]
                t = np.linspace(0, 2 * np.pi, 100)
                # Note: x=lon, y=lat for this plot
                fig.add_trace(go.Scatter(
                    x=center_pts[j, 1] + radius * np.cos(t),
                    y=center_pts[j, 0] + radius * np.sin(t),
                    mode='lines',
                    line=dict(color=color, width=2),
                    opacity=0.6,
                    showlegend=False, hoverinfo='skip',
                ))

        # Centers as black stars
        fig.add_trace(go.Scatter(
            x=center_pts[:, 1], y=center_pts[:, 0],
            mode='markers',
            marker=dict(size=16, color='black', symbol='star',
                        line=dict(color='white', width=1.5)),
            name='Centers',
        ))

        fig.update_layout(
            **PLOTLY_LAYOUT,
            title=title,
            xaxis=dict(title="Longitude (rad)", gridcolor=GRID_COLOR, zeroline=False),
            yaxis=dict(title="Latitude (rad)", gridcolor=GRID_COLOR, zeroline=False,
                       scaleanchor='x', scaleratio=1),
            height=600,
        )

        return fig

    def build_convergence_plot(self, history, title="EM Convergence"):
        """Delegate to the shared convergence chart helper."""
        return convergence_chart(history, title=title)
