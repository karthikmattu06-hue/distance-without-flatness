import numpy as np
import plotly.graph_objects as go
from .base import GeometryBase
from demo.theme import CLUSTER_COLORS, PLOTLY_LAYOUT, blend_colors, hex_to_rgb, BG_COLOR, GRID_COLOR
from demo.plots.plotly_figures import convergence_chart


def _ellipse_points(mean, cov, n_std=2, n_points=100):
    """Generate points on the boundary of a covariance ellipse."""
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    angle = np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1])
    t = np.linspace(0, 2 * np.pi, n_points)
    ellipse = np.array([n_std * np.sqrt(eigenvalues[1]) * np.cos(t),
                         n_std * np.sqrt(eigenvalues[0]) * np.sin(t)])
    R = np.array([[np.cos(angle), -np.sin(angle)],
                   [np.sin(angle), np.cos(angle)]])
    rotated = R @ ellipse
    return mean[0] + rotated[0], mean[1] + rotated[1]


def _log_gaussian_pdf(x, mu, sigma):
    """
    Compute log N(x | mu, sigma) for each point.

    x: (N, d) array of points
    mu: (d,) mean vector
    sigma: (d, d) covariance matrix

    Returns: (N,) array of log-densities
    """
    d = len(mu)
    diff = x - mu  # (N, d)

    # Cholesky for numerical stability and efficient computation
    try:
        L = np.linalg.cholesky(sigma)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(sigma + 1e-6 * np.eye(d))

    # Solve L * z = diff^T  =>  z = L^{-1} diff^T
    z = np.linalg.solve(L, diff.T)  # (d, N)
    mahal_sq = np.sum(z ** 2, axis=0)  # (N,)

    # log|sigma| = 2 * sum(log(diag(L)))
    log_det = 2.0 * np.sum(np.log(np.diag(L)))

    # log N(x|mu,sigma) = -d/2 log(2pi) - 1/2 log|sigma| - 1/2 mahal^2
    log_prob = -0.5 * (d * np.log(2 * np.pi) + log_det + mahal_sq)

    return log_prob


def _log_sum_exp(log_vals, axis=None):
    """Numerically stable log-sum-exp."""
    max_val = np.max(log_vals, axis=axis, keepdims=True)
    return np.squeeze(max_val, axis=axis) + np.log(
        np.sum(np.exp(log_vals - max_val), axis=axis)
    )


class EuclideanGeometry(GeometryBase):
    name = "Euclidean (R²)"
    plot_type = "2d"
    default_K = 4
    default_n_points = 300
    spread_range = (0.0, 3.0)
    spread_label = "Covariance Spread"
    spread_default = 1.0
    spread_help = ("At 0 all clusters are round (K-Means = EM). "
                   "Higher values make clusters elongated and varied — EM's advantage.")
    distance_options = {
        "Euclidean (L2)": "euclidean",
        "Manhattan (L1)": "manhattan",
        "Chebyshev (L∞)": "chebyshev",
    }

    # ------------------------------------------------------------------
    # Data generation
    # ------------------------------------------------------------------
    def generate_data(self, K, n_points, spread, seed):
        """Generate K clusters in 2D with random covariance matrices scaled by *spread*."""
        rng = np.random.default_rng(seed)

        # Lay centres out in a grid-like pattern
        cols = int(np.ceil(np.sqrt(K)))
        centers_list = []
        for i in range(K):
            cx = (i % cols) * 4.0 - (cols - 1) * 2.0
            cy = (i // cols) * 4.0 - (cols - 1) * 2.0
            centers_list.append(np.array([cx, cy]))

        pts_per_cluster = n_points // K
        all_points, all_labels = [], []
        true_means, true_covs = [], []

        for i in range(K):
            n_i = pts_per_cluster if i < K - 1 else n_points - pts_per_cluster * (K - 1)
            # Covariance: blend from identity (spread=0) to random elliptical (spread>0)
            # At spread=0: all clusters are perfectly round (identity covariance)
            # At spread>0: random orientation and eccentricity
            A = rng.standard_normal((2, 2))
            random_cov = (A @ A.T) / 2.0
            cov = (1 - min(spread, 1.0)) * np.eye(2) + spread * random_cov + 0.05 * np.eye(2)
            pts = rng.multivariate_normal(centers_list[i], cov, size=n_i)
            all_points.append(pts)
            all_labels.append(np.full(n_i, i))
            true_means.append(centers_list[i])
            true_covs.append(cov)

        points = np.vstack(all_points)
        labels = np.concatenate(all_labels)

        # Shuffle
        perm = rng.permutation(len(points))
        points = points[perm]
        labels = labels[perm]

        return {
            'points': points,
            'labels': labels,
            'params': {'means': true_means, 'covs': true_covs},
        }

    # ------------------------------------------------------------------
    # K-means
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_distance(points, centroid, metric):
        """Compute distance from each point to a centroid using the given metric."""
        diff = points - centroid
        if metric == "manhattan":
            return np.sum(np.abs(diff), axis=1)
        elif metric == "chebyshev":
            return np.max(np.abs(diff), axis=1)
        else:  # euclidean
            return np.linalg.norm(diff, axis=1)

    @staticmethod
    def _compute_centroid(members, metric):
        """Compute the centroid appropriate for the given metric."""
        if metric == "manhattan":
            return np.median(members, axis=0)
        elif metric == "chebyshev":
            return (members.max(axis=0) + members.min(axis=0)) / 2
        else:  # euclidean
            return members.mean(axis=0)

    def kmeans_init(self, points, K, seed=123, **kwargs):
        """Initialise K-means: pick K random points as centroids."""
        rng = np.random.default_rng(seed)
        N = len(points)
        init_idx = rng.choice(N, size=K, replace=False)
        centroids = points[init_idx].copy()
        distance = kwargs.get('distance', 'euclidean')
        return {
            'centroids': centroids,
            'labels': np.zeros(N, dtype=int),
            'cost_history': [],
            'iteration': 0,
            '_distance': distance,
        }

    def kmeans_step(self, points, state):
        """One K-means iteration using the selected distance metric."""
        centroids = state['centroids']
        metric = state.get('_distance', 'euclidean')
        N = len(points)
        K = len(centroids)

        # Assignment
        dist_matrix = np.zeros((N, K))
        for j in range(K):
            dist_matrix[:, j] = self._compute_distance(points, centroids[j], metric)
        labels = np.argmin(dist_matrix, axis=1)

        # Cost
        cost = float(np.sum(dist_matrix[np.arange(N), labels] ** 2))
        cost_history = state['cost_history'] + [cost]

        # Update centroids
        rng = np.random.default_rng(state['iteration'])
        new_centroids = np.empty_like(centroids)
        for j in range(K):
            members = points[labels == j]
            if len(members) == 0:
                new_centroids[j] = points[rng.choice(N)]
            else:
                new_centroids[j] = self._compute_centroid(members, metric)

        return {
            'centroids': new_centroids,
            'labels': labels,
            'cost_history': cost_history,
            'iteration': state['iteration'] + 1,
            '_shift': np.max(np.linalg.norm(centroids - new_centroids, axis=1)),
            '_distance': metric,
        }

    def kmeans_full(self, points, K, max_iter=100, seed=123, **kwargs):
        """Standard Euclidean K-means."""
        state = self.kmeans_init(points, K, seed=seed, **kwargs)

        for _ in range(max_iter):
            state = self.kmeans_step(points, state)
            if state['_shift'] < 1e-6:
                break

        return {'labels': state['labels'], 'centroids': state['centroids'], 'cost_history': state['cost_history']}

    # ------------------------------------------------------------------
    # EM helpers
    # ------------------------------------------------------------------
    def em_init(self, points, K, seed=123, **kwargs):
        """Initialise GMM parameters."""
        rng = np.random.default_rng(seed)
        N, d = points.shape

        init_idx = rng.choice(N, size=K, replace=False)
        means = points[init_idx].copy()
        covariances = [np.eye(d) for _ in range(K)]
        weights = np.ones(K) / K

        # Initial responsibilities (uniform)
        responsibilities = np.full((N, K), 1.0 / K)

        # Initial log-likelihood
        log_resp = np.zeros((N, K))
        for j in range(K):
            log_resp[:, j] = np.log(weights[j]) + _log_gaussian_pdf(points, means[j], covariances[j])
        ll = float(np.sum(_log_sum_exp(log_resp, axis=1)))

        return {
            'means': means,
            'covariances': covariances,
            'weights': weights,
            'responsibilities': responsibilities,
            'log_likelihood': ll,
            'iteration': 0,
            'll_history': [],
        }

    def em_step(self, points, state):
        """One EM iteration with log-sum-exp trick and Cholesky-based log pdf."""
        means = state['means'].copy()
        covariances = [c.copy() for c in state['covariances']]
        weights = state['weights'].copy()
        N, d = points.shape
        K = len(means)
        reg = 1e-6

        # === E-step ===
        log_resp = np.zeros((N, K))
        for j in range(K):
            log_resp[:, j] = np.log(weights[j]) + _log_gaussian_pdf(
                points, means[j], covariances[j]
            )

        log_resp_norm = _log_sum_exp(log_resp, axis=1)  # (N,)
        log_resp_normalized = log_resp - log_resp_norm[:, np.newaxis]
        responsibilities = np.exp(log_resp_normalized)

        ll = float(np.sum(log_resp_norm))

        # === M-step ===
        for j in range(K):
            r_j = responsibilities[:, j]
            N_j = r_j.sum()
            if N_j < 1e-10:
                continue

            means[j] = (r_j[:, np.newaxis] * points).sum(axis=0) / N_j

            diff = points - means[j]
            covariances[j] = (diff.T @ (diff * r_j[:, np.newaxis])) / N_j
            covariances[j] += reg * np.eye(d)

            weights[j] = N_j / N

        iteration = state['iteration'] + 1
        ll_history = state['ll_history'] + [ll]

        return {
            'means': means,
            'covariances': covariances,
            'weights': weights,
            'responsibilities': responsibilities,
            'log_likelihood': ll,
            'iteration': iteration,
            'll_history': ll_history,
        }

    def em_full(self, points, K, max_iter=200, seed=123):
        """Run EM to convergence."""
        state = self.em_init(points, K, seed=seed)
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
                   show_soft=False, title="", true_labels=None):
        """
        Build a Plotly 2D scatter plot.

        *centers* may be either a plain ndarray of centroids (K-means) or a
        dict with keys ``'means'`` and ``'covariances'`` (EM).
        """
        fig = go.Figure()
        N = len(points)
        K = int(labels.max()) + 1 if len(labels) > 0 else 0

        # Determine whether we have covariance info
        if isinstance(centers, dict):
            center_pts = np.asarray(centers['means'])
            covariances = centers.get('covariances', None)
        else:
            center_pts = np.asarray(centers)
            covariances = None
        has_covs = covariances is not None

        # Marker colours
        if show_soft and responsibilities is not None:
            marker_colors = blend_colors(responsibilities)
        else:
            marker_colors = [CLUSTER_COLORS[int(l) % len(CLUSTER_COLORS)] for l in labels]

        # Data points
        fig.add_trace(go.Scatter(
            x=points[:, 0], y=points[:, 1],
            mode='markers',
            marker=dict(size=5, color=marker_colors, opacity=0.75,
                        line=dict(width=0.3, color='rgba(255,255,255,0.3)')),
            showlegend=False,
            hovertemplate='x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>',
        ))

        # Cluster boundaries
        if covariances is not None:
            # EM mode: draw learned covariance ellipses (1σ and 2σ)
            for j in range(len(covariances)):
                color = CLUSTER_COLORS[j % len(CLUSTER_COLORS)]
                for n_std in (1, 2):
                    ex, ey = _ellipse_points(center_pts[j], covariances[j],
                                             n_std=n_std, n_points=120)
                    fig.add_trace(go.Scatter(
                        x=ex, y=ey, mode='lines',
                        line=dict(color=color, width=2 if n_std == 1 else 1),
                        opacity=0.8 if n_std == 1 else 0.4,
                        showlegend=False,
                        hoverinfo='skip',
                    ))
        else:
            # K-Means mode: draw circles (K-Means assumes spherical clusters)
            for j in range(K):
                members = points[labels == j]
                if len(members) < 2:
                    continue
                radius = float(np.mean(np.linalg.norm(members - center_pts[j], axis=1)))
                color = CLUSTER_COLORS[j % len(CLUSTER_COLORS)]
                t = np.linspace(0, 2 * np.pi, 100)
                fig.add_trace(go.Scatter(
                    x=center_pts[j, 0] + radius * np.cos(t),
                    y=center_pts[j, 1] + radius * np.sin(t),
                    mode='lines',
                    line=dict(color=color, width=2),
                    opacity=0.6,
                    showlegend=False, hoverinfo='skip',
                ))

        # Centers
        fig.add_trace(go.Scatter(
            x=center_pts[:, 0], y=center_pts[:, 1],
            mode='markers',
            marker=dict(size=14, color='black', symbol='star',
                        line=dict(color='white', width=1.5)),
            name='Centers',
        ))

        fig.update_layout(
            **PLOTLY_LAYOUT,
            title=title,
            xaxis=dict(gridcolor=GRID_COLOR, zeroline=False),
            yaxis=dict(gridcolor=GRID_COLOR, zeroline=False,
                       scaleanchor='x', scaleratio=1),
            height=600,
        )

        return fig

    def build_convergence_plot(self, history, title="EM Convergence"):
        return convergence_chart(history, title=title)
