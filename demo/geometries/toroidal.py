import numpy as np
import plotly.graph_objects as go
from scipy.special import i0e  # exponentially scaled I_0
from .base import GeometryBase
from demo.theme import CLUSTER_COLORS, PLOTLY_LAYOUT_3D, blend_colors, hex_to_rgb, BG_COLOR, GRID_COLOR
from demo.plots.plotly_figures import convergence_chart, torus_surface


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def wrap_to_2pi(x):
    """Wrap angles to [0, 2*pi)."""
    return x % (2 * np.pi)


def toroidal_distance(a, b):
    """Toroidal distance between points on [0, 2*pi)^2.

    Computes sqrt of sum of squared shortest-arc distances per dimension.
    """
    diff = np.abs(a - b)
    diff = np.minimum(diff, 2 * np.pi - diff)
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def circular_mean(angles, weights=None):
    """Weighted circular (Frechet) mean of angles."""
    if weights is None:
        weights = np.ones(len(angles))
    S = np.sum(weights * np.sin(angles))
    C = np.sum(weights * np.cos(angles))
    return np.arctan2(S, C) % (2 * np.pi)


def log_von_mises_pdf_1d(x, mu, kappa):
    """Log of von Mises PDF using i0e for numerical stability.

    log p(x|mu,kappa) = kappa*(cos(x-mu)-1) - log(2*pi) - log(i0e(kappa))
    """
    return kappa * (np.cos(x - mu) - 1) - np.log(2 * np.pi) - np.log(i0e(kappa))


def log_von_mises_pdf_2d(x, mu, kappas):
    """Log of product-of-von-Mises PDF on the 2D torus.

    Parameters
    ----------
    x : ndarray, shape (..., 2)
    mu : ndarray, shape (2,)
    kappas : ndarray, shape (2,)

    Returns
    -------
    ndarray — log density for each point.
    """
    return (log_von_mises_pdf_1d(x[..., 0], mu[0], kappas[0])
            + log_von_mises_pdf_1d(x[..., 1], mu[1], kappas[1]))


def estimate_kappa_1d(R_bar):
    """Mardia-Jupp approximation: kappa ~ R_bar*(2 - R_bar^2)/(1 - R_bar^2)."""
    R_bar = np.clip(R_bar, 1e-10, 1 - 1e-10)
    return R_bar * (2 - R_bar ** 2) / (1 - R_bar ** 2)


def _log_sum_exp(log_vals, axis=None):
    """Numerically stable log-sum-exp."""
    max_val = np.max(log_vals, axis=axis, keepdims=True)
    return max_val.squeeze(axis=axis) + np.log(
        np.sum(np.exp(log_vals - max_val), axis=axis)
    )


def torus_embed(theta, phi, R=3, r=1):
    """Convert flat torus coordinates (theta, phi) to 3D (x, y, z)."""
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    return x, y, z


# ---------------------------------------------------------------------------
# Geometry class
# ---------------------------------------------------------------------------

class ToroidalGeometry(GeometryBase):
    name = "Torus (S¹ × S¹)"
    plot_type = "3d"
    default_K = 5
    default_n_points = 300
    spread_range = (2.0, 50.0)  # kappa range
    spread_label = "Concentration (κ)"
    spread_default = 15.0
    spread_help = "Higher κ = tighter clusters. Low values spread across the torus."
    distance_options = {
        "Toroidal (Wrapped)": "toroidal",
    }

    # ------------------------------------------------------------------
    # Data generation
    # ------------------------------------------------------------------

    def generate_data(self, K, n_points, spread, seed):
        """Generate K clusters on [0, 2*pi) x [0, 2*pi) using von Mises."""
        rng = np.random.RandomState(seed)
        n_per = n_points // K
        extras = n_points - n_per * K

        means = rng.uniform(0, 2 * np.pi, size=(K, 2))
        # Make some clusters straddle the wraparound boundary
        for k in range(K):
            if rng.rand() < 0.4:
                dim = rng.randint(2)
                means[k, dim] = rng.choice([
                    rng.uniform(0, 0.3),
                    rng.uniform(2 * np.pi - 0.3, 2 * np.pi),
                ])

        kappas = np.zeros((K, 2))
        points_list = []
        labels_list = []

        # Wide spread variation so EM (which estimates per-cluster kappa)
        # has an advantage over K-Means (which assumes equal spread)
        spread_factors = np.array([0.2, 1.0, 3.0, 0.5, 2.5, 4.0, 0.3, 1.5])[:K]
        rng.shuffle(spread_factors)

        for k in range(K):
            nk = n_per + (1 if k < extras else 0)
            kappas[k, 0] = spread * spread_factors[k]
            kappas[k, 1] = spread * spread_factors[k] * (0.6 + rng.rand() * 0.8)

            theta = wrap_to_2pi(rng.vonmises(means[k, 0], kappas[k, 0], size=nk))
            phi = wrap_to_2pi(rng.vonmises(means[k, 1], kappas[k, 1], size=nk))
            points_list.append(np.column_stack([theta, phi]))
            labels_list.append(np.full(nk, k, dtype=int))

        points = np.vstack(points_list)
        labels = np.concatenate(labels_list)
        return {
            'points': points,
            'labels': labels,
            'params': {'means': means, 'kappas': kappas},
        }

    # ------------------------------------------------------------------
    # K-means
    # ------------------------------------------------------------------

    def kmeans_init(self, points, K, seed=123, **kwargs):
        """Initialise toroidal K-means: pick K random points as centroids."""
        rng = np.random.RandomState(seed)
        N = len(points)
        idx = rng.choice(N, K, replace=False)
        centroids = points[idx].copy()
        return {
            'centroids': centroids,
            'labels': np.zeros(N, dtype=int),
            'cost_history': [],
            'iteration': 0,
        }

    def kmeans_step(self, points, state):
        """One iteration of toroidal K-means: assign by toroidal distance, update by circular mean."""
        centroids = state['centroids']
        N = len(points)
        K = len(centroids)

        # Assign
        dists = np.zeros((N, K))
        for k in range(K):
            dists[:, k] = toroidal_distance(points, centroids[k])
        labels = np.argmin(dists, axis=1)
        cost = np.sum(dists[np.arange(N), labels] ** 2)
        cost_history = state['cost_history'] + [cost]

        # Update centroids via circular mean
        new_centroids = np.zeros_like(centroids)
        for k in range(K):
            mask = labels == k
            if mask.sum() == 0:
                new_centroids[k] = centroids[k]
            else:
                for d in range(2):
                    new_centroids[k, d] = circular_mean(points[mask, d])

        shift = toroidal_distance(centroids, new_centroids).sum()

        return {
            'centroids': new_centroids,
            'labels': labels,
            'cost_history': cost_history,
            'iteration': state['iteration'] + 1,
            '_shift': shift,
        }

    def kmeans_full(self, points, K, max_iter=100, seed=123, **kwargs):
        """Toroidal K-means using wrapped distance and circular mean."""
        state = self.kmeans_init(points, K, seed=seed, **kwargs)

        for _ in range(max_iter):
            state = self.kmeans_step(points, state)
            if state['_shift'] < 1e-8:
                break

        return {'labels': state['labels'], 'centroids': state['centroids'], 'cost_history': state['cost_history']}

    # ------------------------------------------------------------------
    # EM — initialization
    # ------------------------------------------------------------------

    def em_init(self, points, K, seed=123, **kwargs):
        """Initialize product-of-von-Mises mixture."""
        rng = np.random.RandomState(seed)
        N = points.shape[0]

        idx = rng.choice(N, K, replace=False)
        means = points[idx].copy()
        kappas = np.full((K, 2), 5.0)
        weights = np.full(K, 1.0 / K)
        responsibilities = np.full((N, K), 1.0 / K)

        return {
            'means': means,
            'kappas': kappas,
            'weights': weights,
            'responsibilities': responsibilities,
            'log_likelihood': -np.inf,
            'iteration': 0,
            'll_history': [],
        }

    # ------------------------------------------------------------------
    # EM — single step
    # ------------------------------------------------------------------

    def em_step(self, points, state):
        """One EM iteration for product-of-von-Mises mixture."""
        N = points.shape[0]
        K = state['means'].shape[0]
        mu = state['means'].copy()
        kappa = state['kappas'].copy()
        pi_k = state['weights'].copy()

        # === E-step ===
        log_component = np.zeros((N, K))
        for k in range(K):
            for d in range(2):
                log_component[:, k] += log_von_mises_pdf_1d(
                    points[:, d], mu[k, d], kappa[k, d]
                )

        log_weighted = log_component + np.log(pi_k)[np.newaxis, :]
        ll = np.sum(_log_sum_exp(log_weighted, axis=1))

        log_resp = log_weighted - _log_sum_exp(log_weighted, axis=1)[:, np.newaxis]
        resp = np.exp(log_resp)

        # === M-step ===
        Nk = resp.sum(axis=0)
        Nk = np.maximum(Nk, 1e-10)

        pi_k = Nk / N

        for k in range(K):
            for d in range(2):
                C_bar = np.sum(resp[:, k] * np.cos(points[:, d]))
                S_bar = np.sum(resp[:, k] * np.sin(points[:, d]))
                mu[k, d] = np.arctan2(S_bar, C_bar) % (2 * np.pi)
                R_bar = np.sqrt(C_bar ** 2 + S_bar ** 2) / Nk[k]
                kappa[k, d] = estimate_kappa_1d(R_bar)

        ll_history = state['ll_history'] + [ll]

        return {
            'means': mu,
            'kappas': kappa,
            'weights': pi_k,
            'responsibilities': resp,
            'log_likelihood': ll,
            'iteration': state['iteration'] + 1,
            'll_history': ll_history,
        }

    # ------------------------------------------------------------------
    # EM — full run
    # ------------------------------------------------------------------

    def em_full(self, points, K, max_iter=200, seed=123):
        """Full EM until convergence."""
        state = self.em_init(points, K, seed=seed)
        tol = 1e-8

        for _ in range(max_iter):
            state = self.em_step(points, state)
            if len(state['ll_history']) >= 2:
                if abs(state['ll_history'][-1] - state['ll_history'][-2]) < tol:
                    break

        return state

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def build_plot(self, points, labels, centers, responsibilities=None,
                   show_soft=False, title="", true_labels=None):
        """3D Plotly figure with torus surface and embedded points."""
        if isinstance(centers, dict):
            centers = np.asarray(centers["means"])
        R, r = 3, 1
        K = len(centers)
        fig = go.Figure()

        # Torus surface
        fig.add_trace(torus_surface(R=R, r=r, opacity=0.12))

        # Determine point colors
        if show_soft and responsibilities is not None:
            point_colors = blend_colors(responsibilities)
        else:
            point_colors = [CLUSTER_COLORS[int(l) % len(CLUSTER_COLORS)] for l in labels]

        # Embed data points on torus
        x_pts, y_pts, z_pts = torus_embed(points[:, 0], points[:, 1], R=R, r=r)

        fig.add_trace(go.Scatter3d(
            x=x_pts, y=y_pts, z=z_pts,
            mode='markers',
            marker=dict(size=3, color=point_colors, opacity=0.85),
            name='Data',
            showlegend=False,
        ))

        # Cluster spread rings on the torus surface
        for k in range(K):
            members = points[labels == k]
            if len(members) < 2:
                continue
            dists = toroidal_distance(members, centers[k])
            mean_dist = float(np.mean(dists))
            color = CLUSTER_COLORS[k % len(CLUSTER_COLORS)]

            # Draw a ring at angular offset = mean_dist around the center
            t = np.linspace(0, 2 * np.pi, 60)
            ring_theta = centers[k, 0] + mean_dist * np.cos(t)
            ring_phi = centers[k, 1] + mean_dist * np.sin(t)
            rx, ry, rz = torus_embed(ring_theta, ring_phi, R=R, r=r)
            fig.add_trace(go.Scatter3d(
                x=rx, y=ry, z=rz,
                mode='lines',
                line=dict(color=color, width=4),
                opacity=0.7,
                showlegend=False, hoverinfo='skip',
            ))

        # Embed and plot cluster centers
        x_c, y_c, z_c = torus_embed(centers[:, 0], centers[:, 1], R=R, r=r)
        for k in range(K):
            color = CLUSTER_COLORS[k % len(CLUSTER_COLORS)]
            fig.add_trace(go.Scatter3d(
                x=[x_c[k]], y=[y_c[k]], z=[z_c[k]],
                mode='markers',
                marker=dict(
                    size=10,
                    color=color,
                    symbol='diamond',
                    line=dict(color='white', width=2),
                ),
                name=f'Center {k}',
            ))

        fig.update_layout(
            **PLOTLY_LAYOUT_3D,
            title=title,
            height=650,
            showlegend=True,
        )

        return fig

    # ------------------------------------------------------------------
    # Convergence plot
    # ------------------------------------------------------------------

    def build_convergence_plot(self, history, title=""):
        """Delegate to shared convergence_chart helper."""
        return convergence_chart(history, title=title)
