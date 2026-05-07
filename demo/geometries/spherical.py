import numpy as np
import plotly.graph_objects as go
from scipy.special import ive
from .base import GeometryBase
from demo.theme import CLUSTER_COLORS, PLOTLY_LAYOUT_3D, blend_colors, hex_to_rgb, BG_COLOR, GRID_COLOR
from demo.plots.plotly_figures import convergence_chart, sphere_wireframe


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _sample_vmf(mu, kappa, n, rng):
    """
    Sample *n* points from vMF(mu, kappa) on S^2 in R^3.

    Uses the Wood (1994) rejection-sampling algorithm:
      1. Sample the "height" w along the mean direction.
      2. Sample a uniform direction in the tangent plane.
      3. Combine to get a point around the North Pole.
      4. Rotate from the North Pole to *mu* via Rodrigues' formula.
    """
    p = 3
    mu = np.asarray(mu, dtype=np.float64)
    mu = mu / np.linalg.norm(mu)

    # Rejection-sampling envelope (Wood 1994)
    b = (p - 1) / (2 * kappa + np.sqrt(4 * kappa ** 2 + (p - 1) ** 2))
    x0 = (1 - b) / (1 + b)
    c = kappa * x0 + (p - 1) * np.log(1 - x0 ** 2)

    samples = []
    while len(samples) < n:
        batch = max(n * 3, 100)
        z = rng.beta((p - 1) / 2, (p - 1) / 2, size=batch)
        w_candidates = (1 - (1 + b) * z) / (1 - (1 - b) * z)
        u = rng.uniform(0, 1, size=batch)
        criterion = kappa * w_candidates + (p - 1) * np.log(1 - x0 * w_candidates) - c
        accepted = w_candidates[np.log(u) < criterion]
        samples.extend(accepted.tolist())

    w = np.array(samples[:n])

    # Uniform direction in the tangent plane
    v = rng.normal(size=(n, p - 1))
    v = v / np.linalg.norm(v, axis=1, keepdims=True)

    # Points around the North Pole [0, 0, 1]
    r = np.sqrt(1 - w ** 2)
    points_np = np.zeros((n, p))
    points_np[:, 0] = r * v[:, 0]
    points_np[:, 1] = r * v[:, 1]
    points_np[:, 2] = w

    # Rotate from North Pole to mu (Rodrigues' formula)
    north = np.array([0.0, 0.0, 1.0])
    if np.allclose(mu, north):
        return points_np
    elif np.allclose(mu, -north):
        points_np[:, 2] *= -1
        return points_np

    axis = np.cross(north, mu)
    axis = axis / np.linalg.norm(axis)
    cos_angle = np.dot(north, mu)
    sin_angle = np.sqrt(1 - cos_angle ** 2)

    K_mat = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    R = np.eye(3) + sin_angle * K_mat + (1 - cos_angle) * K_mat @ K_mat
    return (R @ points_np.T).T


def _log_vmf_normalizing_constant(kappa, p=3):
    """Log of the vMF normalising constant C_p(kappa), using ive for stability."""
    v = p / 2.0 - 1
    log_ive = np.log(np.clip(ive(v, kappa), 1e-300, None))
    log_bessel = log_ive + kappa
    log_C = (
        v * np.log(np.clip(kappa, 1e-300, None))
        - (p / 2.0) * np.log(2 * np.pi)
        - log_bessel
    )
    return log_C


def _log_vmf_pdf(x, mu, kappa, p=3):
    """Log vMF density.  x: (N, p), mu: (p,), kappa: scalar."""
    log_C = _log_vmf_normalizing_constant(kappa, p)
    return log_C + kappa * (x @ mu)


def _log_sum_exp(log_vals, axis=None):
    """Numerically stable log-sum-exp."""
    max_val = np.max(log_vals, axis=axis, keepdims=True)
    return np.squeeze(max_val, axis=axis) + np.log(
        np.sum(np.exp(log_vals - max_val), axis=axis)
    )


def _estimate_kappa(R_bar, p=3):
    """Banerjee et al. (2005) approximation for kappa from mean resultant length."""
    R_bar = np.clip(R_bar, 1e-10, 1 - 1e-10)
    return R_bar * (p - R_bar ** 2) / (1 - R_bar ** 2)


def _haversine(u, v):
    """Great-circle distance between unit vectors (supports broadcasting)."""
    cos_angle = np.clip(np.sum(u * v, axis=-1), -1, 1)
    return np.arccos(cos_angle)


def _fibonacci_sphere(K):
    """Return K roughly-evenly-spaced points on the unit sphere (Fibonacci spiral)."""
    indices = np.arange(K, dtype=float)
    phi = np.arccos(1 - 2 * (indices + 0.5) / K)
    theta = np.pi * (1 + np.sqrt(5)) * indices
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack([x, y, z])


# ---------------------------------------------------------------------------
# Geometry class
# ---------------------------------------------------------------------------

class SphericalGeometry(GeometryBase):
    name = "Sphere (S\u00b2)"
    plot_type = "3d"
    default_K = 5
    default_n_points = 300
    spread_range = (5.0, 100.0)
    spread_label = "Concentration (\u03ba)"
    spread_default = 40.0
    spread_help = "Higher κ = tighter clusters concentrated around their centers."
    distance_options = {
        "Great-Circle (Haversine)": "haversine",
        "Cosine": "cosine",
    }

    # ------------------------------------------------------------------
    # Data generation
    # ------------------------------------------------------------------
    def generate_data(self, K, n_points, spread, seed):
        rng = np.random.default_rng(seed)
        centers = _fibonacci_sphere(K)
        n_per = [n_points // K] * K
        for i in range(n_points % K):
            n_per[i] += 1

        # Wide kappa variation: some clusters very tight, others very diffuse
        # This is where EM shines — it estimates per-cluster kappa
        spread_factors = np.array([0.15, 1.0, 4.0, 0.4, 2.5, 5.0, 0.3, 1.5])[:K]
        rng.shuffle(spread_factors)

        all_points, all_labels, kappas = [], [], []
        for k in range(K):
            kappa = spread * spread_factors[k]
            kappas.append(kappa)
            pts = _sample_vmf(centers[k], kappa, n_per[k], rng)
            all_points.append(pts)
            all_labels.append(np.full(n_per[k], k))

        points = np.vstack(all_points)
        labels = np.concatenate(all_labels)
        return {
            "points": points,
            "labels": labels,
            "params": {
                "means": centers,
                "kappas": np.array(kappas),
            },
        }

    # ------------------------------------------------------------------
    # Spherical K-means
    # ------------------------------------------------------------------
    @staticmethod
    def _sphere_distance(points, centroid, metric):
        """Compute distance from each point to centroid on the sphere."""
        if metric == "cosine":
            return 1.0 - np.clip(points @ centroid, -1, 1)
        else:  # haversine
            return _haversine(points, centroid)

    def kmeans_init(self, points, K, seed=123, **kwargs):
        """Initialise spherical K-means: pick K random points, normalize to unit sphere."""
        rng = np.random.default_rng(seed)
        n = len(points)
        idx = rng.choice(n, size=K, replace=False)
        centroids = points[idx].copy()
        centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        distance = kwargs.get('distance', 'haversine')
        return {
            'centroids': centroids,
            'labels': np.zeros(n, dtype=int),
            'cost_history': [],
            'iteration': 0,
            '_distance': distance,
        }

    def kmeans_step(self, points, state):
        """One iteration of spherical K-means using the selected distance."""
        centroids = state['centroids']
        metric = state.get('_distance', 'haversine')
        n = len(points)
        K = len(centroids)

        # Assignment
        dist_matrix = np.zeros((n, K))
        for j in range(K):
            dist_matrix[:, j] = self._sphere_distance(points, centroids[j], metric)
        labels = np.argmin(dist_matrix, axis=1)

        total_dist = float(np.sum(dist_matrix[np.arange(n), labels]))
        cost_history = state['cost_history'] + [total_dist]

        # Update: spherical mean (normalised Euclidean mean — same for both metrics)
        rng = np.random.default_rng(state['iteration'])
        new_centroids = np.zeros_like(centroids)
        for j in range(K):
            members = points[labels == j]
            if len(members) == 0:
                new_centroids[j] = points[rng.choice(n)]
            else:
                mean_vec = members.mean(axis=0)
                norm = np.linalg.norm(mean_vec)
                new_centroids[j] = mean_vec / max(norm, 1e-10)

        shift = float(np.max(_haversine(centroids, new_centroids)))

        return {
            'centroids': new_centroids,
            'labels': labels,
            'cost_history': cost_history,
            'iteration': state['iteration'] + 1,
            '_shift': shift,
            '_distance': metric,
        }

    def kmeans_full(self, points, K, max_iter=100, seed=123, **kwargs):
        """Spherical K-means to convergence."""
        state = self.kmeans_init(points, K, seed=seed, **kwargs)

        for _ in range(max_iter):
            state = self.kmeans_step(points, state)
            if state['_shift'] < 1e-6:
                break

        return {"labels": state['labels'], "centroids": state['centroids'], "cost_history": state['cost_history']}

    # ------------------------------------------------------------------
    # EM  —  von Mises-Fisher mixture
    # ------------------------------------------------------------------
    def em_init(self, points, K, seed=123, **kwargs):
        rng = np.random.default_rng(seed)
        N, p = points.shape

        idx = rng.choice(N, size=K, replace=False)
        means = points[idx].copy()
        means = means / np.linalg.norm(means, axis=1, keepdims=True)

        kappas = np.full(K, 20.0)
        weights = np.ones(K) / K
        responsibilities = np.full((N, K), 1.0 / K)

        # Compute initial log-likelihood
        log_resp = np.zeros((N, K))
        for j in range(K):
            log_resp[:, j] = np.log(weights[j]) + _log_vmf_pdf(points, means[j], kappas[j], p)
        ll = np.sum(_log_sum_exp(log_resp, axis=1))

        return {
            "means": means,
            "kappas": kappas,
            "weights": weights,
            "responsibilities": responsibilities,
            "log_likelihood": ll,
            "iteration": 0,
            "ll_history": [],
        }

    def em_step(self, points, state):
        N, p = points.shape
        K = len(state["weights"])
        means = state["means"].copy()
        kappas = state["kappas"].copy()
        weights = state["weights"].copy()

        # --- E-step ---
        log_resp = np.zeros((N, K))
        for j in range(K):
            log_resp[:, j] = np.log(weights[j]) + _log_vmf_pdf(points, means[j], kappas[j], p)

        log_resp_norm = _log_sum_exp(log_resp, axis=1)  # (N,)
        log_resp_normalized = log_resp - log_resp_norm[:, np.newaxis]
        responsibilities = np.exp(log_resp_normalized)

        ll = np.sum(log_resp_norm)

        # --- M-step ---
        rng = np.random.default_rng(state["iteration"])
        for j in range(K):
            r_j = responsibilities[:, j]
            N_j = np.sum(r_j)

            if N_j < 1e-10:
                # Dead component — reinitialise
                means[j] = points[rng.choice(N)]
                kappas[j] = 20.0
                weights[j] = 1.0 / K
                continue

            r_bar_vec = np.sum(r_j[:, np.newaxis] * points, axis=0)
            r_bar_norm = np.linalg.norm(r_bar_vec)

            means[j] = r_bar_vec / max(r_bar_norm, 1e-10)

            R_bar = r_bar_norm / N_j
            kappas[j] = np.clip(_estimate_kappa(R_bar, p), 0.1, 1000.0)

            weights[j] = N_j / N

        ll_history = state["ll_history"] + [ll]

        return {
            "means": means,
            "kappas": kappas,
            "weights": weights,
            "responsibilities": responsibilities,
            "log_likelihood": ll,
            "iteration": state["iteration"] + 1,
            "ll_history": ll_history,
        }

    def em_full(self, points, K, max_iter=200, seed=123):
        state = self.em_init(points, K, seed=seed)
        tol = 1e-6

        for _ in range(max_iter):
            state = self.em_step(points, state)
            if len(state["ll_history"]) >= 2:
                if abs(state["ll_history"][-1] - state["ll_history"][-2]) < tol:
                    break

        state["labels"] = np.argmax(state["responsibilities"], axis=1)
        return state

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def build_plot(self, points, labels, centers, responsibilities=None,
                   show_soft=False, title="", true_labels=None):
        if isinstance(centers, dict):
            centers = np.asarray(centers["means"])
        fig = go.Figure()

        # Translucent sphere wireframe
        fig.add_trace(sphere_wireframe(opacity=0.08))

        N = len(points)
        K = len(centers)

        # Point colours
        if show_soft and responsibilities is not None:
            colors = blend_colors(responsibilities)
        else:
            colors = [CLUSTER_COLORS[int(labels[i]) % len(CLUSTER_COLORS)] for i in range(N)]

        fig.add_trace(go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode="markers",
            marker=dict(size=3, color=colors, opacity=0.85),
            name="Points",
            hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
        ))

        # Cluster spread rings — small circles on the sphere at mean angular distance
        for j in range(K):
            members = points[labels == j]
            if len(members) < 2:
                continue
            dists = _haversine(members, centers[j])
            mean_ang = float(np.mean(dists))
            if mean_ang < 0.01 or mean_ang > np.pi / 2:
                continue

            # Build a small circle of radius mean_ang around centers[j]
            mu = centers[j]
            # Find an orthonormal basis for the tangent plane at mu
            if abs(mu[2]) < 0.9:
                ref = np.array([0., 0., 1.])
            else:
                ref = np.array([1., 0., 0.])
            e1 = np.cross(mu, ref)
            e1 = e1 / np.linalg.norm(e1)
            e2 = np.cross(mu, e1)
            e2 = e2 / np.linalg.norm(e2)

            t = np.linspace(0, 2 * np.pi, 80)
            ring = (np.cos(mean_ang) * mu[np.newaxis, :]
                    + np.sin(mean_ang) * (np.cos(t)[:, np.newaxis] * e1
                                          + np.sin(t)[:, np.newaxis] * e2))
            color = CLUSTER_COLORS[j % len(CLUSTER_COLORS)]
            fig.add_trace(go.Scatter3d(
                x=ring[:, 0], y=ring[:, 1], z=ring[:, 2],
                mode="lines",
                line=dict(color=color, width=4),
                opacity=0.7,
                showlegend=False, hoverinfo="skip",
            ))

        # Cluster centres
        fig.add_trace(go.Scatter3d(
            x=centers[:, 0], y=centers[:, 1], z=centers[:, 2],
            mode="markers",
            marker=dict(
                size=10,
                color="black",
                symbol="diamond",
                line=dict(color="white", width=2),
            ),
            name="Centers",
            hovertemplate="cx=%{x:.3f}<br>cy=%{y:.3f}<br>cz=%{z:.3f}<extra></extra>",
        ))

        fig.update_layout(
            **PLOTLY_LAYOUT_3D,
            title=title,
            height=650,
            scene_camera=dict(
                eye=dict(x=1.4, y=1.4, z=0.8),
                up=dict(x=0, y=0, z=1),
            ),
            showlegend=True,
        )

        return fig

    def build_convergence_plot(self, history, title=""):
        return convergence_chart(history, ylabel="Log-Likelihood", title=title)
