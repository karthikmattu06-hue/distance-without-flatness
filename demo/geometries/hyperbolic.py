import numpy as np
import plotly.graph_objects as go
from functools import lru_cache
from .base import GeometryBase
from demo.theme import CLUSTER_COLORS, PLOTLY_LAYOUT, blend_colors, hex_to_rgb, BG_COLOR, GRID_COLOR
from demo.plots.plotly_figures import convergence_chart, disk_boundary


# ---------------------------------------------------------------------------
# Helper functions for Poincaré disk geometry
# ---------------------------------------------------------------------------

def mobius_addition(a, b):
    """Möbius addition a ⊕ b in the Poincaré disk.

    a ⊕ b = ((1 + 2<a,b> + |b|²)a + (1 - |a|²)b)
             / (1 + 2<a,b> + |a|²|b|²)
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    a_dot_b = np.sum(a * b, axis=-1, keepdims=True)
    a_sq = np.sum(a ** 2, axis=-1, keepdims=True)
    b_sq = np.sum(b ** 2, axis=-1, keepdims=True)

    numerator = (1 + 2 * a_dot_b + b_sq) * a + (1 - a_sq) * b
    denominator = 1 + 2 * a_dot_b + a_sq * b_sq

    result = numerator / denominator

    # Project back inside the disk for numerical safety
    norm = np.sqrt(np.sum(result ** 2, axis=-1, keepdims=True))
    max_norm = 1 - 1e-7
    result = np.where(norm > max_norm, result * max_norm / norm, result)
    return result


def poincare_distance(u, v):
    """Hyperbolic distance in the Poincaré disk model.

    d(u,v) = arccosh(1 + 2||u-v||² / ((1-||u||²)(1-||v||²)))
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    diff_sq = np.sum((u - v) ** 2, axis=-1)
    u_sq = np.sum(u ** 2, axis=-1)
    v_sq = np.sum(v ** 2, axis=-1)

    denom = np.clip((1 - u_sq) * (1 - v_sq), 1e-15, None)
    arg = 1 + 2 * diff_sq / denom
    arg = np.clip(arg, 1.0, None)  # arccosh domain: [1, inf)
    return np.arccosh(arg)


def poincare_distance_matrix(points, centers):
    """Distance matrix between all *points* and all *centers*.

    Returns shape ``(len(points), len(centers))``.
    """
    n_c = len(centers)
    dist = np.zeros((len(points), n_c))
    for j in range(n_c):
        dist[:, j] = poincare_distance(points, centers[j])
    return dist


def poincare_to_klein(p):
    """Map from Poincaré disk to Klein disk model."""
    p = np.asarray(p, dtype=np.float64)
    p_sq = np.sum(p ** 2, axis=-1, keepdims=True)
    return 2 * p / (1 + p_sq)


def klein_to_poincare(k):
    """Map from Klein disk to Poincaré disk model."""
    k = np.asarray(k, dtype=np.float64)
    k_sq = np.sum(k ** 2, axis=-1, keepdims=True)
    denom = 1 + np.sqrt(np.clip(1 - k_sq, 1e-15, None))
    return k / denom


def weighted_einstein_midpoint(points, weights):
    """Weighted Einstein midpoint (hyperbolic centroid).

    1. Map points Poincaré → Klein
    2. Weighted average with Lorentz factors × responsibility weights
    3. Map Klein → Poincaré
    """
    points = np.asarray(points, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    p_sq = np.sum(points ** 2, axis=-1)
    gamma = 1.0 / np.sqrt(np.clip(1 - p_sq, 1e-15, None))

    klein_points = poincare_to_klein(points)

    combined = gamma * weights
    total = np.sum(combined)
    if total < 1e-15:
        return np.array([0.0, 0.0])

    klein_mean = np.sum(combined[:, np.newaxis] * klein_points, axis=0) / total

    km_norm = np.linalg.norm(klein_mean)
    if km_norm >= 1.0:
        klein_mean = klein_mean * 0.999 / km_norm

    return klein_to_poincare(klein_mean)


def einstein_midpoint(points):
    """Unweighted Einstein midpoint (all weights = 1)."""
    return weighted_einstein_midpoint(points, np.ones(len(points)))


@lru_cache(maxsize=512)
def approximate_log_Z(sigma):
    """Numerically approximate log Z(σ) for the wrapped normal on the disk.

    Z(σ) = ∫_D exp(-d_H(x,0)²/(2σ²)) · 4/(1-r²)² dx dy
    By isometry invariance the result is independent of the center μ.
    """
    grid_res = 200
    x = np.linspace(-0.995, 0.995, grid_res)
    dx = x[1] - x[0]
    xx, yy = np.meshgrid(x, x)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=-1)

    r_sq = np.sum(grid ** 2, axis=1)
    inside = r_sq < 0.99

    r = np.sqrt(r_sq[inside])
    d_H = 2 * np.arctanh(np.clip(r, 0, 0.999))

    log_kernel = -d_H ** 2 / (2 * sigma ** 2)
    area_element = 4.0 / (1 - r_sq[inside]) ** 2

    Z = np.sum(np.exp(log_kernel) * area_element) * dx * dx
    return float(np.log(max(Z, 1e-30)))


def _logsumexp(a, axis=None):
    """Numerically stable log-sum-exp."""
    a_max = np.max(a, axis=axis, keepdims=True)
    out = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    if axis is not None:
        out = np.squeeze(out, axis=axis)
    return out


def _clip_to_disk(pts):
    """Ensure all points have norm < 1 - 1e-7."""
    pts = np.asarray(pts, dtype=np.float64)
    norms = np.sqrt(np.sum(pts ** 2, axis=-1, keepdims=True))
    max_norm = 1 - 1e-7
    return np.where(norms > max_norm, pts * max_norm / norms, pts)


# ---------------------------------------------------------------------------
# Geometry class
# ---------------------------------------------------------------------------

class HyperbolicGeometry(GeometryBase):
    name = "Poincaré Disk"
    plot_type = "2d"
    default_K = 5
    default_n_points = 300
    spread_range = (0.05, 0.5)
    spread_label = "Spread (σ)"
    spread_default = 0.15
    spread_help = "Controls how far points scatter from cluster centers inside the disk."
    distance_options = {
        "Poincaré Distance": "poincare",
    }

    # ------------------------------------------------------------------
    # 1. Data generation
    # ------------------------------------------------------------------
    def generate_data(self, K, n_points, spread, seed):
        """Generate *K* clusters inside the Poincaré disk using Möbius addition.

        Centers are placed at radii between 0.3 and 0.85 from the origin,
        with angles evenly spaced around the disk.  Points are produced by
        adding small random perturbations via Möbius addition.
        """
        rng = np.random.default_rng(seed)

        # Place centers at varying radii, evenly spaced in angle
        angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
        radii = np.linspace(0.3, 0.85, K)
        rng.shuffle(radii)

        centers = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=-1)

        # Deliberately vary the spread per cluster so EM can shine
        # Factors range from 0.3x to 3x of the base spread
        spread_factors = np.array([0.3, 1.0, 2.5, 0.6, 1.8, 3.0, 0.4, 1.5])[:K]
        rng.shuffle(spread_factors)
        sigmas = spread * spread_factors

        points_per = n_points // K
        all_points = []
        all_labels = []

        for i in range(K):
            n = points_per if i < K - 1 else n_points - points_per * (K - 1)
            perturbations = rng.normal(0, sigmas[i], size=(n, 2))
            # Clip perturbation norms so they stay well inside the disk
            pnorms = np.sqrt(np.sum(perturbations ** 2, axis=1, keepdims=True))
            perturbations = np.where(pnorms > 0.4, perturbations * 0.4 / pnorms, perturbations)

            cluster_pts = mobius_addition(centers[i], perturbations)
            all_points.append(cluster_pts)
            all_labels.append(np.full(n, i))

        points = _clip_to_disk(np.vstack(all_points))
        labels = np.concatenate(all_labels)

        return {
            "points": points,
            "labels": labels,
            "params": {"means": centers, "sigmas": sigmas},
        }

    # ------------------------------------------------------------------
    # 2. K-means
    # ------------------------------------------------------------------
    def kmeans_init(self, points, K, seed=123, **kwargs):
        """Initialise hyperbolic K-means: pick K random points as centroids."""
        rng = np.random.default_rng(seed)
        n = len(points)
        idx = rng.choice(n, size=K, replace=False)
        centroids = points[idx].copy()
        return {
            'centroids': centroids,
            'labels': np.zeros(n, dtype=int),
            'cost_history': [],
            'iteration': 0,
        }

    def kmeans_step(self, points, state):
        """One iteration of hyperbolic K-means: assign by Poincare distance, update by Einstein midpoint."""
        centroids = state['centroids']
        n = len(points)
        K = len(centroids)

        dist_mat = poincare_distance_matrix(points, centroids)
        labels = np.argmin(dist_mat, axis=1)
        cost = float(np.sum(dist_mat[np.arange(n), labels]))
        cost_history = state['cost_history'] + [cost]

        rng = np.random.default_rng(state['iteration'])
        new_centroids = np.zeros_like(centroids)
        for j in range(K):
            members = points[labels == j]
            if len(members) == 0:
                new_centroids[j] = points[rng.choice(n)]
            else:
                new_centroids[j] = einstein_midpoint(members)

        shift = np.max(poincare_distance(centroids, new_centroids))
        new_centroids = _clip_to_disk(new_centroids)

        return {
            'centroids': new_centroids,
            'labels': labels,
            'cost_history': cost_history,
            'iteration': state['iteration'] + 1,
            '_shift': shift,
        }

    def kmeans_full(self, points, K, max_iter=100, seed=123, **kwargs):
        """Hyperbolic K-means using Poincare distance and Einstein midpoint."""
        state = self.kmeans_init(points, K, seed=seed, **kwargs)

        for _ in range(max_iter):
            state = self.kmeans_step(points, state)
            if state['_shift'] < 1e-6:
                break

        return {"labels": state['labels'], "centroids": state['centroids'], "cost_history": state['cost_history']}

    # ------------------------------------------------------------------
    # 3. EM initialisation
    # ------------------------------------------------------------------
    def em_init(self, points, K, seed=123, **kwargs):
        """Initialise wrapped-normal mixture with a K-means warm start."""
        n = len(points)

        # Warm-start: a few iterations of hyperbolic K-means
        km = self.kmeans_full(points, K, max_iter=15, seed=seed)
        means = km["centroids"].copy()

        # Initial sigma per cluster = RMS Poincaré distance to centroid
        labels = km["labels"]
        sigmas = np.empty(K)
        for k in range(K):
            members = points[labels == k]
            if len(members) == 0:
                sigmas[k] = 0.2
            else:
                d = poincare_distance(members, means[k])
                sigmas[k] = max(float(np.sqrt(np.mean(d ** 2))), 0.05)

        weights = np.array([float(np.sum(labels == k)) for k in range(K)])
        weights = weights / weights.sum()

        # Initial responsibilities (hard from K-means)
        resp = np.zeros((n, K))
        resp[np.arange(n), labels] = 1.0

        # Initial log-likelihood
        ll = self._log_likelihood(points, means, sigmas, weights)

        return {
            "means": means,
            "sigmas": sigmas,
            "weights": weights,
            "responsibilities": resp,
            "log_likelihood": ll,
            "iteration": 0,
            "ll_history": [],
        }

    # ------------------------------------------------------------------
    # 4. Single EM step
    # ------------------------------------------------------------------
    def em_step(self, points, state):
        """One EM iteration for the wrapped-normal mixture."""
        means = state["means"]
        sigmas = state["sigmas"]
        weights = state["weights"]
        K = len(means)
        n = len(points)

        # ---- E-step ----
        dist_mat = poincare_distance_matrix(points, means)  # (N, K)

        log_density = np.empty((n, K))
        for k in range(K):
            s = float(np.clip(sigmas[k], 0.05, 3.0))
            log_Z = approximate_log_Z(round(s, 4))
            log_density[:, k] = (
                np.log(weights[k] + 1e-30) - log_Z - dist_mat[:, k] ** 2 / (2 * s ** 2)
            )

        log_norm = _logsumexp(log_density, axis=1)  # (N,)
        log_resp = log_density - log_norm[:, np.newaxis]
        resp = np.exp(log_resp)

        # Log-likelihood for this iteration
        ll = float(np.sum(log_norm))

        # ---- M-step ----
        N_k = np.sum(resp, axis=0)  # (K,)

        new_means = np.empty_like(means)
        new_sigmas = np.empty(K)
        for k in range(K):
            w = resp[:, k]
            if N_k[k] < 1e-10:
                new_means[k] = means[k]
                new_sigmas[k] = sigmas[k]
                continue

            new_means[k] = weighted_einstein_midpoint(points, w)

            # Weighted RMS Poincaré distance for sigma
            d_k = poincare_distance(points, new_means[k])
            new_sigmas[k] = max(float(np.sqrt(np.sum(w * d_k ** 2) / N_k[k])), 0.05)

        new_means = _clip_to_disk(new_means)
        new_weights = N_k / n

        iteration = state["iteration"] + 1
        ll_history = state["ll_history"] + [ll]

        return {
            "means": new_means,
            "sigmas": new_sigmas,
            "weights": new_weights,
            "responsibilities": resp,
            "log_likelihood": ll,
            "iteration": iteration,
            "ll_history": ll_history,
        }

    # ------------------------------------------------------------------
    # 5. Full EM
    # ------------------------------------------------------------------
    def em_full(self, points, K, max_iter=200, seed=123):
        """Run EM to convergence."""
        state = self.em_init(points, K, seed=seed)
        for _ in range(max_iter):
            prev_ll = state["log_likelihood"]
            state = self.em_step(points, state)
            if state["iteration"] > 1 and abs(state["log_likelihood"] - prev_ll) < 1e-4:
                break
        return state

    # ------------------------------------------------------------------
    # 6. Plotting
    # ------------------------------------------------------------------
    def build_plot(self, points, labels, centers, responsibilities=None,
                   show_soft=False, title="", true_labels=None):
        """2-D Plotly scatter inside the Poincaré unit disk."""
        if isinstance(centers, dict):
            centers = np.asarray(centers["means"])
        fig = go.Figure()
        K = len(centers)

        # Disk boundary
        fig.add_trace(disk_boundary())

        # Concentric hyperbolic-distance circles (d = 1, 2, 3)
        for d in [1, 2, 3]:
            r_euc = float(np.tanh(d / 2))
            theta = np.linspace(0, 2 * np.pi, 200)
            fig.add_trace(go.Scatter(
                x=r_euc * np.cos(theta),
                y=r_euc * np.sin(theta),
                mode="lines",
                line=dict(color=GRID_COLOR, width=1, dash="dash"),
                hoverinfo="skip",
                showlegend=False,
            ))

        # Determine point colours
        if show_soft and responsibilities is not None:
            colors = blend_colors(responsibilities)
        else:
            colors = [CLUSTER_COLORS[int(l) % len(CLUSTER_COLORS)] for l in labels]

        # Data points
        fig.add_trace(go.Scatter(
            x=points[:, 0],
            y=points[:, 1],
            mode="markers",
            marker=dict(size=6, color=colors, line=dict(color="white", width=0.4)),
            name="Points",
        ))

        # Cluster spread circles — show the empirical spread of each cluster
        # as a circle in disk coordinates centred on each centroid
        for j in range(K):
            members = points[labels == j]
            if len(members) < 2:
                continue
            # Mean Poincaré distance from centroid → convert to Euclidean radius
            dists = poincare_distance(members, centers[j])
            mean_dist = float(np.mean(dists))
            r_euc = float(np.tanh(mean_dist / 2))  # Poincaré distance → Euclidean radius
            theta = np.linspace(0, 2 * np.pi, 100)
            cx = centers[j, 0] + r_euc * np.cos(theta)
            cy = centers[j, 1] + r_euc * np.sin(theta)
            color = CLUSTER_COLORS[j % len(CLUSTER_COLORS)]
            fig.add_trace(go.Scatter(
                x=cx, y=cy, mode="lines",
                line=dict(color=color, width=2),
                opacity=0.6,
                showlegend=False, hoverinfo="skip",
            ))

        # Centers
        fig.add_trace(go.Scatter(
            x=centers[:, 0],
            y=centers[:, 1],
            mode="markers",
            marker=dict(
                size=16,
                color="black",
                symbol="star",
                line=dict(color="white", width=1.5),
            ),
            name="Centers",
        ))

        fig.update_layout(
            **PLOTLY_LAYOUT,
            title=title,
            height=650,
            xaxis=dict(
                range=[-1.15, 1.15],
                scaleanchor="y",
                scaleratio=1,
                gridcolor=GRID_COLOR,
                zeroline=False,
            ),
            yaxis=dict(
                range=[-1.15, 1.15],
                gridcolor=GRID_COLOR,
                zeroline=False,
            ),
        )
        return fig

    # ------------------------------------------------------------------
    # 7. Convergence chart
    # ------------------------------------------------------------------
    def build_convergence_plot(self, history, title=""):
        return convergence_chart(history, ylabel="Log-Likelihood", title=title)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _log_likelihood(points, means, sigmas, weights):
        """Compute total log-likelihood of the wrapped-normal mixture."""
        K = len(means)
        n = len(points)
        dist_mat = poincare_distance_matrix(points, means)
        log_density = np.empty((n, K))
        for k in range(K):
            s = float(np.clip(sigmas[k], 0.05, 3.0))
            log_Z = approximate_log_Z(round(s, 4))
            log_density[:, k] = np.log(weights[k] + 1e-30) - log_Z - dist_mat[:, k] ** 2 / (2 * s ** 2)
        return float(np.sum(_logsumexp(log_density, axis=1)))
