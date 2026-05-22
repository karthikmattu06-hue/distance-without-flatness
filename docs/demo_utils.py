"""
demo_utils.py — All distance functions, clustering algorithms, validation functions,
data loaders, and matplotlib theme helpers for "The Shape of Closeness" demo.

All clustering functions return (labels, centers, history).
GMM functions return hard assignments (argmax of responsibilities) as labels.
Pure numpy only (no sklearn).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import io
import os

from scipy.special import ive, i0e

# ──────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────

CLUSTER_COLORS = [
    '#C62828',  # deep red
    '#1565C0',  # deep blue
    '#2E7D32',  # deep green
    '#E65100',  # deep orange
    '#6A1B9A',  # deep purple
    '#AD1457',  # deep rose
    '#00695C',  # deep teal
    '#827717',  # olive
    '#283593',  # deep indigo
    '#4E342E',  # dark brown
]

R_MAJOR = 3.0   # torus major radius (3D embedding)
R_MINOR = 1.0   # torus minor radius (3D embedding)

CLIMATES = ['tropical', 'arid', 'temperate', 'continental', 'polar']

# ── Remote dataset URLs ───────────────────────────────────────
_SPHERE_URLS = {
    'earthquakes': 'https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_month.csv',
    'meteorites':  'https://raw.githubusercontent.com/cvalenzuela/Mappa/master/tutorials/basic/Meteorite_Landings.csv',
    'airports':    'https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat',
}
_FEAT_URLS = {
    'penguins': 'https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv',
    'iris':     'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv',
}

SPHERE_DATASET_LABELS = {
    'earthquakes': 'USGS Earthquakes (M2.5+, past 30 days)',
    'meteorites':  'NASA Meteorite Landings',
    'airports':    'Global Airports (OpenFlights)',
    'synthetic':   'Synthetic sphere clusters',
}
FEAT_DATASET_LABELS = {
    'penguins': 'Palmer Penguins (3 species)',
    'iris':     'Iris (3 species)',
    'synthetic': 'Synthetic Euclidean clusters',
}


def download_if_missing(url, local_path):
    """Download *url* to *local_path* if the file does not already exist."""
    if not os.path.exists(local_path):
        import urllib.request
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        urllib.request.urlretrieve(url, local_path)
TERRAINS = ['coastal', 'mountain', 'plains', 'forest']

# ──────────────────────────────────────────────────────────────
# THEME HELPERS  (Arctic — light)
# ──────────────────────────────────────────────────────────────

BG_FIGURE = '#F7F9FC'
BG_AXES   = '#FFFFFF'

def setup_dark_theme():
    """Apply Arctic light theme for all matplotlib figures."""
    plt.style.use('default')
    plt.rcParams.update({
        'figure.facecolor':  BG_FIGURE,
        'axes.facecolor':    BG_AXES,
        'axes.edgecolor':    '#D0D7E3',
        'axes.labelcolor':   '#1A2030',
        'xtick.color':       '#4A5568',
        'ytick.color':       '#4A5568',
        'text.color':        '#1A2030',
        'grid.color':        '#D0D7E3',
        'grid.alpha':        0.7,
        'legend.facecolor':  '#FFFFFF',
        'legend.edgecolor':  '#D0D7E3',
    })


def fig_to_png(fig):
    """Convert a matplotlib figure to PNG bytes for embedding in marimo."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()


# ──────────────────────────────────────────────────────────────
# DISTANCE FUNCTIONS
# ──────────────────────────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2, r=1.0):
    """
    Great-circle distance on a sphere of radius r using the Haversine formula.
    All angles in radians. Works element-wise on arrays.
    """
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def poincare_distance(u, v):
    """
    Hyperbolic distance in the Poincare disk model.
    d(u,v) = arccosh(1 + 2||u-v||^2 / ((1-||u||^2)(1-||v||^2)))
    u, v: arrays of shape (..., 2). Returns array of distances.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    diff_sq = np.sum((u - v)**2, axis=-1)
    u_sq = np.sum(u**2, axis=-1)
    v_sq = np.sum(v**2, axis=-1)
    denom = np.clip((1 - u_sq) * (1 - v_sq), 1e-15, None)
    arg = np.clip(1 + 2 * diff_sq / denom, 1.0, None)
    return np.arccosh(arg)


def torus_distance(u, v, period=2 * np.pi):
    """
    Flat torus distance: min(|d|, period-|d|) per dimension, then Euclidean norm.
    u, v: arrays of shape (2,) or (N, 2). Returns scalar or (N,) array.
    """
    diff = np.abs(u - v)
    wrapped = np.minimum(diff, period - diff)
    return np.sqrt(np.sum(wrapped**2, axis=-1))


def minkowski_distance(u, v, p=2):
    """
    Minkowski Lp distance between u and v.
    For p >= 100, uses Chebyshev (L-infinity). Works element-wise on arrays.
    """
    diff = np.abs(u - v)
    if p >= 100:
        return np.max(diff, axis=-1)
    return np.sum(diff**p, axis=-1) ** (1.0 / p)


def mahalanobis_distance(u, v, cov_inv):
    """
    Mahalanobis distance between u and v given inverse covariance cov_inv.
    u: (N, D) or (D,); v: (D,). Returns scalar or (N,) array.
    """
    diff = u - v
    if diff.ndim == 1:
        return np.sqrt(diff @ cov_inv @ diff)
    return np.sqrt(np.sum((diff @ cov_inv) * diff, axis=1))


# ──────────────────────────────────────────────────────────────
# COORDINATE TRANSFORMS
# ──────────────────────────────────────────────────────────────

def latlon_to_cartesian(lat, lon, r=1.0):
    """Convert lat/lon in radians to 3D Cartesian coordinates on sphere of radius r."""
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return x, y, z


def cartesian_to_latlon(x, y, z):
    """Convert 3D Cartesian coordinates to lat/lon in radians."""
    norm = np.sqrt(x**2 + y**2 + z**2 + 1e-16)
    lat = np.arcsin(np.clip(z / norm, -1, 1))
    lon = np.arctan2(y, x)
    return lat, lon


def torus_to_3d(theta, phi, R=R_MAJOR, r=R_MINOR):
    """
    Map flat torus coordinates (theta, phi) to 3D embedding.
    theta: angle around tube (small circle); phi: angle around hole (big circle).
    """
    x = (R + r * np.cos(theta)) * np.cos(phi)
    y = (R + r * np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)
    return x, y, z


def mobius_addition(a, b):
    """
    Mobius addition a + b in the Poincare disk (hyperbolic vector addition).
    a, b: arrays of shape (..., 2). Result is clipped to stay inside the unit disk.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a_dot_b = np.sum(a * b, axis=-1, keepdims=True)
    a_sq = np.sum(a**2, axis=-1, keepdims=True)
    b_sq = np.sum(b**2, axis=-1, keepdims=True)
    numerator = (1 + 2 * a_dot_b + b_sq) * a + (1 - a_sq) * b
    denominator = 1 + 2 * a_dot_b + a_sq * b_sq
    result = numerator / denominator
    norm = np.sqrt(np.sum(result**2, axis=-1, keepdims=True))
    result = np.where(norm > 0.999, result * 0.999 / norm, result)
    return result


def poincare_to_klein(p):
    """Map points from Poincare disk to Klein disk model."""
    p = np.asarray(p, dtype=np.float64)
    p_sq = np.sum(p**2, axis=-1, keepdims=True)
    return 2 * p / (1 + p_sq)


def klein_to_poincare(k):
    """Map points from Klein disk to Poincare disk model."""
    k = np.asarray(k, dtype=np.float64)
    k_sq = np.sum(k**2, axis=-1, keepdims=True)
    denom = 1 + np.sqrt(np.clip(1 - k_sq, 1e-15, None))
    return k / denom


# ──────────────────────────────────────────────────────────────
# CENTROID FUNCTIONS
# ──────────────────────────────────────────────────────────────

def spherical_mean(lats, lons):
    """
    Spherical mean of lat/lon points (radians): convert to Cartesian, average, project back.
    Returns (mean_lat, mean_lon) in radians.
    """
    x, y, z = latlon_to_cartesian(lats, lons)
    mx, my, mz = np.mean(x), np.mean(y), np.mean(z)
    norm = np.sqrt(mx**2 + my**2 + mz**2)
    if norm < 1e-10:
        return lats[0], lons[0]
    return cartesian_to_latlon(mx / norm, my / norm, mz / norm)


def circular_mean(angles, period=2 * np.pi):
    """
    Circular (angular) mean of angles, correctly handling wraparound.
    Returns mean in [0, period).
    """
    normalized = angles * (2 * np.pi / period)
    mean_sin = np.mean(np.sin(normalized))
    mean_cos = np.mean(np.cos(normalized))
    return (np.arctan2(mean_sin, mean_cos) * period / (2 * np.pi)) % period


def torus_centroid(points, period=2 * np.pi):
    """Torus centroid: apply circular_mean independently to each angular dimension."""
    return np.array([
        circular_mean(points[:, 0], period),
        circular_mean(points[:, 1], period),
    ])


def einstein_midpoint(points):
    """
    Hyperbolic centroid (Einstein midpoint) of points in the Poincare disk.
    Algorithm: Poincare -> Klein, Lorentz-weighted average, Klein -> Poincare.
    """
    points = np.asarray(points, dtype=np.float64)
    p_sq = np.sum(points**2, axis=-1)
    gamma = 1.0 / np.sqrt(np.clip(1 - p_sq, 1e-15, None))
    klein_pts = poincare_to_klein(points)
    weighted_sum = np.sum(gamma[:, np.newaxis] * klein_pts, axis=0)
    klein_mean = weighted_sum / np.sum(gamma)
    km_norm = np.linalg.norm(klein_mean)
    if km_norm >= 1.0:
        klein_mean = klein_mean * 0.999 / km_norm
    return klein_to_poincare(klein_mean)


def weighted_einstein_midpoint(points, weights):
    """
    Weighted Einstein midpoint (hyperbolic centroid) for GMM M-step.
    weights: (N,) responsibility array. Returns (2,) Poincare disk point.
    """
    points = np.asarray(points, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    p_sq = np.sum(points**2, axis=-1)
    gamma = 1.0 / np.sqrt(np.clip(1 - p_sq, 1e-15, None))
    klein_pts = poincare_to_klein(points)
    combined = gamma * weights
    total = np.sum(combined)
    if total < 1e-15:
        return np.array([0.0, 0.0])
    klein_mean = np.sum(combined[:, np.newaxis] * klein_pts, axis=0) / total
    km_norm = np.linalg.norm(klein_mean)
    if km_norm >= 1.0:
        klein_mean = klein_mean * 0.999 / km_norm
    return klein_to_poincare(klein_mean)


# ──────────────────────────────────────────────────────────────
# INTERNAL: DISTANCE MATRIX HELPERS
# ──────────────────────────────────────────────────────────────

def _haversine_matrix(lats, lons, c_lats, c_lons):
    """Return (N, K) haversine distance matrix."""
    D = np.zeros((len(lats), len(c_lats)))
    for j in range(len(c_lats)):
        D[:, j] = haversine(lats, lons, c_lats[j], c_lons[j])
    return D


def _poincare_matrix(points, centers):
    """Return (N, K) Poincare distance matrix."""
    D = np.zeros((len(points), len(centers)))
    for j in range(len(centers)):
        D[:, j] = poincare_distance(points, centers[j])
    return D


def _torus_matrix(points, centers, period=2 * np.pi):
    """Return (N, K) torus distance matrix."""
    D = np.zeros((len(points), len(centers)))
    for j in range(len(centers)):
        D[:, j] = torus_distance(points, centers[j], period)
    return D


def _minkowski_matrix(points, centers, p=2):
    """Return (N, K) Minkowski distance matrix."""
    D = np.zeros((len(points), len(centers)))
    for j in range(len(centers)):
        D[:, j] = minkowski_distance(points, centers[j], p)
    return D


def _mahal_matrix(points, centers, cov_inv):
    """Return (N, K) Mahalanobis distance matrix."""
    D = np.zeros((len(points), len(centers)))
    for j in range(len(centers)):
        D[:, j] = mahalanobis_distance(points, centers[j], cov_inv)
    return D


# ──────────────────────────────────────────────────────────────
# K-MEANS CLUSTERING  (all return: labels, centers, history)
# ──────────────────────────────────────────────────────────────

def spherical_kmeans(lats, lons, k=3, max_iter=100, tol=1e-6, seed=123):
    """
    K-means on the unit sphere using Haversine distance and spherical mean centroids.

    Parameters
    ----------
    lats, lons : 1-D arrays of point coordinates in radians
    k          : number of clusters
    max_iter   : maximum iterations
    tol        : convergence threshold on max centroid shift (haversine)
    seed       : random seed

    Returns
    -------
    labels  : (N,) cluster assignments
    centers : (k, 2) array of [lat, lon] centroids in radians
    history : list of total intra-cluster Haversine distance per iteration
    """
    rng = np.random.default_rng(seed)
    n = len(lats)
    idx = rng.choice(n, size=k, replace=False)
    c_lats = lats[idx].copy()
    c_lons = lons[idx].copy()
    history = []
    labels = np.zeros(n, dtype=int)

    for _ in range(max_iter):
        D = _haversine_matrix(lats, lons, c_lats, c_lons)
        labels = np.argmin(D, axis=1)
        history.append(float(np.sum(D[np.arange(n), labels])))

        new_lats = np.zeros(k)
        new_lons = np.zeros(k)
        for j in range(k):
            members = np.where(labels == j)[0]
            if len(members) == 0:
                i0 = rng.choice(n)
                new_lats[j], new_lons[j] = lats[i0], lons[i0]
            else:
                new_lats[j], new_lons[j] = spherical_mean(lats[members], lons[members])

        shift = np.max(haversine(c_lats, c_lons, new_lats, new_lons))
        c_lats, c_lons = new_lats, new_lons
        if shift < tol:
            break

    centers = np.stack([c_lats, c_lons], axis=1)
    return labels, centers, history


def hyperbolic_kmeans(points, k=3, max_iter=100, tol=1e-6, seed=123):
    """
    K-means in the Poincare disk using hyperbolic distance and Einstein midpoint centroids.

    Parameters
    ----------
    points   : (N, 2) array of points inside the unit disk
    k        : number of clusters
    max_iter : maximum iterations
    tol      : convergence threshold on max centroid shift (Poincare distance)
    seed     : random seed

    Returns
    -------
    labels    : (N,) cluster assignments
    centroids : (k, 2) Poincare disk centroids
    history   : list of total intra-cluster Poincare distance per iteration
    """
    rng = np.random.default_rng(seed)
    n = len(points)
    idx = rng.choice(n, size=k, replace=False)
    centroids = points[idx].copy()
    history = []
    labels = np.zeros(n, dtype=int)

    for _ in range(max_iter):
        D = _poincare_matrix(points, centroids)
        labels = np.argmin(D, axis=1)
        history.append(float(np.sum(D[np.arange(n), labels])))

        new_c = np.zeros_like(centroids)
        for j in range(k):
            members = points[labels == j]
            new_c[j] = einstein_midpoint(members) if len(members) > 0 else points[rng.choice(n)]

        shift = poincare_distance(centroids, new_c)
        centroids = new_c
        if np.max(shift) < tol:
            break

    return labels, centroids, history


def torus_kmeans(points, k=3, max_iter=100, tol=1e-6, period=2 * np.pi, seed=123):
    """
    K-means on the flat torus using toroidal distance and circular-mean centroids.

    Parameters
    ----------
    points   : (N, 2) array of angular coordinates in [0, period)
    k        : number of clusters
    max_iter : maximum iterations
    tol      : convergence threshold on max centroid shift (torus distance)
    period   : periodicity (default 2*pi)
    seed     : random seed

    Returns
    -------
    labels  : (N,) cluster assignments
    centers : (k, 2) torus centroids
    history : list of total intra-cluster torus distance per iteration
    """
    rng = np.random.default_rng(seed)
    n = len(points)
    idx = rng.choice(n, size=k, replace=False)
    centers = points[idx].copy()
    history = []
    labels = np.zeros(n, dtype=int)

    for _ in range(max_iter):
        D = _torus_matrix(points, centers, period)
        labels = np.argmin(D, axis=1)
        history.append(float(np.sum(D[np.arange(n), labels])))

        new_c = np.zeros_like(centers)
        for j in range(k):
            members = points[labels == j]
            new_c[j] = torus_centroid(members, period) if len(members) > 0 else points[rng.choice(n)]

        shift = np.max(torus_distance(centers, new_c, period))
        centers = new_c
        if shift < tol:
            break

    return labels, centers, history


def minkowski_kmeans(points, k=3, p=2, max_iter=100, tol=1e-6, seed=123):
    """
    K-means using Minkowski Lp distance.
    Centroid update: median (L1), mean (L2), midrange (L-infinity).

    Parameters
    ----------
    points   : (N, D) array
    k        : number of clusters
    p        : Minkowski exponent (p >= 100 is treated as Chebyshev)
    max_iter : maximum iterations
    tol      : convergence threshold on max centroid shift
    seed     : random seed

    Returns
    -------
    labels  : (N,) cluster assignments
    centers : (k, D) centroids
    history : list of total intra-cluster Lp distance per iteration
    """
    rng = np.random.default_rng(seed)
    n = len(points)
    idx = rng.choice(n, size=k, replace=False)
    centers = points[idx].copy()
    history = []
    labels = np.zeros(n, dtype=int)

    for _ in range(max_iter):
        D = _minkowski_matrix(points, centers, p)
        labels = np.argmin(D, axis=1)
        history.append(float(np.sum(D[np.arange(n), labels])))

        new_c = np.zeros_like(centers)
        for j in range(k):
            members = points[labels == j]
            if len(members) == 0:
                new_c[j] = points[rng.choice(n)]
            elif p <= 1.0 + 1e-6:
                new_c[j] = np.median(members, axis=0)
            elif p >= 100:
                new_c[j] = (members.max(axis=0) + members.min(axis=0)) / 2
            else:
                new_c[j] = members.mean(axis=0)

        shift = np.max(np.linalg.norm(centers - new_c, axis=1))
        centers = new_c
        if shift < tol:
            break

    return labels, centers, history


def mahalanobis_kmeans(points, k=3, max_iter=100, tol=1e-6, seed=123):
    """
    K-means with global Mahalanobis distance (covariance computed from full dataset).

    Parameters
    ----------
    points   : (N, D) array
    k        : number of clusters
    max_iter : maximum iterations
    tol      : convergence threshold on max centroid shift
    seed     : random seed

    Returns
    -------
    labels  : (N,) cluster assignments
    centers : (k, D) centroids
    history : list of total intra-cluster Mahalanobis distance per iteration
    """
    rng = np.random.default_rng(seed)
    n, d = points.shape
    cov = np.cov(points.T)
    cov_inv = np.linalg.inv(cov + 1e-6 * np.eye(d))

    idx = rng.choice(n, size=k, replace=False)
    centers = points[idx].copy()
    history = []
    labels = np.zeros(n, dtype=int)

    for _ in range(max_iter):
        D = _mahal_matrix(points, centers, cov_inv)
        labels = np.argmin(D, axis=1)
        history.append(float(np.sum(D[np.arange(n), labels])))

        new_c = np.zeros_like(centers)
        for j in range(k):
            members = points[labels == j]
            new_c[j] = members.mean(axis=0) if len(members) > 0 else points[rng.choice(n)]

        shift = np.max(np.linalg.norm(centers - new_c, axis=1))
        centers = new_c
        if shift < tol:
            break

    return labels, centers, history


def kmedoids_pam(D, k=3, max_iter=100, seed=123):
    """
    K-Medoids using PAM (Partitioning Around Medoids): greedy BUILD + iterative SWAP.
    Operates on a precomputed N x N distance matrix.

    Parameters
    ----------
    D        : (N, N) symmetric distance matrix
    k        : number of medoids
    max_iter : maximum SWAP iterations
    seed     : unused (kept for API consistency)

    Returns
    -------
    labels  : (N,) cluster assignments
    medoids : (k,) indices of medoids in D
    history : list of total assignment cost per SWAP iteration
    """
    n = D.shape[0]

    # BUILD phase: greedy medoid initialization
    medoids = [int(np.argmin(D.sum(axis=1)))]
    for _ in range(1, k):
        dist_nearest = D[:, medoids].min(axis=1)
        candidates = [i for i in range(n) if i not in medoids]
        best_gain, best_c = -np.inf, candidates[0]
        for c in candidates:
            gain = float(np.sum(np.maximum(dist_nearest - D[:, c], 0)))
            if gain > best_gain:
                best_gain, best_c = gain, c
        medoids.append(best_c)

    medoids = np.array(medoids)
    D_med = D[:, medoids]
    labels = np.argmin(D_med, axis=1)
    cost = float(np.sum(D_med[np.arange(n), labels]))
    history = [cost]

    # SWAP phase
    for _ in range(max_iter):
        best_swap, best_delta = None, 0.0
        non_medoids = [i for i in range(n) if i not in medoids]

        for m_idx in range(k):
            for h in non_medoids:
                new_med = medoids.copy()
                new_med[m_idx] = h
                new_D = D[:, new_med]
                new_l = np.argmin(new_D, axis=1)
                new_cost = float(np.sum(new_D[np.arange(n), new_l]))
                delta = new_cost - cost
                if delta < best_delta:
                    best_delta, best_swap = delta, (m_idx, h)

        if best_swap is None:
            break

        medoids[best_swap[0]] = best_swap[1]
        D_med = D[:, medoids]
        labels = np.argmin(D_med, axis=1)
        cost = float(np.sum(D_med[np.arange(n), labels]))
        history.append(cost)

    return labels, medoids, history


def gower_distance_matrix(data, w_geo=1.0, w_climate=1.0, w_terrain=1.0):
    """
    Compute N x N Gower distance matrix for mixed geographic + categorical data.

    Components:
      - Geographic: Haversine distance normalized to [0, 1] (max = pi on unit sphere)
      - Climate:    Simple matching (0 if same, 1 if different)
      - Terrain:    Simple matching (0 if same, 1 if different)

    Parameters
    ----------
    data : dict with keys 'lats', 'lons', 'climate', 'terrain' (all length N)
    w_geo, w_climate, w_terrain : relative weights for each component

    Returns
    -------
    D : (N, N) symmetric distance matrix in [0, 1]
    """
    n = len(data['lats'])
    lats, lons = data['lats'], data['lons']
    climate, terrain = data['climate'], data['terrain']
    total_w = w_geo + w_climate + w_terrain
    D = np.zeros((n, n))

    for i in range(n):
        d_geo = haversine(lats[i], lons[i], lats, lons) / np.pi
        d_climate = (climate[i] != climate).astype(float)
        d_terrain = (terrain[i] != terrain).astype(float)
        D[i, :] = (w_geo * d_geo + w_climate * d_climate + w_terrain * d_terrain) / total_w

    return D


def gower_kmedoids(data, k=3, w_geo=1.0, w_climate=1.0, w_terrain=1.0, max_iter=100, seed=123):
    """
    K-Medoids with Gower distance on mixed geographic + categorical data.

    Parameters
    ----------
    data : dict with keys 'lats', 'lons', 'climate', 'terrain'
    k    : number of medoids
    w_geo, w_climate, w_terrain : Gower component weights
    max_iter : maximum SWAP iterations
    seed     : random seed

    Returns
    -------
    labels  : (N,) cluster assignments
    medoids : (k,) medoid indices
    history : list of total cost per iteration
    """
    D = gower_distance_matrix(data, w_geo, w_climate, w_terrain)
    return kmedoids_pam(D, k=k, max_iter=max_iter, seed=seed)


# ──────────────────────────────────────────────────────────────
# GMM INTERNAL HELPERS
# ──────────────────────────────────────────────────────────────

def _log_sum_exp(log_vals, axis=None):
    """Numerically stable log-sum-exp."""
    max_val = np.max(log_vals, axis=axis, keepdims=True)
    result = max_val + np.log(np.sum(np.exp(log_vals - max_val), axis=axis, keepdims=True))
    if axis is not None:
        result = np.squeeze(result, axis=axis)
    return result


def _log_gaussian_pdf(x, mu, sigma):
    """
    Log N(x | mu, sigma) for each row of x.
    x: (N, d); mu: (d,); sigma: (d, d). Returns (N,) log-densities.
    """
    d = len(mu)
    diff = x - mu
    try:
        L = np.linalg.cholesky(sigma)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(sigma + 1e-6 * np.eye(d))
    z = np.linalg.solve(L, diff.T)
    mahal_sq = np.sum(z**2, axis=0)
    log_det = 2 * np.sum(np.log(np.diag(L)))
    return -0.5 * (d * np.log(2 * np.pi) + log_det + mahal_sq)


def _log_vmf_pdf(x, mu, kappa, p=3):
    """
    Log vMF density for each row of x (unit vectors on S^{p-1}).
    x: (N, p); mu: (p,) unit vector; kappa: scalar. Returns (N,) log-densities.
    """
    v = p / 2.0 - 1
    log_ive = np.log(np.clip(ive(v, kappa), 1e-300, None))
    log_bessel = log_ive + kappa
    log_C = v * np.log(np.clip(kappa, 1e-300, None)) - (p / 2.0) * np.log(2 * np.pi) - log_bessel
    return log_C + kappa * (x @ mu)


def _estimate_kappa_vmf(R_bar, p=3):
    """Estimate vMF kappa from mean resultant length (Banerjee et al. 2005)."""
    R_bar = np.clip(R_bar, 1e-10, 1 - 1e-10)
    return R_bar * (p - R_bar**2) / (1 - R_bar**2)


def _estimate_kappa_vm(R_bar):
    """Estimate von Mises kappa from mean resultant length (Mardia-Jupp)."""
    R_bar = np.clip(R_bar, 1e-10, 1 - 1e-10)
    return R_bar * (2 - R_bar**2) / (1 - R_bar**2)


def _log_von_mises_pdf(x, mu, kappa):
    """Log von Mises PDF using i0e (exponentially-scaled Bessel) for numerical stability."""
    return kappa * (np.cos(x - mu) - 1) - np.log(2 * np.pi) - np.log(i0e(kappa))


# ──────────────────────────────────────────────────────────────
# HYPERBOLIC GMM: normalizing constant cache
# ──────────────────────────────────────────────────────────────

def _compute_log_Z(sigma, grid_res=150):
    """
    Approximate log Z(sigma) for the wrapped-normal on the Poincare disk.
    Integrates exp(-d_H^2/(2*sigma^2)) against the hyperbolic area element.
    """
    x = np.linspace(-0.995, 0.995, grid_res)
    dx = x[1] - x[0]
    xx, yy = np.meshgrid(x, x)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    r_sq = np.sum(grid**2, axis=1)
    inside = r_sq < 0.99
    r = np.sqrt(r_sq[inside])
    d_H = 2 * np.arctanh(np.clip(r, 0, 0.9999))
    kernel = np.exp(-d_H**2 / (2 * sigma**2))
    area = 4.0 / (1 - r_sq[inside])**2
    Z = np.sum(kernel * area) * dx * dx
    return np.log(max(Z, 1e-300))


_HYPER_SIGMA_GRID = None
_HYPER_LOG_Z_VALS = None


def _get_log_Z_interp(sigma):
    """Get log Z(sigma) by linear interpolation from a precomputed grid (lazy init)."""
    global _HYPER_SIGMA_GRID, _HYPER_LOG_Z_VALS
    if _HYPER_SIGMA_GRID is None:
        _HYPER_SIGMA_GRID = np.linspace(0.05, 3.0, 60)
        _HYPER_LOG_Z_VALS = np.array([_compute_log_Z(s) for s in _HYPER_SIGMA_GRID])
    return float(np.interp(np.clip(sigma, 0.05, 3.0), _HYPER_SIGMA_GRID, _HYPER_LOG_Z_VALS))


# ──────────────────────────────────────────────────────────────
# GMM CLUSTERING  (all return: labels, centers, history)
# ──────────────────────────────────────────────────────────────

def euclidean_gmm(points, k=3, max_iter=200, tol=1e-6, seed=123):
    """
    EM for Gaussian Mixture Model in Euclidean space.

    Parameters
    ----------
    points   : (N, D) array
    k        : number of components
    max_iter : maximum EM iterations
    tol      : convergence threshold on log-likelihood change
    seed     : random seed

    Returns
    -------
    labels  : (N,) hard assignments (argmax of responsibilities)
    means   : (k, D) component means
    history : list of log-likelihood per iteration
    """
    rng = np.random.default_rng(seed)
    N, d = points.shape
    idx = rng.choice(N, size=k, replace=False)
    means = points[idx].copy()
    covs = [np.eye(d) for _ in range(k)]
    weights = np.ones(k) / k
    history = []
    reg = 1e-6
    resp = np.ones((N, k)) / k

    for _ in range(max_iter):
        # E-step
        log_resp = np.zeros((N, k))
        for j in range(k):
            log_resp[:, j] = np.log(weights[j] + 1e-30) + _log_gaussian_pdf(points, means[j], covs[j])
        log_norm = _log_sum_exp(log_resp, axis=1)
        resp = np.exp(log_resp - log_norm[:, np.newaxis])
        ll = float(np.sum(log_norm))
        history.append(ll)
        if len(history) > 1 and abs(history[-1] - history[-2]) < tol:
            break

        # M-step
        for j in range(k):
            r_j = resp[:, j]
            N_j = np.sum(r_j)
            if N_j < 1e-10:
                means[j] = points[rng.choice(N)]
                covs[j] = np.eye(d)
                weights[j] = 1.0 / k
                continue
            means[j] = np.sum(r_j[:, np.newaxis] * points, axis=0) / N_j
            diff = points - means[j]
            covs[j] = (diff.T @ (diff * r_j[:, np.newaxis])) / N_j + reg * np.eye(d)
            weights[j] = N_j / N

    labels = np.argmax(resp, axis=1)
    return labels, means, history


def spherical_gmm(points_3d, k=3, max_iter=200, tol=1e-6, seed=123):
    """
    EM for von Mises-Fisher mixture on the unit sphere S^2 (3D unit vectors).

    Parameters
    ----------
    points_3d : (N, 3) array of unit vectors
    k         : number of components
    max_iter  : maximum EM iterations
    tol       : convergence threshold
    seed      : random seed

    Returns
    -------
    labels  : (N,) hard assignments (argmax of responsibilities)
    means   : (k, 3) mean directions (unit vectors)
    history : list of log-likelihood per iteration
    """
    rng = np.random.default_rng(seed)
    N, p = points_3d.shape
    idx = rng.choice(N, size=k, replace=False)
    means = points_3d[idx].copy()
    means /= np.linalg.norm(means, axis=1, keepdims=True)
    kappas = np.full(k, 20.0)
    weights = np.ones(k) / k
    history = []
    resp = np.ones((N, k)) / k

    for _ in range(max_iter):
        # E-step
        log_resp = np.zeros((N, k))
        for j in range(k):
            log_resp[:, j] = np.log(weights[j] + 1e-30) + _log_vmf_pdf(points_3d, means[j], kappas[j], p)
        log_norm = _log_sum_exp(log_resp, axis=1)
        resp = np.exp(log_resp - log_norm[:, np.newaxis])
        ll = float(np.sum(log_norm))
        history.append(ll)
        if len(history) > 1 and abs(history[-1] - history[-2]) < tol:
            break

        # M-step
        for j in range(k):
            r_j = resp[:, j]
            N_j = np.sum(r_j)
            if N_j < 1e-10:
                means[j] = points_3d[rng.choice(N)]
                kappas[j] = 20.0
                weights[j] = 1.0 / k
                continue
            r_bar_vec = np.sum(r_j[:, np.newaxis] * points_3d, axis=0)
            r_norm = np.linalg.norm(r_bar_vec)
            means[j] = r_bar_vec / max(r_norm, 1e-10)
            kappas[j] = np.clip(_estimate_kappa_vmf(r_norm / N_j, p), 0.1, 1000)
            weights[j] = N_j / N

    labels = np.argmax(resp, axis=1)
    return labels, means, history


def torus_gmm(points, k=3, max_iter=200, tol=1e-8, seed=123):
    """
    EM for product-of-von-Mises mixture on the flat torus [0, 2*pi)^2.

    Parameters
    ----------
    points   : (N, 2) array of angles in [0, 2*pi)
    k        : number of components
    max_iter : maximum EM iterations
    tol      : convergence threshold
    seed     : random seed

    Returns
    -------
    labels  : (N,) hard assignments (argmax of responsibilities)
    mu      : (k, 2) mean directions in [0, 2*pi)
    history : list of log-likelihood per iteration
    """
    rng = np.random.RandomState(seed)
    N, D = points.shape
    idx = rng.choice(N, k, replace=False)
    mu = points[idx].copy()
    kappa = np.full((k, D), 5.0)
    pi_k = np.ones(k) / k
    history = []
    resp = np.ones((N, k)) / k

    for _ in range(max_iter):
        # E-step: log p_k(x_n) = sum_d log_vM(x_nd | mu_kd, kappa_kd)
        log_comp = np.zeros((N, k))
        for j in range(k):
            for d in range(D):
                log_comp[:, j] += _log_von_mises_pdf(points[:, d], mu[j, d], kappa[j, d])
        log_w = log_comp + np.log(pi_k)[np.newaxis, :]
        log_norm = _log_sum_exp(log_w, axis=1)
        resp = np.exp(log_w - log_norm[:, np.newaxis])
        ll = float(np.sum(log_norm))
        history.append(ll)
        if len(history) > 1 and abs(history[-1] - history[-2]) < tol:
            break

        # M-step
        N_k = np.maximum(resp.sum(axis=0), 1e-10)
        pi_k = N_k / N
        for j in range(k):
            for d in range(D):
                C = np.sum(resp[:, j] * np.cos(points[:, d]))
                S = np.sum(resp[:, j] * np.sin(points[:, d]))
                mu[j, d] = np.arctan2(S, C) % (2 * np.pi)
                R_bar = np.sqrt(C**2 + S**2) / N_k[j]
                kappa[j, d] = _estimate_kappa_vm(R_bar)

    labels = np.argmax(resp, axis=1)
    return labels, mu, history


def hyperbolic_gmm(points, k=3, max_iter=100, tol=1e-6, seed=123):
    """
    EM for wrapped-normal mixture on the Poincare disk.
    Uses a lazy-precomputed normalizing-constant table for efficiency.
    Warm-starts with a few iterations of hyperbolic K-means.

    Parameters
    ----------
    points   : (N, 2) array of points inside the unit disk
    k        : number of components
    max_iter : maximum EM iterations
    tol      : convergence threshold
    seed     : random seed

    Returns
    -------
    labels  : (N,) hard assignments (argmax of responsibilities)
    means   : (k, 2) component centers in the Poincare disk
    history : list of log-likelihood per iteration
    """
    rng = np.random.default_rng(seed)
    N = len(points)

    # Warm-start with hyperbolic K-means
    idx = rng.choice(N, size=k, replace=False)
    means = points[idx].copy()
    for _ in range(15):
        D = _poincare_matrix(points, means)
        init_labels = np.argmin(D, axis=1)
        new_means = means.copy()
        for j in range(k):
            m = points[init_labels == j]
            if len(m) > 0:
                new_means[j] = weighted_einstein_midpoint(m, np.ones(len(m)))
        if np.max(poincare_distance(means, new_means)) < 1e-6:
            break
        means = new_means

    # Initialize sigmas from K-means cluster spread
    D = _poincare_matrix(points, means)
    init_labels = np.argmin(D, axis=1)
    sigmas = np.array([
        np.clip(
            np.sqrt(np.mean(D[init_labels == j, j]**2)) if np.any(init_labels == j) else 0.5,
            0.05, 2.0,
        )
        for j in range(k)
    ])
    weights = np.ones(k) / k
    history = []
    resp = np.ones((N, k)) / k

    for _ in range(max_iter):
        # E-step
        dist_mat = _poincare_matrix(points, means)
        log_dens = np.zeros((N, k))
        for j in range(k):
            log_Z = _get_log_Z_interp(sigmas[j])
            log_dens[:, j] = (
                np.log(weights[j] + 1e-30)
                - log_Z
                - dist_mat[:, j]**2 / (2 * sigmas[j]**2)
            )

        log_sum = _log_sum_exp(log_dens, axis=1)
        resp = np.exp(log_dens - log_sum[:, np.newaxis])
        resp = np.clip(resp, 1e-15, None)
        resp /= resp.sum(axis=1, keepdims=True)
        ll = float(np.sum(log_sum))
        history.append(ll)
        if len(history) > 1 and abs(history[-1] - history[-2]) < tol:
            break

        # M-step
        N_k = resp.sum(axis=0)
        new_means = np.zeros_like(means)
        new_sigmas = np.zeros(k)
        for j in range(k):
            new_means[j] = weighted_einstein_midpoint(points, resp[:, j])
            dists = poincare_distance(points, new_means[j])
            new_sigmas[j] = np.clip(
                np.sqrt(np.sum(resp[:, j] * dists**2) / max(N_k[j], 1e-10)),
                0.05, 3.0,
            )
        means = new_means
        sigmas = new_sigmas
        weights = N_k / N

    labels = np.argmax(resp, axis=1)
    return labels, means, history


# ──────────────────────────────────────────────────────────────
# VALIDATION
# ──────────────────────────────────────────────────────────────

def compute_distance_matrix(points, dist_fn):
    """
    Build an N x N symmetric distance matrix using dist_fn(u, v) -> scalar.
    Points can be any array; dist_fn receives two rows.
    """
    n = len(points)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = float(dist_fn(points[i], points[j]))
            D[i, j] = d
            D[j, i] = d
    return D


def silhouette_from_matrix(D, labels):
    """
    Mean silhouette score from a precomputed N x N distance matrix.
    s(i) = (b(i) - a(i)) / max(a(i), b(i))
    a(i) = mean intra-cluster distance; b(i) = min mean inter-cluster distance.
    Returns mean silhouette in [-1, 1]. Returns 0 for single-cluster inputs.
    """
    n = len(labels)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0

    s = np.zeros(n)
    for i in range(n):
        same = (labels == labels[i])
        same[i] = False
        if not same.any():
            s[i] = 0.0
            continue
        a_i = float(np.mean(D[i, same]))

        b_i = np.inf
        for c in unique_labels:
            if c == labels[i]:
                continue
            other = labels == c
            if not other.any():
                continue
            b_i = min(b_i, float(np.mean(D[i, other])))

        denom = max(a_i, b_i)
        s[i] = (b_i - a_i) / denom if denom > 0 else 0.0

    return float(np.mean(s))


def pve_score(D, labels):
    """
    Proportion of Variance Explained from a precomputed N x N distance matrix.
    PVE = 1 - TWCSS_K / TWCSS_1
    where TWCSS uses squared distances: sum_{cluster c} sum_{i in c} d(i, centroid_c)^2.
    Since we work from distance matrices, TWCSS is approximated as
    sum_c (sum_{i,j in c} d_{ij}^2) / (2 * n_c).

    Returns PVE in [0, 1]. Higher is better.
    """
    D = np.asarray(D, dtype=float)
    n = len(labels)
    unique = np.unique(labels)

    twcss_k = 0.0
    for c in unique:
        mask = labels == c
        n_c = int(mask.sum())
        if n_c < 2:
            continue
        sub = D[np.ix_(mask, mask)]
        twcss_k += np.sum(sub**2) / (2 * n_c)

    twcss_1 = np.sum(D**2) / (2 * n)
    if twcss_1 < 1e-15:
        return 0.0
    return float(1.0 - twcss_k / twcss_1)


# ──────────────────────────────────────────────────────────────
# DATA GENERATION
# ──────────────────────────────────────────────────────────────

def generate_sphere_clusters(n_points=300, k=5, spread_deg=15, seed=42):
    """
    Generate synthetic clustered points on the unit sphere.

    Returns
    -------
    lats, lons   : 1-D arrays in radians
    true_labels  : (N,) integer labels
    """
    rng = np.random.default_rng(seed)
    spread = np.deg2rad(spread_deg)
    c_lats_deg = np.array([60, -40, 10, -60, 35, 80, -20, 50, -70, 25])
    c_lons_deg = np.array([30, -60, 150, 80, -130, 60, 200, -90, -30, 100])
    c_lats = np.deg2rad(c_lats_deg[:k])
    c_lons = np.deg2rad(c_lons_deg[:k])
    ppc = n_points // k
    all_lats, all_lons, all_labels = [], [], []
    for i in range(k):
        n = ppc if i < k - 1 else n_points - ppc * (k - 1)
        dlat = rng.normal(0, spread, n)
        dlon = rng.normal(0, spread / max(np.cos(c_lats[i]), 0.1), n)
        lats = np.clip(c_lats[i] + dlat, -np.pi / 2, np.pi / 2)
        lons = (c_lons[i] + dlon + np.pi) % (2 * np.pi) - np.pi
        all_lats.append(lats)
        all_lons.append(lons)
        all_labels.append(np.full(n, i))
    return np.concatenate(all_lats), np.concatenate(all_lons), np.concatenate(all_labels)


def generate_torus_clusters(n_points=300, k=5, spread=0.35, seed=42):
    """
    Generate synthetic clustered points on the flat torus [0, 2*pi)^2.
    Some centers placed near boundaries to highlight toroidal wraparound.

    Returns
    -------
    points      : (N, 2) array
    true_labels : (N,) integer labels
    """
    rng = np.random.default_rng(seed)
    configs = [
        (0.3, 0.2), (3.5, 1.5), (1.0, 4.5), (5.0, 5.8), (4.0, 3.2),
        (2.0, 2.0), (1.5, 5.5), (5.5, 1.0), (3.0, 4.0), (0.5, 3.0),
    ]
    centers = np.array(configs[:k])
    ppc = n_points // k
    all_pts, all_labels = [], []
    for i in range(k):
        n = ppc if i < k - 1 else n_points - ppc * (k - 1)
        pts = (centers[i] + rng.normal(0, spread, (n, 2))) % (2 * np.pi)
        all_pts.append(pts)
        all_labels.append(np.full(n, i))
    return np.vstack(all_pts), np.concatenate(all_labels)


def generate_hyperbolic_clusters(n_points=300, k=5, spread=0.08, seed=42):
    """
    Generate synthetic clustered points in the Poincare disk.
    Centers placed at varying depths to demonstrate hyperbolic geometry effects.

    Returns
    -------
    points      : (N, 2) array inside the unit disk
    true_labels : (N,) integer labels
    centers     : (k, 2) seed center coordinates
    """
    rng = np.random.default_rng(seed)
    configs = [
        (0.15, 0.0), (0.50, 2.0), (0.75, 4.2), (0.88, 1.2), (0.60, 5.0),
        (0.30, 3.5), (0.70, 0.5), (0.45, 5.8), (0.82, 3.0), (0.20, 1.8),
    ]
    centers = np.array([[r * np.cos(t), r * np.sin(t)] for r, t in configs[:k]])
    ppc = n_points // k
    all_pts, all_labels = [], []
    for i in range(k):
        n = ppc if i < k - 1 else n_points - ppc * (k - 1)
        pert = rng.normal(0, spread, (n, 2))
        norms = np.linalg.norm(pert, axis=1, keepdims=True)
        pert = np.where(norms > 0.4, pert * 0.4 / norms, pert)
        pts = mobius_addition(centers[i], pert)
        all_pts.append(pts)
        all_labels.append(np.full(n, i))
    return np.vstack(all_pts), np.concatenate(all_labels), centers


def generate_euclidean_clusters(n_points=300, k=4, seed=42):
    """
    Generate 2D Gaussian clusters with different covariance shapes for GMM demos.

    Returns
    -------
    points      : (N, 2) array
    true_labels : (N,) integer labels
    """
    rng = np.random.default_rng(seed)
    configs = [
        (np.array([0.0, 0.0]),   np.array([[1.0, 0.0],  [0.0, 1.0]])),
        (np.array([5.0, 5.0]),   np.array([[3.0, 2.2],  [2.2, 2.0]])),
        (np.array([-4.0, 4.0]),  np.array([[4.0, 0.0],  [0.0, 0.3]])),
        (np.array([3.0, -3.0]),  np.array([[0.3, 0.1],  [0.1, 0.3]])),
        (np.array([-3.0, -1.0]), np.array([[1.5, 0.8],  [0.8, 0.8]])),
    ][:k]
    ppc = n_points // k
    all_pts, all_labels = [], []
    for i, (mu, cov) in enumerate(configs):
        n = ppc if i < k - 1 else n_points - ppc * (k - 1)
        pts = rng.multivariate_normal(mu, cov, size=n)
        all_pts.append(pts)
        all_labels.append(np.full(n, i))
    pts = np.vstack(all_pts)
    lbls = np.concatenate(all_labels)
    perm = rng.permutation(len(pts))
    return pts[perm], lbls[perm]


def generate_mixed_data(n_points=300, n_clusters=5, seed=42):
    """
    Generate mixed geographic + categorical data for Gower / K-Medoids demos.
    Includes clusters that are geographically close but categorically different,
    and clusters that are geographically far but categorically similar.

    Returns
    -------
    dict with keys:
      'lats', 'lons'   : radians
      'climate'        : string array
      'terrain'        : string array
      'true_labels'    : integer array
      'cluster_defs'   : list of cluster definition tuples
    """
    rng = np.random.default_rng(seed)
    spread_rad = np.deg2rad(12)
    cluster_defs = [
        (20, 40,   'tropical',  'coastal',   0.85, 0.80),
        (25, 50,   'arid',      'plains',    0.85, 0.80),
        (-50, -70, 'polar',     'mountain',  0.90, 0.85),
        (55, 10,   'temperate', 'forest',    0.85, 0.80),
        (50, 140,  'temperate', 'forest',    0.85, 0.80),
    ][:n_clusters]
    ppc = n_points // n_clusters
    all_lats, all_lons = [], []
    all_climates, all_terrains, all_labels = [], [], []
    for i, (clat, clon, dom_c, dom_t, c_pur, t_pur) in enumerate(cluster_defs):
        n = ppc if i < n_clusters - 1 else n_points - ppc * (n_clusters - 1)
        clat_r, clon_r = np.deg2rad(clat), np.deg2rad(clon)
        dlat = rng.normal(0, spread_rad, n)
        dlon = rng.normal(0, spread_rad / max(np.cos(clat_r), 0.1), n)
        lats = np.clip(clat_r + dlat, -np.pi / 2, np.pi / 2)
        lons = (clon_r + dlon + np.pi) % (2 * np.pi) - np.pi
        climates = [dom_c if rng.random() < c_pur else rng.choice(CLIMATES) for _ in range(n)]
        terrains = [dom_t if rng.random() < t_pur else rng.choice(TERRAINS) for _ in range(n)]
        all_lats.append(lats)
        all_lons.append(lons)
        all_climates.extend(climates)
        all_terrains.extend(terrains)
        all_labels.append(np.full(n, i))
    return {
        'lats': np.concatenate(all_lats),
        'lons': np.concatenate(all_lons),
        'climate': np.array(all_climates),
        'terrain': np.array(all_terrains),
        'true_labels': np.concatenate(all_labels),
        'cluster_defs': cluster_defs,
    }


def generate_ramachandran_data(n=500, seed=42):
    """
    Generate synthetic Ramachandran-like torsion angle data on the torus [-pi, pi)^2.
    Four regions corresponding to known protein secondary structure zones.

    Returns
    -------
    dict with keys 'phi', 'psi' (radians), 'secondary_structure', 'true_labels'.
    """
    rng = np.random.default_rng(seed)
    # (phi_center_deg, psi_center_deg, fraction, spread_deg, name)
    regions = [
        (-60, -45, 0.35, 15, 'alpha-helix'),
        (-120, 130, 0.30, 20, 'beta-sheet'),
        (-65, 135, 0.20, 18, 'polyproline'),
        (60,   45, 0.15, 12, 'left-helix'),
    ]
    counts = [int(r[2] * n) for r in regions]
    counts[-1] = n - sum(counts[:-1])
    phi_list, psi_list, labels, ss = [], [], [], []
    for i, (phi_c, psi_c, _, spread, name) in enumerate(regions):
        s = np.deg2rad(spread)
        phi = np.clip(rng.normal(np.deg2rad(phi_c), s, counts[i]), -np.pi, np.pi)
        psi = np.clip(rng.normal(np.deg2rad(psi_c), s, counts[i]), -np.pi, np.pi)
        phi_list.append(phi)
        psi_list.append(psi)
        labels.append(np.full(counts[i], i))
        ss.extend([name] * counts[i])
    phi_arr = np.concatenate(phi_list)
    psi_arr = np.concatenate(psi_list)
    true_labels = np.concatenate(labels)
    perm = rng.permutation(n)
    return {
        'phi': phi_arr[perm],
        'psi': psi_arr[perm],
        'secondary_structure': np.array(ss)[perm],
        'true_labels': true_labels[perm],
    }


# ──────────────────────────────────────────────────────────────
# DATA LOADERS  (with synthetic fallbacks)
# ──────────────────────────────────────────────────────────────

def load_earthquakes(path=None):
    """
    Load USGS earthquake data (lat/lon in degrees, magnitude).
    Falls back to synthetic sphere cluster data if file is missing or unreadable.

    Returns
    -------
    dict with keys: 'lats' (radians), 'lons' (radians), 'magnitude' (array or None).
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), 'data', 'earthquakes.csv')
    try:
        import csv
        lats, lons, mags = [], [], []
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    lats.append(float(row['latitude']))
                    lons.append(float(row['longitude']))
                    mags.append(float(row.get('mag', row.get('magnitude', 0.0))))
                except (KeyError, ValueError):
                    continue
        if len(lats) < 10:
            raise ValueError("Too few rows")
        return {
            'lats': np.deg2rad(np.array(lats)),
            'lons': np.deg2rad(np.array(lons)),
            'magnitude': np.array(mags),
        }
    except Exception:
        lats, lons, _ = generate_sphere_clusters(n_points=400, k=5, seed=0)
        return {'lats': lats, 'lons': lons, 'magnitude': None}


def load_meteorites(path):
    """
    Load NASA meteorite landings CSV (reclat / reclong columns, degrees).
    Returns dict with 'lats', 'lons' (radians), 'magnitude' (mass in g or None).
    """
    import csv
    lats, lons, masses = [], [], []
    try:
        with open(path, newline='', encoding='latin-1') as f:
            reader = csv.DictReader(f)
            # column name may be 'reclong' or start with 'reclong'
            lon_col = next(
                (c for c in reader.fieldnames or [] if c.startswith('reclong')), None
            )
            for row in reader:
                try:
                    lat = float(row['reclat'])
                    lon = float(row[lon_col]) if lon_col else float(row['reclong'])
                    mass_str = row.get('mass (g)', '').strip()
                    masses.append(float(mass_str) if mass_str else 0.0)
                    lats.append(lat)
                    lons.append(lon)
                except (KeyError, ValueError, TypeError):
                    continue
        if len(lats) < 10:
            raise ValueError("Too few rows")
        return {
            'lats': np.deg2rad(np.array(lats)),
            'lons': np.deg2rad(np.array(lons)),
            'magnitude': np.array(masses),
        }
    except Exception:
        lats, lons, _ = generate_sphere_clusters(n_points=400, k=5, seed=1)
        return {'lats': lats, 'lons': lons, 'magnitude': None}


def load_airports(path):
    """
    Load OpenFlights airports.dat (no header, comma-delimited).
    Fields: id,name,city,country,iata,icao,lat,lon,alt,...
    Returns dict with 'lats', 'lons' (radians), 'magnitude' (None).
    """
    import csv
    lats, lons = [], []
    try:
        with open(path, newline='', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            for row in reader:
                try:
                    lat = float(row[6])
                    lon = float(row[7])
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        lats.append(lat)
                        lons.append(lon)
                except (IndexError, ValueError):
                    continue
        if len(lats) < 10:
            raise ValueError("Too few rows")
        return {
            'lats': np.deg2rad(np.array(lats)),
            'lons': np.deg2rad(np.array(lons)),
            'magnitude': None,
        }
    except Exception:
        lats, lons, _ = generate_sphere_clusters(n_points=400, k=5, seed=2)
        return {'lats': lats, 'lons': lons, 'magnitude': None}


def load_penguins(path=None):
    """
    Load Palmer Penguins data (bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g).
    Falls back to synthetic Euclidean cluster data if file is missing or unreadable.

    Returns
    -------
    dict with keys:
      'points'       : (N, 2) normalized (first 2 features, for 2D plotting)
      'points_full'  : (N, 4) all normalized features
      'species'      : array of species strings, or None
      'feature_names': list of feature name strings
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), 'data', 'penguins.csv')
    try:
        import csv
        features, species = [], []
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    bl = float(row['bill_length_mm'])
                    bd = float(row['bill_depth_mm'])
                    fl = float(row['flipper_length_mm'])
                    bm = float(row['body_mass_g'])
                    features.append([bl, bd, fl, bm])
                    species.append(row.get('species', ''))
                except (KeyError, ValueError):
                    continue
        if len(features) < 10:
            raise ValueError("Too few rows")
        pts = np.array(features, dtype=float)
        pts = (pts - pts.mean(axis=0)) / (pts.std(axis=0) + 1e-8)
        return {
            'points': pts[:, :2],
            'points_full': pts,
            'species': np.array(species),
            'feature_names': ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g'],
        }
    except Exception:
        pts, _ = generate_euclidean_clusters(n_points=333, k=3, seed=99)
        return {
            'points': pts,
            'points_full': pts,
            'species': None,
            'feature_names': ['feature_1', 'feature_2'],
        }


def load_iris(path):
    """
    Load the Iris dataset (sepal_length, sepal_width, petal_length, petal_width, species).
    Returns same dict shape as load_penguins: points, points_full, species, feature_names.
    """
    import csv
    features, species = [], []
    try:
        with open(path, newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    sl = float(row['sepal_length'])
                    sw = float(row['sepal_width'])
                    pl = float(row['petal_length'])
                    pw = float(row['petal_width'])
                    features.append([sl, sw, pl, pw])
                    species.append(row.get('species', ''))
                except (KeyError, ValueError):
                    continue
        if len(features) < 10:
            raise ValueError("Too few rows")
        pts = np.array(features, dtype=float)
        pts = (pts - pts.mean(axis=0)) / (pts.std(axis=0) + 1e-8)
        return {
            'points':       pts[:, :2],
            'points_full':  pts,
            'species':      np.array(species),
            'feature_names': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
        }
    except Exception:
        from sklearn.datasets import load_iris as _li  # type: ignore
        _d = _li()
        pts = (_d.data - _d.data.mean(0)) / (_d.data.std(0) + 1e-8)
        return {
            'points':       pts[:, :2],
            'points_full':  pts,
            'species':      np.array([_d.target_names[t] for t in _d.target]),
            'feature_names': list(_d.feature_names),
        }


def load_ramachandran(path=None):
    """
    Load Ramachandran torsion angle data (phi, psi in radians).
    Falls back to synthetic torus cluster data if file is missing or unreadable.

    Returns
    -------
    dict with keys: 'phi', 'psi' (radians), 'secondary_structure' (array or None).
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), 'data', 'ramachandran.csv')
    try:
        import csv
        phi_vals, psi_vals, ss_vals = [], [], []
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    phi_vals.append(float(row['phi']))
                    psi_vals.append(float(row['psi']))
                    ss_vals.append(row.get('secondary_structure', ''))
                except (KeyError, ValueError):
                    continue
        if len(phi_vals) < 10:
            raise ValueError("Too few rows")
        return {
            'phi': np.array(phi_vals),
            'psi': np.array(psi_vals),
            'secondary_structure': np.array(ss_vals),
        }
    except Exception:
        data = generate_ramachandran_data(n=500)
        return {
            'phi': data['phi'],
            'psi': data['psi'],
            'secondary_structure': data['secondary_structure'],
        }


def save_ramachandran_csv(path=None):
    """
    Generate and save ramachandran.csv to disk. Returns the path written.
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), 'data', 'ramachandran.csv')
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = generate_ramachandran_data(n=500)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['phi', 'psi', 'secondary_structure'])
        writer.writeheader()
        for phi, psi, ss in zip(data['phi'], data['psi'], data['secondary_structure']):
            writer.writerow({'phi': phi, 'psi': psi, 'secondary_structure': ss})
    return path


# ──────────────────────────────────────────────────────────────
# PHASE-2 HELPERS  (demo_marimo.py API)
# ──────────────────────────────────────────────────────────────

def apply_dark_theme():
    """Apply Arctic light theme (alias for setup_dark_theme)."""
    setup_dark_theme()


def load_sphere_data(data_dir, dataset='earthquakes'):
    """
    Load sphere data for the given *dataset* key, auto-downloading if needed.

    dataset : 'earthquakes' | 'meteorites' | 'airports' | 'synthetic'
    Returns (data_dict, label_str, is_synthetic_bool).
    data_dict keys: 'lats', 'lons' (radians), 'magnitude' (array or None).
    """
    if dataset == 'meteorites':
        path = os.path.join(data_dir, 'meteorites.csv')
        download_if_missing(_SPHERE_URLS['meteorites'], path)
        data = load_meteorites(path)
        is_synthetic = data['magnitude'] is None
    elif dataset == 'airports':
        path = os.path.join(data_dir, 'airports.dat')
        download_if_missing(_SPHERE_URLS['airports'], path)
        data = load_airports(path)
        is_synthetic = False
    elif dataset == 'synthetic':
        lats, lons, _ = generate_sphere_clusters(n_points=400, k=5, seed=0)
        data = {'lats': lats, 'lons': lons, 'magnitude': None}
        is_synthetic = True
    else:  # 'earthquakes' (default)
        path = os.path.join(data_dir, 'earthquakes.csv')
        download_if_missing(_SPHERE_URLS['earthquakes'], path)
        data = load_earthquakes(path)
        is_synthetic = data['magnitude'] is None

    label = SPHERE_DATASET_LABELS.get(dataset, dataset)
    if is_synthetic:
        label = SPHERE_DATASET_LABELS['synthetic']
    return data, label, is_synthetic


def compute_haversine_matrix(lats, lons):
    """Compute NxN haversine distance matrix from lat/lon arrays (radians)."""
    n = len(lats)
    D = np.zeros((n, n))
    for i in range(n):
        D[i, :] = haversine(lats[i], lons[i], lats, lons)
    return D


def silhouette_scores_per_point(D, labels):
    """
    Return per-point silhouette scores (N,) from a precomputed NxN distance matrix.
    s(i) = (b(i) - a(i)) / max(a(i), b(i))
    """
    n = len(labels)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return np.zeros(n)
    s = np.zeros(n)
    for i in range(n):
        same = labels == labels[i]
        same[i] = False
        if not same.any():
            continue
        a_i = float(np.mean(D[i, same]))
        b_i = np.inf
        for c in unique_labels:
            if c == labels[i]:
                continue
            other = labels == c
            if not other.any():
                continue
            b_i = min(b_i, float(np.mean(D[i, other])))
        denom = max(a_i, b_i)
        s[i] = (b_i - a_i) / denom if denom > 0 else 0.0
    return s


def sweep_k(run_fn, D, k_range):
    """
    Sweep k values for validation.

    Parameters
    ----------
    run_fn  : callable(k, D) → (labels, centers, history)
              Called once per k.  May ignore D (e.g. K-Means on explicit coords).
    D       : (N, N) precomputed distance matrix used for PVE & silhouette.
    k_range : iterable of integer k values.

    Returns
    -------
    dict with keys:
      k_values          : list of k
      pve               : list of PVE scores
      silhouette_mean   : list of mean silhouette scores
      silhouette_per_point : list of (N,) arrays
      labels_per_k      : list of (N,) label arrays
      centers_per_k     : list of raw center arrays (format depends on algorithm)
    """
    k_values = list(k_range)
    pve_list, sil_mean_list, sil_pp_list, labels_list, centers_list = [], [], [], [], []
    for k in k_values:
        labels, centers, _ = run_fn(k, D)
        pve_list.append(pve_score(D, labels))
        scores = silhouette_scores_per_point(D, labels)
        sil_mean_list.append(float(np.mean(scores)))
        sil_pp_list.append(scores)
        labels_list.append(labels)
        centers_list.append(centers)
    return {
        'k_values':             k_values,
        'pve':                  pve_list,
        'silhouette_mean':      sil_mean_list,
        'silhouette_per_point': sil_pp_list,
        'labels_per_k':         labels_list,
        'centers_per_k':        centers_list,
    }


# ──────────────────────────────────────────────────────────────
# LLOYD'S ALGORITHM STEP-BY-STEP  (for animation)
# ──────────────────────────────────────────────────────────────

def lloyd_steps(points, k, geometry='euclidean', seed=42, max_iter=15):
    """
    Run Lloyd's K-means step by step, returning every intermediate state.

    Parameters
    ----------
    points   : ndarray (n, 2)
               Euclidean: arbitrary 2-D feature vectors.
               Spherical: (lat, lon) in radians.
               Torus    : (phi, psi) in radians.
               Hyperbolic: 2-D Poincaré disk coords.
    k        : int   number of clusters
    geometry : 'euclidean' | 'spherical' | 'torus' | 'hyperbolic'
    seed     : int   controls random initialisation
    max_iter : int   maximum Lloyd iterations

    Returns
    -------
    list of dicts, each with:
        iteration : int   (0 = initialisation)
        phase     : 'init' | 'assign' | 'update'
        centers   : ndarray (k, 2)
        labels    : ndarray (n,) int  (-1 before first assignment)
        wcss      : float  (inf before first assignment)
    """
    rng = np.random.default_rng(seed)
    n = len(points)

    # ── Distance matrix ──────────────────────────────────────────────────────────
    if geometry == 'euclidean':
        def _dmat(pts, ctrs):
            return np.sqrt(((pts[:, None] - ctrs[None])**2).sum(-1))

    elif geometry == 'spherical':
        def _dmat(pts, ctrs):
            D = np.zeros((len(pts), len(ctrs)))
            for j, c in enumerate(ctrs):
                D[:, j] = haversine(pts[:, 0], pts[:, 1], c[0], c[1])
            return D

    elif geometry == 'torus':
        _L = 2 * np.pi
        def _dmat(pts, ctrs):
            D = np.zeros((len(pts), len(ctrs)))
            for j, c in enumerate(ctrs):
                diff = np.abs(pts - c)
                wrapped = np.minimum(diff, _L - diff)
                D[:, j] = np.sqrt((wrapped**2).sum(-1))
            return D

    elif geometry == 'hyperbolic':
        def _dmat(pts, ctrs):
            D = np.zeros((len(pts), len(ctrs)))
            for j, c in enumerate(ctrs):
                diff_sq = np.sum((pts - c)**2, axis=1)
                denom = np.maximum(
                    (1 - np.sum(pts**2, axis=1)) * (1 - np.sum(c**2)), 1e-12
                )
                D[:, j] = np.arccosh(np.maximum(1.0 + 2 * diff_sq / denom, 1.0))
            return D

    # ── Centroid update ───────────────────────────────────────────────────────────
    if geometry == 'euclidean':
        def _update(pts, lbls):
            return np.array([
                pts[lbls == c].mean(0) if (lbls == c).any()
                else pts[rng.integers(n)]
                for c in range(k)
            ])

    elif geometry == 'spherical':
        def _update(pts, lbls):
            xs, ys, zs = latlon_to_cartesian(pts[:, 0], pts[:, 1])
            centers = []
            for c in range(k):
                m = lbls == c
                if m.any():
                    mx, my, mz = xs[m].mean(), ys[m].mean(), zs[m].mean()
                    nr = np.sqrt(mx**2 + my**2 + mz**2)
                    if nr > 1e-8:
                        mx, my, mz = mx/nr, my/nr, mz/nr
                    centers.append(list(cartesian_to_latlon(mx, my, mz)))
                else:
                    centers.append(pts[rng.integers(n)].tolist())
            return np.array(centers)

    elif geometry == 'torus':
        def _update(pts, lbls):
            centers = []
            for c in range(k):
                m = lbls == c
                if m.any():
                    a = pts[m]
                    centers.append(np.arctan2(np.sin(a).mean(0), np.cos(a).mean(0)))
                else:
                    centers.append(pts[rng.integers(n)].copy())
            return np.array(centers)

    elif geometry == 'hyperbolic':
        def _update(pts, lbls):
            centers = []
            for c in range(k):
                m = lbls == c
                if not m.any():
                    centers.append(pts[rng.integers(n)] * 0.8)
                    continue
                sub = pts[m]
                norms_sq = np.clip(np.sum(sub**2, axis=1), 0, 1 - 1e-8)
                gamma = 1.0 / np.sqrt(1 - norms_sq)
                midpoint = (sub * gamma[:, None]).sum(0) / gamma.sum()
                r = np.linalg.norm(midpoint)
                if r >= 1:
                    midpoint = midpoint * 0.99 / r
                centers.append(midpoint)
            return np.array(centers)

    # ── Run ───────────────────────────────────────────────────────────────────────
    centers = points[rng.choice(n, k, replace=False)].copy().astype(float)

    steps = [{'iteration': 0, 'phase': 'init',
               'centers': centers.copy(),
               'labels': np.full(n, -1, dtype=int),
               'wcss': np.inf}]

    for it in range(1, max_iter + 1):
        D_mat  = _dmat(points, centers)
        labels = np.argmin(D_mat, axis=1)
        wcss   = float(np.sum(np.min(D_mat, axis=1)**2))

        steps.append({'iteration': it, 'phase': 'assign',
                      'centers': centers.copy(), 'labels': labels.copy(), 'wcss': wcss})

        new_centers = _update(points, labels)

        steps.append({'iteration': it, 'phase': 'update',
                      'centers': new_centers.copy(), 'labels': labels.copy(), 'wcss': wcss})

        if np.allclose(centers, new_centers, atol=1e-5):
            break
        centers = new_centers

    return steps


# ──────────────────────────────────────────────────────────────
# PLOTTING HELPERS  (used by demo_marimo.py)
# ──────────────────────────────────────────────────────────────

def plot_pve_curve(ax, k_values, pve_values, current_k):
    """Plot PVE curve with a vertical marker at current_k."""
    ax.plot(k_values, pve_values, 'o-', color='#3498db', lw=2.5,
            markersize=9, zorder=3, label='PVE')
    ax.axvline(current_k, color='#e74c3c', lw=2, ls='--', alpha=0.85,
               label=f'K = {current_k}')
    # Highlight the elbow: point after the biggest drop in marginal gain
    # (argmin of second differences — where diminishing returns kick in hardest)
    if len(pve_values) >= 3:
        _d1 = np.diff(pve_values)          # marginal gain per step
        _d2 = np.diff(_d1)                 # change in marginal gain
        elbow_idx = int(np.argmin(_d2)) + 1  # step AFTER the sharpest drop
        ax.scatter([k_values[elbow_idx]], [pve_values[elbow_idx]],
                   s=180, color='#f39c12', zorder=5,
                   label=f'Elbow ≈ K={k_values[elbow_idx]}')
    ax.set_xlabel('Number of Clusters  K', fontsize=14)
    ax.set_ylabel('PVE', fontsize=14)
    ax.set_title('Proportion of Variance Explained', fontsize=16, fontweight='bold')
    ax.set_xticks(k_values)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.35)
    ax.set_ylim(0, 1.05)


def plot_silhouette_curve(ax, k_values, sil_means, current_k):
    """Plot mean silhouette curve with vertical markers at current_k and best_k."""
    ax.plot(k_values, sil_means, 's-', color='#2ecc71', lw=2.5,
            markersize=9, zorder=3, label='Mean silhouette')
    ax.axvline(current_k, color='#e74c3c', lw=2, ls='--', alpha=0.85,
               label=f'K = {current_k}')
    best_k = k_values[int(np.argmax(sil_means))]
    if best_k != current_k:
        ax.axvline(best_k, color='#f39c12', lw=1.5, ls=':', alpha=0.9,
                   label=f'Best K = {best_k}')
    ax.set_xlabel('Number of Clusters  K', fontsize=14)
    ax.set_ylabel('Mean Silhouette', fontsize=14)
    ax.set_title('Mean Silhouette Score', fontsize=16, fontweight='bold')
    ax.set_xticks(k_values)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.35)
    ax.set_ylim(-0.2, 1.0)


def plot_sphere_map(ax, lats_rad, lons_rad, labels, c_lats_rad, c_lons_rad,
                    title, metric='haversine', center_marker='*'):
    """
    Equirectangular projection map with Voronoi territory shading.

    Parameters
    ----------
    ax            : matplotlib Axes
    lats_rad      : (N,) latitudes in radians
    lons_rad      : (N,) longitudes in radians
    labels        : (N,) integer cluster assignments
    c_lats_rad    : (k,) cluster-centre latitudes in radians
    c_lons_rad    : (k,) cluster-centre longitudes in radians
    title         : axes title
    metric        : 'haversine' or 'euclidean'
    center_marker : matplotlib marker string for centroids
    """
    import matplotlib.colors as _mc

    k = len(c_lats_rad)
    colors = [CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(k)]

    # ── 1-degree Voronoi background ──────────────────────────────────────────
    lon_deg = np.linspace(-180, 180, 360)
    lat_deg = np.linspace(-90, 90, 180)
    lg, latg = np.meshgrid(lon_deg, lat_deg)
    lr   = np.deg2rad(lg.ravel())
    latr = np.deg2rad(latg.ravel())

    D_bg = np.zeros((len(lr), k))
    for j in range(k):
        if metric == 'haversine':
            D_bg[:, j] = haversine(latr, lr, c_lats_rad[j], c_lons_rad[j])
        else:
            D_bg[:, j] = np.sqrt((latr - c_lats_rad[j])**2 + (lr - c_lons_rad[j])**2)

    voronoi = np.argmin(D_bg, axis=1).reshape(latg.shape)
    rgb = np.array([_mc.to_rgb(c) for c in colors])
    img = rgb[voronoi]
    ax.imshow(img, extent=[-180, 180, -90, 90], aspect='auto',
              alpha=0.15, origin='lower')

    # ── Data points ───────────────────────────────────────────────────────────
    lons_d = np.rad2deg(lons_rad)
    lats_d = np.rad2deg(lats_rad)
    for c in range(k):
        mask = labels == c
        ax.scatter(lons_d[mask], lats_d[mask],
                   c=colors[c], s=18, alpha=0.85, linewidths=0, zorder=3)

    # ── Centroids ─────────────────────────────────────────────────────────────
    ax.scatter(np.rad2deg(c_lons_rad), np.rad2deg(c_lats_rad),
               c='white', s=260, marker=center_marker, zorder=5,
               edgecolors='#111111', linewidths=0.8)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel('Longitude', fontsize=14)
    ax.set_ylabel('Latitude', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=8)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3)


# ──────────────────────────────────────────────────────────────
# PHASE-3 HELPERS  (Torus + Hyperbolic tabs)
# ──────────────────────────────────────────────────────────────

def load_torus_data(data_dir, seed=42):
    """
    Load Ramachandran torsion angle data for torus clustering.
    Converts phi/psi from [-pi, pi] to [0, 2*pi).
    *seed* is used only when falling back to purely synthetic data.

    Returns (data_dict, label_str, is_simulated).
    is_simulated=True  → info callout (simulated regions)
    is_simulated=None  → warning callout (pure synthetic fallback)
    data_dict keys: 'points' (N,2) in [0,2pi), 'phi', 'psi', 'secondary_structure'.
    """
    path = os.path.join(data_dir, 'ramachandran.csv')
    raw = load_ramachandran(path)
    phi = raw['phi'] % (2 * np.pi)
    psi = raw['psi'] % (2 * np.pi)
    points = np.stack([phi, psi], axis=1)
    ss = raw.get('secondary_structure')
    has_ss = ss is not None and len(ss) > 0 and str(ss[0]) not in ('', 'None')
    if has_ss:
        label = 'Ramachandran torsion angles (simulated secondary structure regions)'
        is_simulated = True
    else:
        # Pure synthetic fallback — regenerate with given seed
        _d = generate_ramachandran_data(n=500, seed=seed)
        phi = _d['phi'] % (2 * np.pi)
        psi = _d['psi'] % (2 * np.pi)
        points = np.stack([phi, psi], axis=1)
        ss = _d['secondary_structure']
        label = f'Synthetic Ramachandran angles (seed={seed})'
        is_simulated = None
    return (
        {'points': points, 'phi': phi, 'psi': psi, 'secondary_structure': ss},
        label,
        is_simulated,
    )


def load_hyperbolic_data(data_dir, seed=42):
    """
    Load or generate animal taxonomy Poincaré disk embedding.
    *seed* controls the synthetic fallback generator.
    Returns (data_dict, label_str, is_synthetic_bool).
    data_dict keys: 'points' (N,2) inside unit disk, 'labels' (N,).
    """
    path = os.path.join(data_dir, 'taxonomy_poincare.csv')
    try:
        import csv as _csv
        pts, lbls = [], []
        with open(path, newline='') as f:
            reader = _csv.DictReader(f)
            for row in reader:
                pts.append([float(row['x']), float(row['y'])])
                lbls.append(int(row.get('label', 0)))
        if len(pts) < 10:
            raise ValueError("Too few rows")
        pts_arr = np.array(pts)
        norms = np.linalg.norm(pts_arr, axis=1, keepdims=True)
        pts_arr = np.where(norms >= 0.99, pts_arr * 0.97 / norms, pts_arr)
        return (
            {'points': pts_arr, 'labels': np.array(lbls)},
            'Animal taxonomy (Poincaré embedding)',
            False,
        )
    except Exception:
        pts, lbls, _ = generate_hyperbolic_clusters(n_points=300, k=5, seed=seed)
        return (
            {'points': pts, 'labels': lbls},
            f'Simulated hierarchical embedding (seed={seed})',
            True,
        )


def compute_torus_matrix(points, period=2 * np.pi):
    """Compute full NxN torus distance matrix."""
    n = len(points)
    D = np.zeros((n, n))
    for i in range(n):
        D[i, :] = torus_distance(points, points[i], period)
    return D


def compute_poincare_matrix(points):
    """Compute full NxN Poincaré distance matrix."""
    n = len(points)
    D = np.zeros((n, n))
    for i in range(n):
        D[i, :] = poincare_distance(points, points[i])
    return D


def plot_torus_map(ax, pts_rad, labels, centers_rad, title,
                   metric='toroidal', period=2 * np.pi):
    """
    Flat torus scatter with Voronoi shading, displayed in degrees [0, 360).

    pts_rad     : (N, 2) angular coordinates in [0, period)
    centers_rad : (k, 2) cluster centres in [0, period)
    metric      : 'toroidal' or 'euclidean'
    """
    import matplotlib.colors as _mc

    k = len(centers_rad)
    colors = [CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(k)]
    to_deg = 360.0 / period

    # Voronoi background
    res = 200
    gx = np.linspace(0, period, res)
    gy = np.linspace(0, period, res)
    gxx, gyy = np.meshgrid(gx, gy)
    grid = np.stack([gxx.ravel(), gyy.ravel()], axis=1)

    D_bg = np.zeros((len(grid), k))
    for j in range(k):
        if metric == 'toroidal':
            D_bg[:, j] = torus_distance(grid, centers_rad[j], period)
        else:
            D_bg[:, j] = np.linalg.norm(grid - centers_rad[j], axis=1)
    voronoi = np.argmin(D_bg, axis=1).reshape(res, res)
    rgb = np.array([_mc.to_rgb(c) for c in colors])
    img = rgb[voronoi]
    ax.imshow(img, extent=[0, 360, 0, 360], aspect='auto', alpha=0.15, origin='lower')

    # Data points
    pts_deg = pts_rad * to_deg
    for c in range(k):
        mask = labels == c
        ax.scatter(pts_deg[mask, 0], pts_deg[mask, 1],
                   c=colors[c], s=12, alpha=0.80, linewidths=0, zorder=3)

    # Centroids
    c_deg = centers_rad * to_deg
    ax.scatter(c_deg[:, 0], c_deg[:, 1],
               c='white', s=220, marker='*', zorder=5,
               edgecolors='#111111', linewidths=0.8)

    # Wraparound boundaries (orange dashed at 0° and 360° on both axes)
    for v in [0, 360]:
        ax.axvline(v, color='#ff8c00', lw=1.5, ls='--', alpha=0.7, zorder=4)
        ax.axhline(v, color='#ff8c00', lw=1.5, ls='--', alpha=0.7, zorder=4)

    ax.set_xlim(0, 360)
    ax.set_ylim(0, 360)
    ax.set_xlabel('φ (degrees)', fontsize=14)
    ax.set_ylabel('ψ (degrees)', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=8)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3)


def plot_poincare_disk(ax, pts, labels, centers, title, metric='hyperbolic'):
    """
    Poincaré disk with Voronoi shading, depth rings, and unit-circle boundary.

    pts     : (N, 2) points inside the unit disk
    centers : (k, 2) cluster centres inside the unit disk
    metric  : 'hyperbolic' or 'euclidean'
    """
    import matplotlib.colors as _mc
    import matplotlib.patches as _mp

    k = len(centers)
    colors = [CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(k)]

    # Voronoi background (masked outside unit disk)
    res = 250
    lin = np.linspace(-1, 1, res)
    gxx, gyy = np.meshgrid(lin, lin)
    grid = np.stack([gxx.ravel(), gyy.ravel()], axis=1)
    inside = np.sum(grid**2, axis=1) < 0.9999

    bg_rgb = np.array(_mc.to_rgb('#1a1a2e'))
    voronoi_img = np.tile(bg_rgb, (res * res, 1))
    if np.any(inside) and k > 0:
        grid_in = grid[inside]
        D_bg = np.zeros((len(grid_in), k))
        for j in range(k):
            if metric == 'hyperbolic':
                D_bg[:, j] = poincare_distance(grid_in, centers[j])
            else:
                D_bg[:, j] = np.linalg.norm(grid_in - centers[j], axis=1)
        rgb = np.array([_mc.to_rgb(c) for c in colors])
        voronoi_img[inside] = rgb[np.argmin(D_bg, axis=1)]

    voronoi_img = voronoi_img.reshape(res, res, 3)
    ax.imshow(voronoi_img, extent=[-1, 1, -1, 1], aspect='equal',
              alpha=0.15, origin='lower', interpolation='bilinear')

    # Depth circles with hyperbolic distance annotations
    for r in [0.3, 0.6, 0.8, 0.9, 0.95]:
        circ = _mp.Circle((0, 0), r, fill=False, edgecolor='#555577',
                          lw=0.9, ls=':', zorder=2)
        ax.add_patch(circ)
        d_h = 2 * np.arctanh(r)
        ax.text(r * 0.71, r * 0.71, f'd≈{d_h:.1f}',
                fontsize=7, color='#7777aa', ha='left', va='bottom', zorder=3)

    # Unit circle boundary (thick)
    boundary = _mp.Circle((0, 0), 1.0, fill=False,
                           edgecolor='#bbbbcc', lw=2.5, zorder=4)
    ax.add_patch(boundary)

    # Data points
    for c in range(k):
        mask = labels == c
        ax.scatter(pts[mask, 0], pts[mask, 1],
                   c=colors[c], s=18, alpha=0.85, linewidths=0, zorder=5)

    # Centroids
    ax.scatter(centers[:, 0], centers[:, 1],
               c='white', s=220, marker='*', zorder=6,
               edgecolors='#111111', linewidths=0.8)

    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.set_aspect('equal')
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('y', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=8)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.2)


# ──────────────────────────────────────────────────────────────
# PHASE-4 HELPERS  (Feature Space + Mixed Type tabs)
# ──────────────────────────────────────────────────────────────

def compute_minkowski_matrix_full(points, p):
    """Compute full NxN Minkowski distance matrix."""
    n = len(points)
    D = np.zeros((n, n))
    for i in range(n):
        D[i, :] = minkowski_distance(points, points[i], p)
    return D


def load_feature_data(data_dir, dataset='penguins'):
    """
    Load feature-space data, standardize, run PCA to get 2D projection.

    dataset : 'penguins' | 'iris' | 'synthetic'
    Returns (data_dict, label_str, is_synthetic_bool).
    data_dict keys: 'points', 'points_2d', 'species', 'feature_names'.
    """
    if dataset == 'iris':
        path = os.path.join(data_dir, 'iris.csv')
        download_if_missing(_FEAT_URLS['iris'], path)
        raw = load_iris(path)
    elif dataset == 'synthetic':
        raw = load_penguins(None)   # triggers synthetic fallback
        raw['species'] = None
    else:  # 'penguins' (default)
        path = os.path.join(data_dir, 'penguins.csv')
        download_if_missing(_FEAT_URLS['penguins'], path)
        raw = load_penguins(path)

    pts_full = raw['points_full']
    is_synthetic = raw['species'] is None

    centered = pts_full - pts_full.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    pts_2d = centered @ Vt[:2].T

    label = FEAT_DATASET_LABELS.get(dataset, dataset)
    if is_synthetic:
        label = FEAT_DATASET_LABELS['synthetic']
    return (
        {
            'points':        pts_full,
            'points_2d':     pts_2d,
            'species':       raw['species'],
            'feature_names': raw['feature_names'],
        },
        label,
        is_synthetic,
    )


def load_mixed_data(data_dir):
    """
    Load USGS earthquake data and derive magnitude_class + depth_class categories.
    Falls back to synthetic weather-station data with climate/terrain.

    Returns (data_dict, label_str, is_synthetic_bool).
    data_dict keys:
      'lats', 'lons'           : radians
      'magnitude_class'        : string array  ('low'/'moderate'/'high')
      'depth_class'            : string array  ('shallow'/'intermediate'/'deep')
      'magnitude'              : float array or None
      'true_labels'            : None (no ground truth for real data)
    """
    path = os.path.join(data_dir, 'earthquakes.csv')
    raw = load_earthquakes(path)
    is_synthetic = raw['magnitude'] is None

    if not is_synthetic:
        mags = raw['magnitude']
        mag_class = np.where(mags < 4.5, 'low',
                    np.where(mags < 6.0, 'moderate', 'high'))
        # We don't have depth in the earthquake loader — derive proxy from latitude belt
        lats_deg = np.rad2deg(raw['lats'])
        depth_class = np.where(np.abs(lats_deg) < 20, 'shallow',
                      np.where(np.abs(lats_deg) < 50, 'intermediate', 'deep'))
        label = 'USGS Earthquakes + derived magnitude/depth classes'
    else:
        n = len(raw['lats'])
        rng = np.random.default_rng(77)
        mag_class = rng.choice(['low', 'moderate', 'high'], size=n)
        depth_class = rng.choice(['shallow', 'intermediate', 'deep'], size=n)
        label = 'Synthetic sphere clusters (earthquakes.csv not found)'

    return (
        {
            'lats':            raw['lats'],
            'lons':            raw['lons'],
            'magnitude':       raw['magnitude'],
            'magnitude_class': mag_class,
            'depth_class':     depth_class,
            'true_labels':     None,
        },
        label,
        is_synthetic,
    )


def gower_distance_matrix_v2(data, w_geo=1.0, w_cat=1.0):
    """
    Gower distance for mixed earthquake data (lat/lon + magnitude_class + depth_class).

    w_geo : weight for Haversine component (normalized to [0,1])
    w_cat : weight applied to each categorical column independently

    Each categorical column contributes w_cat to the weighted sum.
    Total weight = w_geo + n_cat * w_cat  (n_cat = 2 here).
    Distance is normalized to [0, 1].
    """
    n = len(data['lats'])
    lats, lons = data['lats'], data['lons']
    mag_cl  = data['magnitude_class']
    dep_cl  = data['depth_class']

    n_cat   = 2
    total_w = w_geo + n_cat * w_cat
    if total_w < 1e-15:
        total_w = 1.0

    D = np.zeros((n, n))
    for i in range(n):
        d_geo = haversine(lats[i], lons[i], lats, lons) / np.pi   # [0, 1]
        d_mag = (mag_cl[i] != mag_cl).astype(float)
        d_dep = (dep_cl[i] != dep_cl).astype(float)
        D[i, :] = (w_geo * d_geo + w_cat * d_mag + w_cat * d_dep) / total_w

    return D


def plot_unit_ball(ax, p, color, label=None):
    """
    Draw the unit ball {x : ||x||_p <= 1} in 2D.
    p >= 100 draws the Chebyshev (L-infinity) square.
    """
    theta = np.linspace(0, 2 * np.pi, 400)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    if p >= 100:
        # L-infinity: square with corners at ±1
        t = np.linspace(-1, 1, 100)
        xs = np.concatenate([t, np.ones(100), t[::-1], -np.ones(100)])
        ys = np.concatenate([-np.ones(100), t, np.ones(100), t[::-1]])
    else:
        # |x|^p + |y|^p = 1  →  x = sign(cos)*|cos|^(2/p)
        xs = np.sign(cos_t) * np.abs(cos_t) ** (2.0 / p)
        ys = np.sign(sin_t) * np.abs(sin_t) ** (2.0 / p)

    ax.fill(xs, ys, color=color, alpha=0.35)
    ax.plot(xs, ys, color=color, lw=2.0)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axhline(0, color='#555566', lw=0.7)
    ax.axvline(0, color='#555566', lw=0.7)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.25)
    if label:
        ax.set_title(label, fontsize=12, fontweight='bold', pad=6)


def plot_feature_scatter_2d(ax, pts_2d, labels, centers_2d, title, p=2.0):
    """
    2D scatter plot on PCA axes with soft Voronoi shading using Minkowski distance.

    pts_2d     : (N, 2)  PCA-projected data
    labels     : (N,)    cluster assignments
    centers_2d : (k, 2)  cluster centres in PCA space
    p          : Minkowski p for Voronoi
    """
    import matplotlib.colors as _mc

    k = len(centers_2d)
    colors = [CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(k)]

    # Voronoi background in PCA coordinate box
    x_lo, x_hi = pts_2d[:, 0].min() - 0.5, pts_2d[:, 0].max() + 0.5
    y_lo, y_hi = pts_2d[:, 1].min() - 0.5, pts_2d[:, 1].max() + 0.5
    res = 160
    gx = np.linspace(x_lo, x_hi, res)
    gy = np.linspace(y_lo, y_hi, res)
    gxx, gyy = np.meshgrid(gx, gy)
    grid = np.stack([gxx.ravel(), gyy.ravel()], axis=1)

    D_bg = np.zeros((len(grid), k))
    for j in range(k):
        D_bg[:, j] = minkowski_distance(grid, centers_2d[j], p)
    voronoi = np.argmin(D_bg, axis=1).reshape(res, res)
    rgb = np.array([_mc.to_rgb(c) for c in colors])
    img = rgb[voronoi]
    ax.imshow(img, extent=[x_lo, x_hi, y_lo, y_hi],
              aspect='auto', alpha=0.15, origin='lower')

    for c in range(k):
        mask = labels == c
        ax.scatter(pts_2d[mask, 0], pts_2d[mask, 1],
                   c=colors[c], s=18, alpha=0.85, linewidths=0, zorder=3)

    ax.scatter(centers_2d[:, 0], centers_2d[:, 1],
               c='white', s=220, marker='*', zorder=5,
               edgecolors='#111111', linewidths=0.8)

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel('PC 1', fontsize=14)
    ax.set_ylabel('PC 2', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=8)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3)


def plot_geo_map(ax, lats_rad, lons_rad, labels, medoid_idx, title,
                 cat_values=None, cat_markers=None, outlier_mask=None):
    """
    Equirectangular map coloured by cluster label.
    medoid_idx : (k,) indices of medoids (or None to skip centroid markers)
    cat_values : optional (N,) categorical array for marker shapes
    cat_markers: dict mapping category string → matplotlib marker
    outlier_mask: optional boolean (N,) for outlier overlay
    """
    import matplotlib.colors as _mc

    k = int(labels.max()) + 1 if len(labels) > 0 else 1
    colors = [CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(k)]

    lons_d = np.rad2deg(lons_rad)
    lats_d = np.rad2deg(lats_rad)

    if cat_values is not None and cat_markers is not None:
        cat_vals_arr = np.asarray(cat_values)
        for cat, mkr in cat_markers.items():
            mask_cat = cat_vals_arr == cat
            for c in range(k):
                mask = (labels == c) & mask_cat
                if not outlier_mask is None:
                    mask = mask & ~outlier_mask
                if mask.any():
                    ax.scatter(lons_d[mask], lats_d[mask],
                               c=colors[c], marker=mkr, s=20,
                               alpha=0.80, linewidths=0, zorder=3)
    else:
        for c in range(k):
            mask = labels == c
            if outlier_mask is not None:
                mask = mask & ~outlier_mask
            ax.scatter(lons_d[mask], lats_d[mask],
                       c=colors[c], s=18, alpha=0.80, linewidths=0, zorder=3)

    if medoid_idx is not None:
        ax.scatter(lons_d[medoid_idx], lats_d[medoid_idx],
                   c='white', s=260, marker='*', zorder=5,
                   edgecolors='#111111', linewidths=0.8)

    if outlier_mask is not None and outlier_mask.any():
        ax.scatter(lons_d[outlier_mask], lats_d[outlier_mask],
                   c='#e74c3c', marker='x', s=60, linewidths=1.5,
                   zorder=6, label='Outliers')

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel('Longitude', fontsize=14)
    ax.set_ylabel('Latitude', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=7)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.3)
