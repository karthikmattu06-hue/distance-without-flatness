"""
test_utils.py — Smoke test for demo_utils.py.

Imports everything, runs K=3 clustering on each geometry, runs validation,
generates plots, and prints "All tests passed".
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'demo'))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for testing
import matplotlib.pyplot as plt

from demo_utils import (
    # Distance functions
    haversine, poincare_distance, torus_distance, minkowski_distance, mahalanobis_distance,
    # Coordinate transforms
    latlon_to_cartesian, cartesian_to_latlon, torus_to_3d, mobius_addition,
    poincare_to_klein, klein_to_poincare,
    # Centroid functions
    spherical_mean, circular_mean, torus_centroid, einstein_midpoint, weighted_einstein_midpoint,
    # K-means clustering
    spherical_kmeans, hyperbolic_kmeans, torus_kmeans,
    minkowski_kmeans, mahalanobis_kmeans, kmedoids_pam,
    gower_distance_matrix, gower_kmedoids,
    # GMM clustering
    euclidean_gmm, spherical_gmm, torus_gmm, hyperbolic_gmm,
    # Validation
    silhouette_from_matrix, pve_score, compute_distance_matrix,
    # Data generation
    generate_sphere_clusters, generate_torus_clusters, generate_hyperbolic_clusters,
    generate_euclidean_clusters, generate_mixed_data, generate_ramachandran_data,
    # Data loaders (with fallbacks)
    load_earthquakes, load_penguins, load_ramachandran, save_ramachandran_csv,
    # Plotting helpers
    setup_dark_theme, fig_to_png,
    # Constants
    CLUSTER_COLORS,
)

K = 3
ERRORS = []


def check(name, condition, msg=""):
    if not condition:
        ERRORS.append(f"FAIL [{name}]: {msg}")
        print(f"  FAIL: {msg}")
    else:
        print(f"  OK:   {name}")


def check_clustering_result(name, result, n_points, k):
    labels, centers, history = result
    check(f"{name} labels shape", labels.shape == (n_points,), f"got {labels.shape}")
    check(f"{name} centers shape[0]", centers.shape[0] == k, f"got {centers.shape}")
    check(f"{name} history non-empty", len(history) > 0, "empty history")
    check(f"{name} history decreasing", history[-1] <= history[0] + 1e-6,
          f"first={history[0]:.4f} last={history[-1]:.4f}")
    check(f"{name} all clusters assigned", len(np.unique(labels)) > 0, "no clusters")


# ──────────────────────────────────────────────────────────────
# 1. DISTANCE FUNCTION SANITY CHECKS
# ──────────────────────────────────────────────────────────────
print("\n=== Distance functions ===")
# Haversine: antipodal = pi
d = haversine(0.0, 0.0, 0.0, np.pi)
check("haversine antipodal", abs(d - np.pi) < 1e-6, f"got {d:.4f}")

# Poincare: origin to (0.5, 0) = 2*arctanh(0.5)
d = poincare_distance(np.array([0.0, 0.0]), np.array([0.5, 0.0]))
check("poincare distance", abs(d - 2 * np.arctanh(0.5)) < 1e-6, f"got {d:.4f}")

# Torus: wraparound
d = torus_distance(np.array([0.1, 0.0]), np.array([2 * np.pi - 0.1, 0.0]))
check("torus wraparound", d < 0.5, f"should wrap, got {d:.4f}")

# Minkowski L1 = L-infinity check
u, v = np.array([1.0, 0.0]), np.array([0.0, 1.0])
check("minkowski L1", abs(minkowski_distance(u, v, p=1) - 2.0) < 1e-6)
check("minkowski L2", abs(minkowski_distance(u, v, p=2) - np.sqrt(2)) < 1e-6)
check("minkowski Linf", abs(minkowski_distance(u, v, p=100) - 1.0) < 1e-6)

# Mahalanobis with identity = Euclidean L2
cov_inv = np.eye(2)
check("mahalanobis=L2", abs(mahalanobis_distance(u, v, cov_inv) - np.sqrt(2)) < 1e-6)


# ──────────────────────────────────────────────────────────────
# 2. COORDINATE TRANSFORMS
# ──────────────────────────────────────────────────────────────
print("\n=== Coordinate transforms ===")
lat, lon = np.deg2rad(45), np.deg2rad(90)
x, y, z = latlon_to_cartesian(lat, lon)
lat2, lon2 = cartesian_to_latlon(x, y, z)
check("latlon roundtrip lat", abs(lat - lat2) < 1e-10)
check("latlon roundtrip lon", abs(lon - lon2) < 1e-10)

# Mobius: identity element
a = np.array([0.3, 0.1])
b = np.array([0.0, 0.0])
result = mobius_addition(a, b)
check("mobius with zero", np.allclose(result, a, atol=1e-6))

# Klein roundtrip
p = np.array([0.4, 0.3])
check("klein roundtrip", np.allclose(klein_to_poincare(poincare_to_klein(p)), p, atol=1e-9))


# ──────────────────────────────────────────────────────────────
# 3. K-MEANS CLUSTERING
# ──────────────────────────────────────────────────────────────
print("\n=== K-means clustering ===")

# Spherical
lats, lons, _ = generate_sphere_clusters(n_points=150, k=K, seed=7)
result = spherical_kmeans(lats, lons, k=K, max_iter=30, seed=42)
check_clustering_result("spherical_kmeans", result, len(lats), K)
assert result[1].shape == (K, 2), "centers should be (K,2) lat/lon"

# Hyperbolic
pts_hyp, _, _ = generate_hyperbolic_clusters(n_points=150, k=K, seed=7)
result = hyperbolic_kmeans(pts_hyp, k=K, max_iter=30, seed=42)
check_clustering_result("hyperbolic_kmeans", result, len(pts_hyp), K)

# Torus
pts_tor, _ = generate_torus_clusters(n_points=150, k=K, seed=7)
result = torus_kmeans(pts_tor, k=K, max_iter=30, seed=42)
check_clustering_result("torus_kmeans", result, len(pts_tor), K)

# Minkowski (L1, L2, Linf)
# Note: L-infinity uses midrange centroid (approximation), so history may not be monotone.
pts_euc, _ = generate_euclidean_clusters(n_points=150, k=K, seed=7)
for p_val in [1, 2]:
    result = minkowski_kmeans(pts_euc, k=K, p=p_val, max_iter=30, seed=42)
    check_clustering_result(f"minkowski_kmeans(p={p_val})", result, len(pts_euc), K)
# L-infinity: check shape only (midrange centroid doesn't guarantee monotone descent)
result_linf = minkowski_kmeans(pts_euc, k=K, p=100, max_iter=30, seed=42)
labels_li, centers_li, history_li = result_linf
check("minkowski_kmeans(p=100) labels shape", labels_li.shape == (len(pts_euc),))
check("minkowski_kmeans(p=100) centers shape", centers_li.shape[0] == K)
check("minkowski_kmeans(p=100) history non-empty", len(history_li) > 0)

# Mahalanobis
result = mahalanobis_kmeans(pts_euc, k=K, max_iter=30, seed=42)
check_clustering_result("mahalanobis_kmeans", result, len(pts_euc), K)

# K-Medoids on small distance matrix
D_small = np.random.default_rng(42).random((60, 60))
D_small = (D_small + D_small.T) / 2
np.fill_diagonal(D_small, 0)
result = kmedoids_pam(D_small, k=K, max_iter=5)
labels_km, medoids_km, history_km = result
check("kmedoids labels shape", labels_km.shape == (60,))
check("kmedoids medoids count", len(medoids_km) == K)

# Gower
mixed = generate_mixed_data(n_points=60, n_clusters=K, seed=7)
result = gower_kmedoids(mixed, k=K, max_iter=3)
check_clustering_result("gower_kmedoids", result, 60, K)


# ──────────────────────────────────────────────────────────────
# 4. GMM CLUSTERING
# ──────────────────────────────────────────────────────────────
print("\n=== GMM clustering ===")

# Euclidean GMM
pts_euc, _ = generate_euclidean_clusters(n_points=120, k=K, seed=7)
labels, means, history = euclidean_gmm(pts_euc, k=K, max_iter=30, seed=42)
check("euclidean_gmm labels", labels.shape == (len(pts_euc),))
check("euclidean_gmm means", means.shape == (K, 2))
check("euclidean_gmm history", len(history) > 0)

# Spherical GMM (needs 3D unit vectors)
lats, lons, _ = generate_sphere_clusters(n_points=120, k=K, seed=7)
x, y, z = latlon_to_cartesian(lats, lons)
pts_3d = np.stack([x, y, z], axis=1)
labels, means_3d, history = spherical_gmm(pts_3d, k=K, max_iter=30, seed=42)
check("spherical_gmm labels", labels.shape == (len(pts_3d),))
check("spherical_gmm means shape", means_3d.shape == (K, 3))
check("spherical_gmm means unit", np.allclose(np.linalg.norm(means_3d, axis=1), 1.0, atol=1e-6))

# Torus GMM
pts_tor, _ = generate_torus_clusters(n_points=120, k=K, seed=7)
labels, mu, history = torus_gmm(pts_tor, k=K, max_iter=30, seed=42)
check("torus_gmm labels", labels.shape == (len(pts_tor),))
check("torus_gmm mu shape", mu.shape == (K, 2))

# Hyperbolic GMM (slow due to Z-cache init; uses small N)
pts_hyp, _, _ = generate_hyperbolic_clusters(n_points=60, k=K, seed=7)
labels, means_h, history = hyperbolic_gmm(pts_hyp, k=K, max_iter=20, seed=42)
check("hyperbolic_gmm labels", labels.shape == (len(pts_hyp),))
check("hyperbolic_gmm means in disk", np.all(np.linalg.norm(means_h, axis=1) < 1.0))


# ──────────────────────────────────────────────────────────────
# 5. VALIDATION
# ──────────────────────────────────────────────────────────────
print("\n=== Validation ===")

# Build a small distance matrix and test silhouette
pts_small, true_labels = generate_euclidean_clusters(n_points=60, k=K, seed=7)
_, centers, _ = minkowski_kmeans(pts_small, k=K, p=2, max_iter=20, seed=42)
D_euc = compute_distance_matrix(pts_small, lambda u, v: np.sqrt(np.sum((u - v)**2)))
check("compute_distance_matrix shape", D_euc.shape == (60, 60))
check("compute_distance_matrix symmetric", np.allclose(D_euc, D_euc.T))

labels_km, _, _ = minkowski_kmeans(pts_small, k=K, p=2, max_iter=20, seed=42)
sil = silhouette_from_matrix(D_euc, labels_km)
check("silhouette in [-1,1]", -1 <= sil <= 1, f"got {sil:.4f}")

pve = pve_score(D_euc, labels_km)
check("pve in [0,1]", 0 <= pve <= 1, f"got {pve:.4f}")

# Single cluster should give PVE=0
labels_one = np.zeros(60, dtype=int)
pve_one = pve_score(D_euc, labels_one)
check("pve single cluster = 0", pve_one == 0.0, f"got {pve_one:.4f}")


# ──────────────────────────────────────────────────────────────
# 6. DATA LOADERS
# ──────────────────────────────────────────────────────────────
print("\n=== Data loaders ===")

eq = load_earthquakes()
check("load_earthquakes lats", 'lats' in eq and len(eq['lats']) > 0)
check("load_earthquakes lons", 'lons' in eq and len(eq['lons']) > 0)

pg = load_penguins()
check("load_penguins points", 'points' in pg and pg['points'].ndim == 2)
check("load_penguins feature_names", 'feature_names' in pg)

# Save and load ramachandran
csv_path = save_ramachandran_csv()
check("save_ramachandran_csv", os.path.exists(csv_path))
rama = load_ramachandran()
check("load_ramachandran phi", 'phi' in rama and len(rama['phi']) > 0)
check("load_ramachandran psi", 'psi' in rama and len(rama['psi']) > 0)


# ──────────────────────────────────────────────────────────────
# 7. PLOTTING HELPERS
# ──────────────────────────────────────────────────────────────
print("\n=== Plotting helpers ===")

setup_dark_theme()
fig, ax = plt.subplots(figsize=(4, 3))
ax.scatter([1, 2, 3], [1, 2, 3], c=CLUSTER_COLORS[:3])
ax.set_title("test")
png_bytes = fig_to_png(fig)
plt.close(fig)
check("fig_to_png returns bytes", isinstance(png_bytes, bytes) and len(png_bytes) > 100)
check("CLUSTER_COLORS length", len(CLUSTER_COLORS) >= 5)


# ──────────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
if ERRORS:
    print(f"FAILED ({len(ERRORS)} errors):")
    for e in ERRORS:
        print(f"  {e}")
    sys.exit(1)
else:
    print("All tests passed")
