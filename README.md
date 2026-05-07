# Distance Without Flatness: Metric-Driven Clustering on Spheres, Tori and Hyperbolic Spaces

**Karthik Mattu, Adit Dhall, Thejas Nagesh Gowda**  
Data and Predictive Analytics Center, School of Mathematics and Statistics  
Rochester Institute of Technology

---

## Overview

The standard K-means algorithm minimizes within-cluster sum of **squared Euclidean distances** and relies on the linear structure of ℝⁿ to compute centroids. On curved surfaces this assumption breaks: squared Euclidean cost no longer reflects the intrinsic separation between points.

This project revisits K-means under three non-Euclidean geometries and studies how the choice of metric reshapes the resulting clusters:

| Surface | Geometry | Distance Metric |
|---|---|---|
| Sphere S² | Positive curvature | Haversine distance |
| Flat torus T² | Flat periodicity | Periodic Euclidean (modulo 2π) |
| Poincaré disk D | Negative curvature | Poincaré metric: `2/(1 − ‖x‖²)` |

Beyond these three, the framework extends naturally to product manifolds, Stiefel and Grassmann manifolds, spaces of symmetric positive-definite matrices, and further Riemannian settings encountered in modern data analysis.

---

## Repository Structure

```
.
├── spherical_kmeans.ipynb          # K-means with Haversine distance on S²
├── torus_kmeans.ipynb              # K-means with periodic metric on T²
├── hyperbolic_kmeans.ipynb         # K-means with Poincaré metric on D
├── mahalanobis_minkowski_kmeans.ipynb  # Minkowski (Lᵖ) and Mahalanobis K-means
├── gower_kmedoids.ipynb            # K-Medoids with Gower distance (mixed-type data)
├── kmedoids_sphere.ipynb           # K-Medoids on the sphere
├── euclidean_gmm.ipynb             # Gaussian Mixture Models (Euclidean baseline)
├── hyperbolic_gmm.ipynb            # GMM in hyperbolic space
├── spherical_vmf_mixture.ipynb     # von Mises–Fisher mixture on S²
├── toroidal_vmises_mixture.ipynb   # Toroidal von Mises mixture on T²
├── mixed_type_mixture.ipynb        # Mixture models for mixed-type data
├── test_utils.py                   # Shared test utilities
├── Karthik_Mattu_abstract.pdf      # Conference abstract (MAA)
├── Karthik_Mattu_abstract.tex      # LaTeX source for abstract
└── demo/                           # Interactive Marimo demo app
    ├── demo_marimo.py              # Main app (all 6 tabs)
    ├── demo_utils.py               # Shared utilities
    ├── geometries/                 # Geometry implementations
    │   ├── base.py
    │   ├── euclidean.py
    │   ├── spherical.py
    │   ├── toroidal.py
    │   ├── hyperbolic.py
    │   └── mixed_type.py
    ├── data/                       # Real datasets
    ├── requirements.txt
    └── README.md
```

---

## Notebooks

### Core Clustering Methods

| Notebook | Description |
|---|---|
| `spherical_kmeans.ipynb` | Spherical K-means using Haversine distance; centroids updated via Cartesian mean re-projected to S² |
| `torus_kmeans.ipynb` | Toroidal K-means using `min(|d|, 2π−|d|)` per angular dimension; centroids via circular mean |
| `hyperbolic_kmeans.ipynb` | Hyperbolic K-means on the Poincaré disk; centroids via Einstein (Lorentz-weighted) midpoint |
| `mahalanobis_minkowski_kmeans.ipynb` | Minkowski K-means (varying p from L¹ to L∞) and Mahalanobis K-means for correlated/elliptical clusters |

### Medoid-Based Methods

| Notebook | Description |
|---|---|
| `kmedoids_sphere.ipynb` | PAM K-Medoids on S² — robust to outliers since the medoid must be an actual data point |
| `gower_kmedoids.ipynb` | K-Medoids with Gower distance for mixed numeric + categorical data |

### Mixture Models

| Notebook | Description |
|---|---|
| `euclidean_gmm.ipynb` | Gaussian Mixture Models (EM algorithm) as Euclidean baseline |
| `hyperbolic_gmm.ipynb` | GMM adapted for hyperbolic space |
| `spherical_vmf_mixture.ipynb` | von Mises–Fisher mixture model on the sphere |
| `toroidal_vmises_mixture.ipynb` | Wrapped von Mises mixture model on the flat torus |
| `mixed_type_mixture.ipynb` | Mixture models for datasets with heterogeneous feature types |

---

## Interactive Demo

An interactive demo was built in **Marimo** for a live talk at the **MAA Seaway Section, Spring 2026**, themed *"The Shape of Closeness"*.

### Run the Demo

```bash
cd demo
pip install -r requirements.txt
marimo run demo_marimo.py
```

For development mode (code visible):
```bash
marimo edit demo_marimo.py
```

### Optional: Real Datasets

The demo works out of the box with synthetic data. To use real datasets:

```bash
# USGS Earthquakes M4.5+ (2024) — for the Sphere tab
curl -o demo/data/earthquakes.csv "https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime=2024-01-01&endtime=2024-12-31&minmagnitude=4.5&limit=500"

# Palmer Penguins — for the Features tab
curl -o demo/data/penguins.csv "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv"
```

### Demo Tabs

| Tab | Surface | Distance | Real Dataset |
|---|---|---|---|
| Sphere | S² | Haversine | USGS Earthquakes M4.5+ (2024) |
| Torus | T² | Wraparound | Protein backbone angles (Ramachandran) |
| Hyperbolic | Poincaré disk | Poincaré metric | Simulated animal taxonomy |
| Features | ℝⁿ | Minkowski / Mahalanobis | Palmer Penguins |
| Mixed | S² + categories | Gower | Earthquakes + derived classes |

---

## Key Concepts

**Why non-Euclidean distance matters:**  
On a sphere, two points near opposite poles are far apart in Haversine distance but may appear close when their lat/lon coordinates are treated as flat 2D. Euclidean K-means will misclassify points near the poles. The same failure occurs on the torus (clusters split at the boundary) and in the Poincaré disk (points near the boundary are exponentially far from the center, not merely "close to the edge").

**Centroid updates on curved spaces:**

- **Sphere:** Compute the Cartesian mean of unit vectors, then re-normalize to S²
- **Torus:** Use the circular mean: `atan2(mean(sin θ), mean(cos θ))` per angular dimension
- **Poincaré disk:** Use the Einstein (Lorentz-factor-weighted) midpoint in the hyperboloid model

**When means don't exist — K-Medoids:**  
For mixed-type or non-metric data, the centroid may not be a valid data point (or may not exist at all). K-Medoids (PAM) always selects an actual observation as the cluster representative, making it both interpretable and robust to outliers.

---

## Requirements

```bash
pip install numpy scipy matplotlib jupyter
```

For the interactive demo:
```bash
pip install marimo numpy scipy matplotlib
```

All clustering and distance computations are implemented from scratch in pure NumPy — no scikit-learn dependency.

---

## References

1. S. Lloyd, *Least squares quantization in PCM*, IEEE Trans. Inform. Theory **28** (1982), 129–137.
2. K. V. Mardia and P. E. Jupp, *Directional Statistics*, Wiley, Chichester, 2000.
3. M. P. do Carmo, *Riemannian Geometry*, Birkhäuser, Boston, 1992.
4. M. Nickel and D. Kiela, *Poincaré embeddings for learning hierarchical representations*, NeurIPS **30** (2017), 6338–6347.
5. M. Barbosu and T. Wiandt, *On the Riemannian geometry of the planetary three-body problem*, Romanian Astron. J. **29** (2019), 149–156.
6. X. Pennec, *Intrinsic statistics on Riemannian manifolds*, J. Math. Imaging Vis. **25** (2006), 127–154.
