# MAA Demo Spec: "The Shape of Closeness"
# Interactive Streamlit Demo for Live Talk
# ═══════════════════════════════════════════════════════════════

## OVERVIEW

Build a Streamlit multi-page app for a live talk at the Mathematical Association of America.
The app demonstrates how different distance metrics affect clustering on curved and mixed-feature surfaces.

**Narrative structure:** 6 "Acts", each revealing a failure of naive distance metrics and the fix.
**Design philosophy:** "Show the wrong answer first, then fix it live."
**Audience:** Math professors, grad students, undergrads. They know geometry but may not know clustering.

---

## GLOBAL DESIGN REQUIREMENTS

### Theme: Dark (projector-optimized)
- Background: #0E1117 (Streamlit dark default) or custom dark
- Text: white/light gray
- Plot backgrounds: dark (#1a1a2e or similar)
- Use matplotlib dark_background style or equivalent for all plots
- All plots: large axis labels (14pt+), large titles (16pt+), thick lines (2pt+)
- No plot element should be thin or light enough to wash out on a projector

### Layout
- NO sidebar navigation. Use a horizontal tab bar or page selector at the top.
- Each act fills ONE screen — no scrolling needed
- Big action buttons for dramatic reveals (e.g., "Switch to Correct Metric", "Add Outliers")
- Minimal text on screen — the presenter talks, the demo shows
- Act indicator at top (e.g., "Act 3 of 6 — Infinite Space in a Finite Circle")

### Color Palette for Clusters
```python
CLUSTER_COLORS = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12', '#9b59b6',
                  '#1abc9c', '#e67e22', '#ecf0f1']
```

### Typography
- Title of each act: large, bold, evocative (not technical)
- One-line subtitle: the technical translation
- Use st.markdown with custom CSS for sizing

---

## DATA GENERATION

All data should be generated with fixed seeds for reproducibility.
Each act uses ~300 points in 5 pre-clustered groups.
Data generation functions should be defined in a shared utils module.

---

## ACT 1: "The Flat World Assumption"
**Subtitle:** *Euclidean vs Haversine distance on the sphere*

### Layout (single screen):
```
┌──────────────────────────────────────────────┐
│  Act 1 of 6 — The Flat World Assumption      │
│  Euclidean vs Haversine distance on a sphere  │
├──────────────────────┬───────────────────────┤
│                      │                       │
│   3D Globe View      │   3D Globe View       │
│   (Euclidean K-Means)│   (Haversine K-Means) │
│                      │                       │
├──────────────────────┴───────────────────────┤
│  [🔄 Regenerate Data]                        │
│  Cluster sizes: ...  |  Cluster sizes: ...   │
│  Agreement: X% of points differ              │
└──────────────────────────────────────────────┘
```

### Functionality:
- Generate 300 points pre-clustered on a unit sphere (5 clusters, spread_deg=15)
- LEFT: Run standard Euclidean K-means on the (lat, lon) coordinates directly (treating them as flat 2D)
- RIGHT: Run Spherical K-means with Haversine distance + spherical mean centroids
- Both plots: 3D sphere wireframe + colored points + centroid markers
- Below: show number of points that differ between the two methods
- Button: "Regenerate Data" with a different seed

### Key visual:
- Use different centroid markers: Euclidean gets a "?" or "✗" marker, Haversine gets a "★"
- Highlight disagreeing points (points assigned differently) with a red ring

### Implementation notes:
- Haversine formula, spherical mean (Cartesian average + re-project)
- For Euclidean: just use (lat, lon) as 2D points with np.linalg.norm
- Use matplotlib 3D scatter with dark background
- Sphere wireframe: translucent surface

---

## ACT 2: "The Edge of the World"
**Subtitle:** *Toroidal distance and the wraparound problem*

### Layout:
```
┌──────────────────────────────────────────────┐
│  Act 2 of 6 — The Edge of the World          │
│  What happens when space wraps around?        │
├──────────────────────┬───────────────────────┤
│                      │                       │
│  2D Flat Torus       │   2D Flat Torus       │
│  (Euclidean K-Means) │   (Toroidal K-Means)  │
│  points split at     │   cluster heals       │
│  boundary            │   across boundary     │
│                      │                       │
├──────────────────────┴───────────────────────┤
│  [Toggle 3D Donut View]                      │
│  Disagreeing points: N (X% near boundary)    │
└──────────────────────────────────────────────┘
```

### Functionality:
- 300 points on flat torus [0, 2π)², with clusters deliberately straddling the wraparound boundary
- LEFT: Euclidean K-means (ignores wrapping) — boundary clusters get torn apart
- RIGHT: Toroidal K-means (wraparound distance + circular mean centroids)
- Both plots: 2D square with Voronoi territory shading
- Mark the wraparound boundaries with colored dashed lines
- Show: "X% of disagreements are near a boundary"
- Toggle button: switch to 3D torus (donut) surface view with points plotted on it

### Key visual:
- On the 2D plot, draw arrows at edges showing "this edge connects to that edge"
- Voronoi shading makes the boundary behavior very visible

### Implementation notes:
- Toroidal distance: min(|d|, L-|d|) per dimension
- Circular mean: atan2(mean(sin), mean(cos)) per dimension
- Generate at least 2 clusters near the (0, 2π) boundary

---

## ACT 3: "Infinite Space in a Finite Circle"
**Subtitle:** *Hyperbolic distance on the Poincaré disk*

### Layout:
```
┌──────────────────────────────────────────────┐
│  Act 3 of 6 — Infinite Space in a Finite Circle │
│  Clustering in hyperbolic geometry            │
├──────────────────────┬───────────────────────┤
│                      │                       │
│  Poincaré Disk       │  Poincaré Disk        │
│  (Euclidean K-Means) │  (Hyperbolic K-Means) │
│  + Voronoi regions   │  + Voronoi regions    │
│                      │                       │
├──────────────────────┴───────────────────────┤
│  Distance from center:                       │
│  r=0.5 → d=1.1  |  r=0.9 → d=2.9  |  r=0.99 → d=5.3  │
│  [Show Distance Distortion Plot]             │
└──────────────────────────────────────────────┘
```

### Functionality:
- 300 points in Poincaré disk, clusters at varying depths (some near center, some near boundary)
- LEFT: Euclidean K-means
- RIGHT: Hyperbolic K-means (Poincaré distance + Einstein midpoint)
- Both: unit disk boundary, concentric depth circles with hyperbolic distance labels
- Voronoi shading using each respective metric
- Bottom: distance-from-center table showing exponential blowup
- Toggle: show scatter plot of Euclidean vs Hyperbolic pairwise distances (colored by radius)

### Key visual:
- Concentric circles with hyperbolic distance annotations — this is the "wow" visual
- Voronoi boundaries curve differently between the two methods

### Implementation notes:
- Poincaré distance: arccosh(1 + 2||u-v||² / ((1-||u||²)(1-||v||²)))
- Einstein midpoint: Lorentz-factor-weighted average
- Keep points inside disk (max norm 0.97)

---

## ACT 4: "Stretching the Rulers"
**Subtitle:** *Minkowski norms and Mahalanobis distance*

### Layout:
```
┌──────────────────────────────────────────────┐
│  Act 4 of 6 — Stretching the Rulers          │
│  When data has shape, your distance should too │
├─────────┬──────────┬──────────┬──────────────┤
│  Unit   │  Unit    │  Unit    │  Unit        │
│  Ball   │  Ball    │  Ball    │  Ball        │
│  p=1    │  p=2     │  p=∞    │  Mahalanobis │
├─────────┴──────────┴──────────┴──────────────┤
│         [p slider: 0.5 ──●── ∞]              │
├──────────────────────┬───────────────────────┤
│  Clustering with     │  Mahalanobis          │
│  current p value     │  "Unwarping" view     │
│  + Voronoi           │  Original → Whitened  │
└──────────────────────┴───────────────────────┘
```

### Functionality:
- 300 points in elongated, correlated elliptical clusters (tilted at different angles)
- TOP ROW: Unit ball gallery showing neighborhood shapes for p=1, 2, ∞, and Mahalanobis ellipse
- SLIDER: continuously vary p from 0.5 to 20 (with named stops: L1, L2, L∞)
  - Clustering and Voronoi update live as p changes
- LEFT BOTTOM: Minkowski K-means with current p, showing Voronoi regions
- RIGHT BOTTOM: Side-by-side original space vs Mahalanobis-whitened space
  - Show covariance ellipses on original
  - Show the same data after whitening transform — ellipses become circles

### Key visual:
- The unit ball morphing as p changes — diamond → circle → square
- The "unwarping" panel — visually transforming correlated data to uncorrelated

### Implementation notes:
- Minkowski distance: (Σ|u_i - v_i|^p)^(1/p)
- Mahalanobis: sqrt((u-v)ᵀ Σ⁻¹ (u-v))
- Whitening transform: multiply by Σ^(-1/2)

---

## ACT 5: "When Means Don't Exist"
**Subtitle:** *K-Medoids, Gower distance, and mixed-type data*

### Layout:
```
┌──────────────────────────────────────────────┐
│  Act 5 of 6 — When Means Don't Exist         │
│  K-Medoids + Gower distance for mixed data    │
├──────────────────────┬───────────────────────┤
│                      │                       │
│  K-Means on Sphere   │  K-Medoids on Sphere  │
│                      │                       │
├──────────────────────┴───────────────────────┤
│  [💥 Add 15 Outliers]   [🔄 Reset]           │
│  Center shift: K-Means Xᵒ vs K-Medoids Yᵒ   │
├──────────────────────────────────────────────┤
│  ─── Mixed Data: Geography + Categories ───   │
├────────┬─────────┬───────────────────────────┤
│ Geo    │ Cat     │ Blended Gower             │
│ Only   │ Only    │ (w_geo slider)            │
└────────┴─────────┴───────────────────────────┘
```

### Functionality:
- TOP HALF: Outlier robustness demo
  - Same 300 spherical points, K-Means vs K-Medoids side by side (2D map projection)
  - Big "Add 15 Outliers" button — when pressed:
    - Outliers appear as red X markers on both plots
    - Both methods re-cluster
    - Show centroid/medoid shift in degrees
    - K-Means centroids visibly drift; K-Medoids medoids stay anchored
  - "Reset" button clears outliers

- BOTTOM HALF: Gower distance demo
  - 300 points on sphere with categorical features (climate, terrain)
  - Three small maps side by side: geography-only, categories-only, blended Gower
  - Slider: geographic weight (0 = categories only → 5 = geography dominates)
    - Maps update live as weight changes
  - Points shaped by terrain marker, colored by cluster

### Key visual:
- The outlier button is the dramatic reveal — audience sees centroids jump
- The weight slider shows smooth transition between geographic and categorical clustering

### Implementation notes:
- K-Medoids PAM: BUILD + SWAP on precomputed distance matrix
- Gower: weighted average of normalized Haversine + categorical matching
- Outliers: uniform random on sphere

---

## ACT 6: "How Many Clusters?"
**Subtitle:** *PVE, Silhouette, and choosing K*

### Layout:
```
┌──────────────────────────────────────────────┐
│  Act 6 of 6 — How Many Clusters?             │
│  Validating your choice of K                  │
├──────────────────────────────────────────────┤
│  Surface: [Sphere ▼]  Metric: [Haversine ▼]  │
│  Algorithm: [K-Means ▼]                       │
├──────────────────────┬───────────────────────┤
│                      │                       │
│  PVE Elbow Curve     │  Silhouette Score     │
│  (K=2..10)           │  vs K                 │
│  vertical line at    │  optimal K highlighted│
│  chosen K            │                       │
│                      │                       │
├──────────────────────┴───────────────────────┤
│  K slider: 2 ─────────●───────────── 10      │
├──────────────────────┬───────────────────────┤
│                      │                       │
│  Current clustering  │  Silhouette strip     │
│  visualization       │  plot (per point,     │
│  (updates with K)    │  colored by cluster)  │
│                      │                       │
└──────────────────────┴───────────────────────┘
```

### Functionality:
- Dropdowns to select surface (Sphere, Torus, Hyperbolic, Flat/Minkowski), metric, and algorithm
- Pre-compute clustering for K=2..10 for the selected combination
- TOP: PVE elbow curve + Silhouette score curve, both with vertical line at current K
- K SLIDER: dragging it updates the bottom panels live
- BOTTOM LEFT: the clustering visualization for current K (appropriate to selected surface — globe, disk, flat, etc.)
- BOTTOM RIGHT: Silhouette strip plot — horizontal bars for each point, colored by cluster, sorted by score
  - This shows WHICH points are well-placed and which are borderline

### Key formulas (display on hover or in a collapsible section):
- PVE(K) = 1 - TWCSS_K / TWCSS_1
- TWCSS_K = Σᵢ d(xᵢ, cₖ)² (using the selected distance metric)
- Silhouette: s(i) = (b(i) - a(i)) / max(a(i), b(i))

### Key visual:
- The elbow is immediately visible
- The strip plot makes silhouette intuitive — "tall green bars = good, short/red bars = bad"
- Changing K slider and watching the strip plot react is very engaging

### Implementation notes:
- Silhouette from precomputed N×N distance matrix (works with any metric)
- TWCSS uses squared distance from each point to its center (mean, medoid, or GMM component center depending on algorithm)
- Cache the sweep results — don't recompute on every slider change

---

## FILE STRUCTURE

```
demo/
├── app.py                  # Main Streamlit app with page routing
├── utils/
│   ├── __init__.py
│   ├── data_generation.py  # All data generation functions (sphere, torus, disk, elliptical, mixed)
│   ├── distances.py        # All distance functions (haversine, poincare, torus, minkowski, mahalanobis, gower)
│   ├── clustering.py       # All clustering algorithms (spherical kmeans, hyperbolic kmeans, torus kmeans, minkowski kmeans, mahalanobis kmeans, kmedoids PAM)
│   ├── validation.py       # PVE, silhouette, TWCSS computation
│   ├── plotting.py         # All plotting functions (dark theme, consistent style)
│   └── theme.py            # CSS, colors, fonts, dark theme config
├── pages/
│   ├── act1_sphere.py
│   ├── act2_torus.py
│   ├── act3_hyperbolic.py
│   ├── act4_minkowski.py
│   ├── act5_medoids.py
│   └── act6_choosing_k.py
└── requirements.txt        # streamlit, numpy, matplotlib, scipy
```

---

## STREAMLIT CONFIGURATION

```python
st.set_page_config(
    page_title="The Shape of Closeness",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="collapsed"  # No sidebar — use top navigation
)
```

### Custom CSS (inject via st.markdown):
```css
/* Dark theme overrides */
.stApp { background-color: #0E1117; }

/* Large act titles */
.act-title {
    font-size: 2.5rem;
    font-weight: bold;
    color: #ECF0F1;
    margin-bottom: 0;
}
.act-subtitle {
    font-size: 1.2rem;
    color: #95A5A6;
    font-style: italic;
    margin-top: 0;
}

/* Big action buttons */
.stButton > button {
    font-size: 1.3rem;
    padding: 0.75rem 2rem;
    border-radius: 8px;
}

/* Act indicator */
.act-indicator {
    font-size: 0.9rem;
    color: #7F8C8D;
    letter-spacing: 2px;
    text-transform: uppercase;
}
```

---

## PLOTTING STYLE (apply to ALL matplotlib figures)

```python
import matplotlib.pyplot as plt

def apply_dark_theme():
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': '#0E1117',
        'axes.facecolor': '#1a1a2e',
        'axes.edgecolor': '#2C3E50',
        'axes.labelcolor': '#ECF0F1',
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.color': '#95A5A6',
        'ytick.color': '#95A5A6',
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'legend.facecolor': '#1a1a2e',
        'legend.edgecolor': '#2C3E50',
        'text.color': '#ECF0F1',
        'grid.color': '#2C3E50',
        'grid.alpha': 0.3,
        'lines.linewidth': 2,
        'figure.dpi': 100,
    })
```

---

## PERFORMANCE NOTES

- Use @st.cache_data for data generation (keyed by seed)
- Use @st.cache_data for clustering results (keyed by data hash + K + metric)
- Pre-compute the K=2..10 sweep when Act 6 loads, show a spinner
- Distance matrices for N=300 are small — compute on the fly, no issues
- 3D matplotlib plots can be slow in Streamlit — consider using plotly for 3D if performance is poor
- All computations must be pure numpy (no sklearn dependency needed, we implement everything from scratch)

---

## NAVIGATION

Use streamlit-option-menu or simple st.radio with horizontal layout:

```python
act = st.radio(
    "",
    ["Act 1: Sphere", "Act 2: Torus", "Act 3: Hyperbolic",
     "Act 4: Minkowski", "Act 5: Medoids", "Act 6: Choosing K"],
    horizontal=True
)
```

Each act loads its corresponding page module.

---

## SUMMARY OF KEY INTERACTIONS PER ACT

| Act | Primary Interaction | Dramatic Reveal |
|-----|-------------------|-----------------|
| 1 | Side-by-side auto-loads | Euclidean fails on sphere → Haversine fixes it |
| 2 | Toggle 3D donut view | Boundary cluster splits → heals with toroidal metric |
| 3 | Show distance distortion plot | r=0.99 is distance 5.3, not 0.99 |
| 4 | p slider (live Voronoi update) | Unit ball morphing + Mahalanobis unwarping |
| 5 | "Add Outliers" button | K-Means drifts, K-Medoids holds + weight slider for Gower |
| 6 | K slider + surface/metric dropdowns | Elbow + silhouette strip plot update live |
