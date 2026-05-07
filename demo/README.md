# The Shape of Closeness — Interactive Demo

An interactive marimo demo exploring distance metrics for clustering
across geometric surfaces. Built for a live talk at the Mathematical
Association of America (MAA Seaway Section, Spring 2026).

## Quick Start

```bash
pip install -r requirements.txt
marimo run demo_marimo.py
```

For development with code visible:

```bash
marimo edit demo_marimo.py
```

## Data Setup

The demo works without any downloads (synthetic fallbacks kick in automatically
with a visible warning). For real datasets, download before running:

```bash
# USGS Earthquakes M4.5+ (2024)
curl -o data/earthquakes.csv "https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime=2024-01-01&endtime=2024-12-31&minmagnitude=4.5&limit=500"

# Palmer Penguins
curl -o data/penguins.csv "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv"
```

The Ramachandran protein angle data (`data/ramachandran.csv`) is bundled and
uses simulated secondary structure regions — no download needed.

## Surfaces & Datasets

| Tab | Surface | Distance | Real Data |
|-----|---------|----------|-----------|
| 🌍 Sphere | Sphere | Haversine | USGS Earthquakes M4.5+ (2024) |
| 🍩 Torus | Torus | Wraparound | Protein Backbone Angles (Ramachandran) |
| 🔵 Hyperbolic | Poincaré Disk | Poincaré metric | Simulated Animal Taxonomy |
| 📏 Features | ℝⁿ | Minkowski, Mahalanobis | Palmer Penguins |
| 🔀 Mixed | Sphere + Categories | Gower | Earthquakes + derived classes |

## Presenter Notes

- Use `marimo run demo_marimo.py` for presentation mode (code hidden)
- Use `marimo edit demo_marimo.py` for development (code visible)
- Each tab is self-contained — present in any order
- The **K slider** lets the audience discover the natural number of clusters
- The **algorithm selector** (K-Means / K-Medoids / GMM) shows that distance matters more than algorithm
- The **Key Insight** callout at the bottom of each tab is a clean takeaway for the audience

### Suggested Flow per Tab

1. Note the data source label at the top
2. Observe the side-by-side Euclidean vs correct-metric comparison
3. Slide K from 2 to 10 — watch the PVE elbow form
4. Find the "best" K from the silhouette peak
5. Switch algorithms — do the clusters change much?
6. Check the strip plot at the best K — how many points have negative silhouette?

### Tab-Specific Interactions

- **Tab 2 (Torus):** Toggle the 3D donut view checkbox
- **Tab 3 (Hyperbolic):** Toggle the distance distortion scatter plot
- **Tab 4 (Features):** Slide the `p` value and watch Voronoi boundaries change shape
- **Tab 5 (Mixed):** Press "Add 15 Outliers", observe centroid shift vs medoid stability, press "Reset"

## File Structure

```
demo/
├── demo_marimo.py          # Main app (all 6 tabs, fully polished)
├── demo_utils.py           # Shared utilities (distances, clustering, plotting)
├── data/
│   ├── earthquakes.csv     # USGS earthquake data (download separately)
│   ├── penguins.csv        # Palmer Penguins (download separately)
│   └── ramachandran.csv    # Protein angles (bundled, simulated)
├── requirements.txt        # marimo, numpy, matplotlib, scipy
└── README.md               # This file
```
