from abc import ABC, abstractmethod


class GeometryBase(ABC):
    name: str = ""
    plot_type: str = "2d"  # "2d" or "3d"
    default_K: int = 4
    default_n_points: int = 200
    spread_range: tuple = (0.1, 5.0)
    spread_label: str = "Cluster Spread"
    spread_default: float = 1.0
    spread_help: str = "Controls how spread out each cluster is."

    # Distance metrics available for K-Means in this geometry.
    # Keys = display names, values = internal identifiers passed to kmeans methods.
    distance_options: dict = {"Default": "default"}

    @abstractmethod
    def generate_data(self, K, n_points, spread, seed):
        """Return {'points': ndarray, 'labels': ndarray, 'params': dict}"""

    @abstractmethod
    def kmeans_init(self, points, K, seed=123, **kwargs):
        """Return initial K-Means state dict with keys:
        'centroids', 'labels', 'cost_history', 'iteration'"""

    @abstractmethod
    def kmeans_step(self, points, state):
        """One K-Means iteration. Return updated state dict."""

    @abstractmethod
    def kmeans_full(self, points, K, max_iter=100, seed=123, **kwargs):
        """Return {'labels', 'centroids', 'cost_history'}"""

    @abstractmethod
    def em_init(self, points, K, seed=123, **kwargs):
        """Return initial EM state dict."""

    @abstractmethod
    def em_step(self, points, state):
        """One EM iteration. Return updated state dict."""

    @abstractmethod
    def em_full(self, points, K, max_iter=200, seed=123, **kwargs):
        """Full EM. Return final state + 'll_history'."""

    @abstractmethod
    def build_plot(self, points, labels, centers, responsibilities=None,
                   show_soft=False, title="", true_labels=None):
        """Build the main Plotly figure."""

    @abstractmethod
    def build_convergence_plot(self, history, title=""):
        """Build convergence line chart."""
