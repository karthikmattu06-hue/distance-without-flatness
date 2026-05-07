def _get_registry():
    """Lazy import to allow testing individual modules."""
    from .euclidean import EuclideanGeometry
    from .spherical import SphericalGeometry
    from .hyperbolic import HyperbolicGeometry
    from .toroidal import ToroidalGeometry
    from .mixed_type import MixedTypeGeometry

    return {
        "Euclidean (R²)": EuclideanGeometry(),
        "Sphere (S²)": SphericalGeometry(),
        "Poincaré Disk": HyperbolicGeometry(),
        "Torus (S¹ × S¹)": ToroidalGeometry(),
        "Mixed-Type": MixedTypeGeometry(),
    }


# Will be populated on first access in app.py
GEOMETRY_REGISTRY = None


def get_registry():
    global GEOMETRY_REGISTRY
    if GEOMETRY_REGISTRY is None:
        GEOMETRY_REGISTRY = _get_registry()
    return GEOMETRY_REGISTRY
