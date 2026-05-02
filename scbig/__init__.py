"""scBIG: biologically informed gene modules for perturbation prediction.

The top-level package intentionally avoids importing JAX/Diffrax-heavy modules.
Import subpackages such as ``scbig.models`` or ``scbig.experiments`` directly
when those dependencies are available.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
