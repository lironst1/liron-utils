from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("liron_utils")
except PackageNotFoundError:  # package not installed (e.g., running from a source checkout)
    __version__ = "0.0.0"

# Import all submodules
from . import (
    files,
    graphics,
    machine_learning,
    manim_animations,
    pure_python,
    signal_processing,
    symbolic_math,
    time,
    uncertainties_math,
    web,
)

__all__ = [
    "__version__",
    "graphics",
    "signal_processing",
    "machine_learning",
    "pure_python",
    "symbolic_math",
    "time",
    "uncertainties_math",
    "files",
    "web",
    "manim_animations",
]
