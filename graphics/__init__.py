from .base import *
from .axes import *
from .plotting import *
from .utils import *

__all__ = [s for s in dir() if not s.startswith('_')]

# TODO:
#   - matplotlib.animation.FuncAnimation
#   - Transfer my default kwargs to merge with mpl.rcParams
