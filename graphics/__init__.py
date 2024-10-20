from .axes import *
from .plotting import *
from .utils import *

__all__ = [s for s in dir() if not s.startswith('_')]

# todo: matplotlib.animation.FuncAnimation
# todo: transfer my default kwargs to merge with mpl.rcParams
