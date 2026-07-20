# flake8: noqa: F401, F403

from ...pure_python import is_notebook
from ..common import COLORS, get_pixel_color, get_savefig_file_name, hex2rgb, rgb2hex
from .axes import *
from .plotting import *
from .utils import *

__all__ = [s for s in dir() if not s.startswith("_")]

update_rc_params()  # Change default MatPlotLib parameters (e.g, figure size, label size, grid, colors, etc.)

if is_notebook():
    update_rc_params("liron-utils-notebook")

# TODO:
#   - matplotlib.animation.FuncAnimation
#   - Transfer my default kwargs to merge with mpl.rcParams
