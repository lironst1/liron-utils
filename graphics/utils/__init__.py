import matplotlib.style

# matplotlib.style.use('seaborn')


from .rc_params import *
from . import COLORS

update_rcParams(RC_PARAMS)  # Change default MatPlotLib parameters (e.g, figure size, label size, grid, colors, etc.)
