import matplotlib as mpl
import matplotlib.style

# matplotlib.style.use('seaborn')


from .RC_PARAMS import rcParams, plot_color_cycler
from . import COLORS


print("Changing default MatPlotLib rcParams...")
mpl.rcParams.update(rcParams)  # Change default MatPlotLib parameters (e.g, figure size, label size, grid, colors, etc.)


def update_rcParams(new_rcParams: dict):
    mpl.rcParams.update(new_rcParams)
