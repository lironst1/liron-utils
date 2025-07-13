from scipy.signal import windows

from . import COLORS

DEFAULT_KWARGS = {

	# Figure
	"FIG_KW":          {

	},

	# Axes
	"SET_PROPS_KW":    {
		"sup_title":   None,  # str
		"ax_title":    None,  # list(str)
		"axis":        None,  # list(bool)
		"spines":      None,  # list(bool)
		"ticks":       None,  # list(bool/list)
		"tick_labels": None,  # list(bool/list)
		"labels":      None,  # list(list(str))
		"limits":      None,  # list(list(float))
		"view":        None,  # list(list(float))
		"grid":        None,  # list(bool)
		"legend":      True,  # list(bool/list(str))
		"legend_loc":  None,  # list(str)
		"colorbar":    False,  # list(bool/list)
		"xy_lines":    True,  # list(bool)
		"face_color":  None,  # list(color)
		"show_fig":    True,  # bool
		"open_dir":    False,  # bool
		"close_fig":   False,  # bool
	},

	"XY_LINES_KW":     {
		"color":     COLORS.DARK_GREY,
		"linewidth": 2
	},

	# 2D Plotting
	"PLOT_KW":         {

	},

	"ERRORBAR_KW":     {
		"linestyle":  "none",
		"marker":     ".",
		"markersize": 10,
		"ecolor":     COLORS.RED_E,
		"elinewidth": 1.4
	},

	"FILL_BETWEEN_KW": {
		"linestyle": "-",
		"color":     COLORS.LIGHT_GRAY,
		"alpha":     0.4
	},

	"SPECGRAM_KW":     {
		"NFFT":     4096,
		"window":   windows.blackmanharris(4096),
		"noverlap": int(0.85 * 4096),
		"pad_to": 4096 + int(1 * 4096),
		"cmap":     'inferno',
	},

	# 3D Plotting

	"PLOT_SURFACE_KW": {
		"cmap": 'viridis'
	}

}


def update_kwargs(key=None, **kwargs):
	"""
	Update the default kwargs with the new ones.

	Args:
		key (str):  Key to KWARGS to update.
		**kwargs:   New kwargs to update or merge.

	Returns:
		dict: Updated KWARGS.
	"""

	if key:
		DEFAULT_KWARGS[key.upper()].update(kwargs)

	else:
		for key in kwargs.keys():
			DEFAULT_KWARGS[key.upper()].update(kwargs[key])

	return DEFAULT_KWARGS


def merge_kwargs(**kwargs):
	"""
	Merge between empty/partially filled kwargs to the default ones, giving priority to the new settings.

	Args:
		**kwargs:

	Returns:
		kwargs | KWARGS (take KWARGS and overwrite it with kwargs where needed)
	"""

	for key in kwargs.keys():
		if kwargs[key] is None:
			kwargs[key] = dict()

		kwargs[key] = DEFAULT_KWARGS[key.upper()] | kwargs[key]

	return kwargs
