from . import COLORS

KWARGS = {

	# Figure
	"FIG_KW":          {

	},

	# Axes
	"SET_PROPS_KW":    {
		"sup_title":  None,  # str
		"ax_title":   None,  # list(str)
		"labels":     None,  # list(str)
		"limits":     None,  # list(str)
		"view":       None,  # list(str)
		"grid":       None,  # list(bool)
		"legend":     None,  # list(str)
		"legend_loc": "best",  # list(str)
		"colorbar": True,  # list(bool/str/mappable)
		"axis_lines": True,  # list(bool)
		"face_color": None,  # list(color)
	},

	"AXIS_LINES_KW":   {
		"color":     COLORS.DARK_GREY,
		"linewidth": 2
	},

	# 2D Plotting
	"PLOT_KW":         {

	},

	"ERRORBAR_KW":     {
		"fmt":        ".",
		# "color":      COLORS.DARK_BLUE,
		"markersize": 10,
		"ecolor":     COLORS.RED_E,
		"elinewidth": 1.4
	},

	"SPECGRAM_KW":     {

	},

	# 3D Plotting

	"PLOT_SURFACE_KW": {
		"cmap": 'viridis'
	}

}


def update_kwargs(**kwargs):
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

		kwargs[key] = KWARGS[key.upper()] | kwargs[key]

	return kwargs
