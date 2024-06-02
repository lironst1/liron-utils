import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from .utils.default_kwargs import update_kwargs
from ..time import get_time_str
from ..files import MAIN_FILE_DIR, open_file
from ..pure_python.dicts import DL_to_LD
from ..pure_python.docstring import copy_docstring_and_deprecators


class AxesLironUpper:
	def __init__(self,
			shape: tuple = (1, 1),
			sharex: (bool, str) = False, sharey: (bool, str) = False,
			projection: str = None,
			fig: Figure = None, axs: Axes = None,
			subplot_kw: dict = None, gridspec_kw: dict = None, **figure_kw):
		"""
		Create a new figure with (possibly) subplots

		Parameters
		----------
		shape :             tuple (int, int)
							number of rows, columns (in case of subplots). Default is (1,1)
		sharex, sharey :    bool or {'none', 'all', 'row', 'col'}
							Share the x or y `~matplotlib.axis` with sharex and/or sharey.
				            The axis will have the same limits, ticks, and scale as the axis
				            of the shared axes.
		projection :        {None, 'aitoff', 'hammer', 'lambert', 'mollweide', 'polar', 'rectilinear', str}, optional
							The projection type of the subplot (`~.axes.Axes`). *str* is the
				            name of a custom projection, see `~matplotlib.projections`. The
				            default None results in a 'rectilinear' projection.
		fig :               Figure, optional
							Usually None, or send own figure
		axs :               Axes, optional
							Usually None, or send own axis/axes
		subplot_kw :
		gridspec_kw :
		figure_kw :


		Examples
		--------
			>> from plotting import plot
			>> Ax = AxesLiron(2, 3)
			>> t = np.linspace(0, 10, 1001)
			>> plot(Ax.axs[0,0], t, np.sin(t))
			>> plot(Ax.axs[1,0], t, np.cos(t))
			>> plot(Ax.axs[0,1], t, np.log(t))
			>> plot(Ax.axs[1,1], t, np.exp(t))
			>> plot(Ax.axs[0,2], t, np.sqrt(t))
			>> plot(Ax.axs[1,2], t, np.square(t))

			>> Ax.set_props(sup_title="abc", ax_title=[1, 2, 3, 4, 5, 6], grid=[True, True, False, True, False, True], show_fig=True)

		"""

		self.fig = fig
		self.axs = np.atleast_2d(axs)
		self.func_animation = None

		if fig is None and axs is None:
			if subplot_kw is None:
				subplot_kw = dict()
			subplot_kw = {'projection': projection} | subplot_kw

			nrows, ncols = shape
			self.fig, self.axs = plt.subplots(nrows=nrows, ncols=ncols,
					sharex=sharex, sharey=sharey,
					squeeze=False,
					subplot_kw=subplot_kw, gridspec_kw=gridspec_kw, **figure_kw)

		elif fig is not None:
			self.axs = np.atleast_2d(self.fig.axes)

		elif axs is not None:
			self.fig = self.axs[0, 0].figure

	@staticmethod
	def _vectorize(cls, **vec_params):
		def vectorize_decorator(func):
			def vectorize_wrapper(*args, **kwargs):
				m, n = cls.axs.shape

				params_list = DL_to_LD(vec_params)
				if params_list is None or len(params_list) != m * n:
					params_list = np.repeat(vec_params, m * n)

				out = np.empty((m, n), dtype=object)

				for i in range(m):
					for j in range(n):
						out[i, j] = func(cls.axs[i, j], *args, **params_list[j * m + i], **kwargs)

				return out

			return vectorize_wrapper

		return vectorize_decorator

	def draw_xy_lines(self, **axis_lines_kw):
		@self._vectorize(cls=self)
		def _draw_xy_lines(ax: Axes, **axis_lines_kw):
			"""
			Draw x-y axes lines to look bolder than the rest of the grid lines

			Args:
				ax:
				**axis_lines_kw:

			Returns:

			"""

			if hasattr(ax, 'zaxis') or hasattr(ax, "axis_lines_drawn"):  # Don't draw axis lines for 3D plots
				return

			axis_lines_kw = update_kwargs(axis_lines_kw=axis_lines_kw)["axis_lines_kw"]

			xlim = ax.get_xlim()
			ylim = ax.get_ylim()

			ax.axhline(**axis_lines_kw)
			ax.axvline(**axis_lines_kw)

			ax.set_xlim(*xlim, auto=True)
			ax.set_ylim(*ylim, auto=True)

		_draw_xy_lines()

	def sup_title(self, title: str):
		self.fig.suptitle(title)

	def ax_axis(self, axis: bool):
		@self._vectorize(cls=self, axis=axis)
		def _ax_axis(ax: Axes, axis: (bool, str)):
			"""

			Parameters
			----------
			ax :
			axis :  ================ ===========================================================
		            Value            Description
		            ================ ===========================================================
		            'off' or `False` Hide all axis decorations, i.e. axis labels, spines,
		                             tick marks, tick labels, and grid lines.
		                             This is the same as `~.Axes.set_axis_off()`.
		            'on' or `True`   Do not hide all axis decorations, i.e. axis labels, spines,
		                             tick marks, tick labels, and grid lines.
		                             This is the same as `~.Axes.set_axis_on()`.
		            'equal'          Set equal scaling (i.e., make circles circular) by
		                             changing the axis limits. This is the same as
		                             ``ax.set_aspect('equal', adjustable='datalim')``.
		                             Explicit data limits may not be respected in this case.
		            'scaled'         Set equal scaling (i.e., make circles circular) by
		                             changing dimensions of the plot box. This is the same as
		                             ``ax.set_aspect('equal', adjustable='box', anchor='C')``.
		                             Additionally, further autoscaling will be disabled.
		            'tight'          Set limits just large enough to show all data, then
		                             disable further autoscaling.
		            'auto'           Automatic scaling (fill plot box with data).
		            'image'          'scaled' with axis limits equal to data limits.
		            'square'         Square plot; similar to 'scaled', but initially forcing
		                             ``xmax-xmin == ymax-ymin``.
		            ================ ===========================================================

			Returns
			-------

			"""
			ax.axis(axis)

		_ax_axis()

	def ax_spines(self, spines: bool):
		@self._vectorize(cls=self, spines=spines)
		def _ax_spines(ax: Axes, spines: (str, list, bool)):
			locs = np.array(["left", "bottom", "top", "right"])

			if type(spines) is str:
				spines = [spines]

			if type(spines) is list:
				locs = np.array(spines)
				idx = [True] * locs.size
			elif type(spines) is bool:
				idx = [False] * locs.size
				if spines:
					idx = [True] * locs.size
			else:
				raise ValueError("'spines' must be one of (str, list, bool).")

			if locs[idx].size > 0:
				ax.spines[locs[idx].tolist()].set_visible(True)
			if locs[np.logical_not(idx)].size > 0:
				ax.spines[locs[np.logical_not(idx)].tolist()].set_visible(False)

		_ax_spines()

	def ax_ticks(self, ticks: (bool, dict, list)):
		@self._vectorize(cls=self, ticks=ticks)
		def _ax_ticks(ax: Axes, ticks: (bool, list)):
			tick_values = [[], [], []]
			tick_labels = [[], [], []]

			if ticks is False:
				pass
			elif type(ticks) is list:
				assert len(ticks) <= 3, "len(ticks) must be the same as the graph dimensionality."
				for i in range(len(ticks)):
					tick_values[i] = ticks[i].keys()
					tick_labels[i] = ticks[i].values()

			else:
				raise ValueError(
						"'ticks' must be given either a boolean or a list of dicts of the form {val: 'label'}.")

			ax.set_xticks(tick_values[0], tick_labels[0])
			ax.set_yticks(tick_values[1], tick_labels[1])
			if hasattr(ax, "set_zticks") and len(ticks) == 3:
				ax.zticks(tick_values[2], tick_labels[2])

		_ax_ticks()

	def ax_title(self, title: str):
		@self._vectorize(cls=self, title=title)
		def _ax_title(ax: Axes, title: str):
			ax.set_title(title)

		_ax_title()

	def ax_labels(self, labels: list):
		@self._vectorize(cls=self, labels=labels)
		def _ax_labels(ax: Axes, labels: list):
			labels = np.atleast_1d(labels)  # In case labels=None or labels is just 1 list (only xlabel)
			ax.set_xlabel(labels[0])
			if np.size(labels) >= 2:
				ax.set_ylabel(labels[1])
			if np.size(labels) >= 3:
				ax.set_zlabel(labels[2])

		_ax_labels()

	def ax_limits(self, limits: list):
		@self._vectorize(cls=self, limits=limits)
		def _ax_limits(ax: Axes, limits: list):
			limits = np.array(limits, dtype=object)
			limits = np.atleast_1d(limits)  # In case limits=None or limits is just 1 list (only xlim)

			ax.set_xlim(limits[0])
			if len(limits) >= 2:
				ax.set_ylim(limits[1])
			if len(limits) >= 3:
				ax.set_zlim(limits[2])

		_ax_limits()

	def ax_view(self, view: list):
		@self._vectorize(cls=self, view=view)
		def _ax_view(ax: Axes, view: list):
			ax.view_init(view[0], view[1])

		_ax_view()

	def ax_grid(self, grid):
		@self._vectorize(cls=self, grid=grid)
		def _ax_grid(ax: Axes, grid):
			ax.grid(grid)

		_ax_grid()

	def ax_legend(self, legend: list, legend_loc):
		@self._vectorize(cls=self, legend=legend, legend_loc=legend_loc)
		def _ax_legend(ax: Axes, legend: list, legend_loc):
			if legend is None or legend is True:
				handles, labels = ax.get_legend_handles_labels()
				if len(handles) > 0:
					ax.legend(handles, labels, loc=legend_loc)  # todo [::-1] ?
			else:
				ax.legend(legend, loc=legend_loc)

		_ax_legend()

	def ax_colorbar(self, mappable: matplotlib.cm.ScalarMappable, colorbar_each: bool):
		@self._vectorize(cls=self, mappable=mappable, colorbar_each=colorbar_each)
		def _ax_colorbar(ax, mappable: matplotlib.cm.ScalarMappable, colorbar_each: bool):
			if (mappable is False
					or len(ax.images) == 0
					or hasattr(ax.figure, "colorbar_drawn")
					or hasattr(ax, "colorbar_drawn")):
				return
			elif mappable is None or mappable is True:
				mappable = ax.images[0]

			if colorbar_each:
				ax.figure.colorbar(mappable=mappable, ax=ax)
				ax.colorbar_drawn = True

			else:  # one common colorbar
				ax.figure.subplots_adjust(right=0.8)
				cax = ax.figure.add_axes([0.85, 0.15, 0.05, 0.7])
				ax.figure.colorbar(mappable=mappable, cax=cax)
				ax.figure.colorbar_drawn = True

		_ax_colorbar()

	def ax_face_color(self, face_color):
		@self._vectorize(cls=self, face_color=face_color)
		def _ax_face_color(ax, face_color):
			ax.set_facecolor(face_color)

		_ax_face_color()

	def save_fig(self, file_name: str = None, **savefig_kw):
		"""
		Save figure as file

		Args:
			file_name:              File name
			**savefig_kw:           kwargs for function fig.savefig

		Returns:

		"""

		file_name = _get_savefig_file_name(file_name)

		format = os.path.splitext(file_name)[-1]

		if format == ".gif":
			assert hasattr(self, "func_animation"), "Call plot_animation() before saving gif."
			self.func_animation.save(file_name, **savefig_kw)
		else:
			self.fig.savefig(file_name, **savefig_kw)

		return file_name

	def show_fig(self):
		self.fig.show()

	def set_props(self,
			save_file_name: (str, bool) = False, save_fig_kw: dict = None,
			**set_props_kw):
		"""

		Args:
			save_file_name:         File name. False - don't save, True - save to MAIN_FILE_DIR, str - specify location
			save_fig_kw:            kwargs for function fig.savefig()
			**set_props_kw:         See utils.default_kwargs.KWARGS{"SET_PROPS_KW"}

		Returns:

		"""

		set_props_kw = update_kwargs(set_props_kw=set_props_kw)["set_props_kw"]

		caller = lambda func, arg: func(arg) if arg is not None else None

		# Supreme Title
		caller(self.sup_title, set_props_kw["sup_title"])

		# Axes Title
		caller(self.ax_title, set_props_kw["ax_title"])

		# Axis
		caller(self.ax_axis, set_props_kw["axis"])

		# Axis Spines
		caller(self.ax_spines, set_props_kw["spines"])

		# Axis Ticks
		caller(self.ax_ticks, set_props_kw["ticks"])

		# Axis Labels
		caller(self.ax_labels, set_props_kw["labels"])

		# Axis Limits
		caller(self.ax_limits, set_props_kw["limits"])

		# View (in case of 3D)
		caller(self.ax_view, set_props_kw["view"])

		# Grid
		caller(self.ax_grid, set_props_kw["grid"])

		# Legend
		self.ax_legend(set_props_kw["legend"], set_props_kw["legend_loc"])

		# Colorbar
		self.ax_colorbar(set_props_kw["colorbar"], set_props_kw["colorbar_each"])

		# x-y Lines (through the origin)
		if set_props_kw["xy_lines"]:
			self.draw_xy_lines()

		# Face Color
		caller(self.ax_face_color, set_props_kw["face_color"])

		# -------------------------------------------------------

		# Save Figure
		file_name = None
		if save_file_name is not False:  # if user wants to save fig
			if save_file_name is True:
				# True sets default file name, otherwise provide a string
				save_file_name = None
			if save_fig_kw is None:
				save_fig_kw = dict()

			file_name = self.save_fig(file_name=save_file_name, **save_fig_kw)

		# Show Figure
		if set_props_kw["show_fig"]:
			self.show_fig()
			if set_props_kw["open_dir"]:
				open_file(os.path.split(file_name)[0])

		return file_name


def new_figure(nrows=1, ncols=1,
		sharex=False, sharey=False,
		projection=None,
		squeeze=True,
		subplot_kw=None, gridspec_kw=None, **figure_kw) -> (Figure, Axes):
	"""
	Create new figure with (possibly) subplots

	Args:
		nrows:              number of rows (in case of subplots)
		ncols:              number of columns (in case of subplots)
		sharex:             Link x-axis (zoom together)
		sharey:             Link y-axis (zoom together)
		squeeze:            Extra dimensions are squeezed out
		projection:         ['3d', 'aitoff', 'hammer', 'lambert', 'mollweide', 'polar', 'rectilinear']
		subplot_kw:
		gridspec_kw:
		**figure_kw

	Returns:

	"""

	if subplot_kw is None:
		subplot_kw = dict()
	subplot_kw = {'projection': projection} | subplot_kw

	fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
			sharex=sharex, sharey=sharey,
			squeeze=squeeze,
			subplot_kw=subplot_kw, gridspec_kw=gridspec_kw, **figure_kw)

	return fig, axs


@copy_docstring_and_deprecators(AxesLironUpper.draw_xy_lines)
def draw_xy_lines(ax: Axes, **axis_lines_kw):
	AxesLironUpper(axs=ax).draw_xy_lines(**axis_lines_kw)


@copy_docstring_and_deprecators(AxesLironUpper.save_fig)
def save_fig(fig: Figure = None, file_name: str = None, **savefig_kw):
	"""
	Save figure as file

	Args:
		fig:                    Figure to be saved
		file_name:              File name
		**savefig_kw:           kwargs for function fig.savefig

	Returns:

	"""

	if fig is None:
		fig = plt.gcf()

	AxesLironUpper(fig=fig).save_fig(file_name, **savefig_kw)


@copy_docstring_and_deprecators(AxesLironUpper.set_props)
def set_props(ax: Axes = None,
		save_file_name: (str, bool) = False, save_fig_kw: dict = None,
		**set_props_kw):
	if ax is None:
		ax = plt.gca()

	AxesLironUpper(axs=ax).set_props(save_file_name=save_file_name,
			save_fig_kw=save_fig_kw,
			**set_props_kw)


def _get_savefig_file_name(file_name: str):
	"""

	Parameters
	----------
	file_name :     str
					If <file_name> doesn't contain a directory name, the default saving directory
					will be "<CUR_DIR>/figs", where <CUR_DIR> is the current directory of the directory
					of the Python file being called.

	Returns
	-------
		Full path of file to be saved.
	"""

	if file_name is None or os.path.dirname(file_name) == '':
		dir_name = os.path.join(MAIN_FILE_DIR, "figs")
	else:
		dir_name = os.path.dirname(file_name)

	if not os.path.exists(dir_name):
		os.mkdir(dir_name)

	if file_name is None:
		file_name = os.path.join(dir_name, f"fig {get_time_str()}")
	elif os.path.dirname(file_name) == '':
		file_name = os.path.join(dir_name, file_name)

	return file_name
