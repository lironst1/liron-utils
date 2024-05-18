import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from .utils.default_kwargs import update_kwargs
from ..time import get_time_str
from ..files import MAIN_FILE_DIR
from ..pure_python.dicts import DL_to_LD
from ..pure_python.docstring import copy_docstring_and_deprecators

SHOW_FIG = True


class AxesLironUpper:
    def __init__(self,
                 shape: tuple = (1, 1),
                 sharex: (bool, str) = False, sharey: (bool, str) = False,
                 projection: str = None,
                 fig: Figure = None, axs: Axes = None,
                 subplot_kw: dict = None, gridspec_kw: dict = None, **figure_kw):
        """
        Create new figure with (possibly) subplots

        Parameters
        ----------
        shape :             tuple (int, int)
                            number of rows, columns (in case of subplots). Default is (1,1)
        sharex, sharey :    bool or {'none', 'all', 'row', 'col'}
                            Link x, y axes (zoom together)
        projection :        str, optional
                            one of ['3d', 'aitoff', 'hammer', 'lambert', 'mollweide', 'polar', 'rectilinear'].
                            Default is 'rectilinear'
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

    def draw_axis_lines(self):
        @self._vectorize(cls=self)
        def _draw_axis_lines(ax: Axes, **axis_lines_kw):
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

        _draw_axis_lines()

    def sup_title(self, title: str):
        self.fig.suptitle(title)

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
            if grid is False:
                grid = None
            elif type(grid) is not str:  # todo: check condition
                grid = 'both'

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

    def ax_colorbar(self, mappable: matplotlib.cm.ScalarMappable):
        @self._vectorize(cls=self, mappable=mappable)
        def _ax_colorbar(ax, mappable: matplotlib.cm.ScalarMappable):
            if mappable is False or len(ax.images) == 0 or hasattr(ax, "colorbar_drawn"):  # if no image is plotted
                return
            elif mappable is True:
                mappable = ax.images[0]

            ax.figure.colorbar(mappable=mappable, ax=ax)
            ax.colorbar_drawn = True

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
                  save_file_name: (str, bool) = False, save_fig_kw: dict = None, show_fig: bool = SHOW_FIG,
                  **set_props_kw):
        """

        Args:
            save_file_name:         File name. False - don't save, True - save to MAIN_FILE_DIR, str - specify location
            save_fig_kw:            kwargs for function fig.savefig()
            show_fig:               Show figure or keep it hidden
            **set_props_kw:         See utils.default_kwargs.KWARGS{"SET_PROPS_KW"}

        Returns:

        """

        set_props_kw = update_kwargs(set_props_kw=set_props_kw)["set_props_kw"]

        caller = lambda func, arg: func(arg) if arg is not None else None

        # Supreme Title
        caller(self.sup_title, set_props_kw["sup_title"])

        # Axes Title
        caller(self.ax_title, set_props_kw["ax_title"])

        # Labels
        caller(self.ax_labels, set_props_kw["labels"])

        # Limits
        caller(self.ax_limits, set_props_kw["limits"])

        # View (in case of 3D)
        caller(self.ax_view, set_props_kw["view"])

        # Grid
        caller(self.ax_grid, set_props_kw["grid"])

        # Legend
        self.ax_legend(set_props_kw["legend"], set_props_kw["legend_loc"])

        # Colorbar
        caller(self.ax_colorbar, set_props_kw["colorbar"])

        # Axis Lines
        if set_props_kw["axis_lines"]:
            self.draw_axis_lines()

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
        if show_fig:
            self.show_fig()

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


@copy_docstring_and_deprecators(AxesLironUpper.draw_axis_lines)
def draw_axis_lines(ax: Axes, **axis_lines_kw):

    AxesLironUpper(axs=ax, **axis_lines_kw)


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
              save_file_name: (str, bool) = False, save_fig_kw: dict = None, show_fig: bool = SHOW_FIG,
              **set_props_kw):
    if ax is None:
        ax = plt.gca()

    AxesLironUpper(axs=ax).set_props(save_file_name=save_file_name,
                                     save_fig_kw=save_fig_kw,
                                     show_fig=show_fig,
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
