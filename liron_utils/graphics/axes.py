# pylint: disable=no-value-for-parameter,too-many-lines

import copy
import dataclasses
import functools
import os
import typing
import warnings
from collections.abc import Callable, Iterable, Sequence

import matplotlib.animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes as Axes_plt
from matplotlib.figure import Figure
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.ticker import ScalarFormatter

from ..files import MAIN_FILE_DIR, mkdirs, open_file
from ..pure_python.dicts import dl_to_ld
from ..time import TIME_STR, get_time_str
from .utils.default_kwargs import merge_kwargs

# TODO: change Ax.axs to be 1D (top->down, left->right), enable custom shapes (large, small axes)

_R = typing.TypeVar("_R")
_P = typing.ParamSpec("_P")

# Catch-all for opaque/arbitrary-dim arrays.
_Array = np.ndarray[typing.Any, np.dtype[typing.Any]]


def _extract_layout_padding(fig_kw: dict[str, typing.Any]) -> list[typing.Any]:
    """Pop layout padding keys from ``fig_kw`` and return them in fixed order.

    Args:
        fig_kw: Figure kwargs from which ``w_pad``, ``h_pad``, ``wspace``, ``hspace``
            are removed in place.

    Returns:
        Length-4 list ``[w_pad, h_pad, wspace, hspace]`` (entries default to None).
    """
    padding: list[typing.Any] = [None] * 4
    for i, key in enumerate(["w_pad", "h_pad", "wspace", "hspace"]):
        if key in fig_kw:
            padding[i] = fig_kw.pop(key)
    return padding


def _apply_constrained_layout_padding(fig: Figure, padding: list[typing.Any]) -> None:
    """Apply padding values to ``fig`` when it uses a constrained layout engine.

    Args:
        fig: Target figure.
        padding: Length-4 list ``[w_pad, h_pad, wspace, hspace]`` (see
            :func:`_extract_layout_padding`).
    """
    layout_engine = fig.get_layout_engine()
    if isinstance(layout_engine, ConstrainedLayoutEngine):
        typing.cast(typing.Any, layout_engine).set(
            w_pad=padding[0],
            h_pad=padding[1],
            hspace=padding[2],
            wspace=padding[3],
        )


def _normalize_share_flag(share: bool | str) -> bool:
    """Normalize an ``"all"`` / ``"none"`` / bool axis-share flag to a plain bool.

    Args:
        share: ``"all"`` (True), ``"none"`` (False), or an existing bool.

    Returns:
        Bool form of ``share``.

    Raises:
        AssertionError: If ``share`` is not one of the accepted forms.
    """
    if share == "all":
        return True
    if share == "none":
        return False
    assert isinstance(share, bool), "'sharex' and 'sharey' must be bool when using grid_layout."
    return share


def _as_range(value: int | tuple[int, int]) -> tuple[int, int]:
    """Promote a single int row/col index to a ``[start, end)`` range tuple.

    Args:
        value: Either a tuple ``(start, end)`` or a single index ``i``.

    Returns:
        ``(value, value + 1)`` for an int input, or ``value`` unchanged for a tuple.
    """
    if isinstance(value, int):
        return value, value + 1
    return value


@dataclasses.dataclass
class _FigureBuildConfig:
    """Configuration bundle for constructing a new figure with subplots.

    Attributes:
        shape: ``(nrows, ncols)`` for the subplot grid.
        grid_layout: Optional span layout (see :class:`_Axes`).
        share: ``(sharex, sharey)`` pair (bool or ``"all"`` / ``"none"``).
        subplot_kw: Forwarded to ``Figure.add_subplot``.
        gridspec_kw: Forwarded to ``Figure.add_gridspec``.
        fig_kw: Forwarded to ``plt.figure`` (with layout shortcuts already injected).
    """

    shape: tuple[int, int]
    grid_layout: Sequence[Sequence[int | tuple[int, int]]] | None
    share: tuple[bool | str, bool | str]  # (sharex, sharey)
    subplot_kw: dict[str, typing.Any]
    gridspec_kw: dict[str, typing.Any] | None
    fig_kw: dict[str, typing.Any]


def _normalize_ticks_labels(
    ticks: typing.Any,
    labels: typing.Any,
    ndim: int,
) -> tuple[list[typing.Any], list[typing.Any]]:
    """Expand bool/None/dict ticks/labels into per-dimension lists.

    Args:
        ticks: Scalar (bool/None/dict) or per-dim list specifying ticks.
        labels: Scalar (bool/None) or per-dim list specifying labels.
        ndim: Axis dimensionality (2 or 3).

    Returns:
        Tuple ``(ticks_per_dim, labels_per_dim)``, each of length ``ndim``.
    """
    if ticks is None or ticks is True:
        ticks = [True] * ndim
        if labels is None:
            labels = [True] * ndim
    elif ticks is False:
        ticks = [False] * ndim
        if labels is None:
            labels = [False] * ndim
    elif isinstance(ticks, dict):
        ticks = [ticks] + [True] * (ndim - 1)

    if labels is True:
        labels = [True] * ndim
    elif labels is False:
        labels = [False] * ndim
    elif labels is None:
        labels = [None] * ndim

    return ticks, labels


def _collect_current_ticks_labels(ax: Axes_plt, ndim: int) -> dict[str, dict[int, typing.Any]]:
    """Snapshot the current ticks and tick-labels of an axis.

    Args:
        ax: Source axes.
        ndim: Axis dimensionality (2 or 3).

    Returns:
        Mapping ``{"ticks": {0: x, 1: y, 2: z}, "labels": {0: x, 1: y, 2: z}}``.
    """
    return {
        "ticks": {
            0: ax.get_xticks(),
            1: ax.get_yticks(),
            2: ax.get_zticks() if ndim == 3 else [],  # type: ignore[attr-defined]
        },
        "labels": {
            0: ax.get_xticklabels(),
            1: ax.get_yticklabels(),
            2: ax.get_zticklabels() if ndim == 3 else [],  # type: ignore[attr-defined]
        },
    }


def _filter_ticks_within_limits(
    ax: Axes_plt,
    d: dict[str, dict[int, typing.Any]],
    ndim: int,
) -> None:
    """Drop ticks (and their labels) that fall outside ``ax``'s current limits.

    Args:
        ax: Source axes (provides the per-dim limit getters).
        d: Tick/label dict produced by :func:`_collect_current_ticks_labels`, mutated in place.
        ndim: Axis dimensionality (2 or 3).
    """
    funcs = [ax.get_xlim, ax.get_ylim]
    if ndim == 3:
        funcs += [ax.get_zlim]  # type: ignore[attr-defined]

    for i, func in enumerate(funcs):
        lim = func()
        idx = np.logical_and(min(lim) <= d["ticks"][i], d["ticks"][i] <= max(lim))
        if len(d["ticks"][i]) > 0:
            d["ticks"][i] = d["ticks"][i][idx]
        if len(d["labels"][i]) > 0:
            d["labels"][i] = np.array(d["labels"][i])[idx]


def _apply_tick_override(
    d: dict[str, dict[int, typing.Any]],
    i: int,
    tick_i: typing.Any,
    label_i: typing.Any,
) -> None:
    """Update ``d`` in place for one axis with the caller-provided tick spec.

    Args:
        d: Tick/label dict (see :func:`_collect_current_ticks_labels`).
        i: Axis index (0=x, 1=y, 2=z).
        tick_i: Bool, dict (``{position: label}``), or iterable of positions.
        label_i: Companion label spec (only consulted to validate ``tick_i``).

    Raises:
        ValueError: If ``tick_i`` is not bool, dict, or iterable.
    """
    if tick_i is None or tick_i is True or tick_i is np.True_:
        return
    if tick_i is False or tick_i is np.False_:
        d["ticks"][i] = []
        if label_i is None:
            d["labels"][i] = []
        return
    if isinstance(tick_i, dict):
        assert label_i is None or label_i is np.True_, "When 'ticks' is a dict, 'labels' must not be given."
        d["ticks"][i] = tick_i.keys()
        d["labels"][i] = tick_i.values()
        return
    if isinstance(tick_i, Iterable):
        d["ticks"][i] = tick_i
        d["labels"][i] = tick_i
        return
    raise ValueError("'ticks' must be given either as a boolean, list[list] or list[dict].")


def _apply_label_override(
    d: dict[str, dict[int, typing.Any]],
    i: int,
    tick_i: typing.Any,
    label_i: typing.Any,
) -> None:
    """Update ``d`` in place for one axis with the caller-provided label spec.

    Args:
        d: Tick/label dict (see :func:`_collect_current_ticks_labels`).
        i: Axis index (0=x, 1=y, 2=z).
        tick_i: Companion tick spec; used to validate the label length.
        label_i: Bool or iterable of labels.

    Raises:
        AssertionError: If ``len(label_i) != len(tick_i)``.
        ValueError: If ``label_i`` is not bool or iterable.
    """
    if label_i is None or label_i is True or label_i is np.True_:
        return
    if label_i is False or label_i is np.False_:
        d["labels"][i] = []
        return
    if isinstance(label_i, Iterable):
        label_i_list = list(label_i)
        assert len(label_i_list) == len(tick_i), "len(labels[i]) must equal len(ticks[i])."
        d["labels"][i] = label_i_list
        return
    raise ValueError("'labels' must be given either as a boolean or list[list].")


def _apply_tick_label_overrides(
    d: dict[str, dict[int, typing.Any]],
    ticks: list[typing.Any],
    labels: list[typing.Any],
) -> None:
    """Apply per-dim tick and label overrides to ``d`` in place.

    Args:
        d: Tick/label dict (see :func:`_collect_current_ticks_labels`).
        ticks: Per-dim tick specs (length ``ndim``).
        labels: Per-dim label specs (length ``ndim``).
    """
    for i, (tick_i, label_i) in enumerate(zip(ticks, labels)):
        _apply_tick_override(d, i, tick_i, label_i)
        _apply_label_override(d, i, ticks[i], label_i)


def _set_ticks_on_axes(ax: Axes_plt, d: dict[str, dict[int, typing.Any]], ndim: int) -> None:
    """Write the (overridden) ticks and labels in ``d`` back onto ``ax``.

    Args:
        ax: Target axes.
        d: Tick/label dict (see :func:`_collect_current_ticks_labels`).
        ndim: Axis dimensionality (2 or 3).
    """
    ax.set_xticks(d["ticks"][0], d["labels"][0])
    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=True))

    ax.set_yticks(d["ticks"][1], d["labels"][1])
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=True))

    if ndim == 3:
        ax.set_zticks(d["ticks"][2], d["labels"][2])  # type: ignore[attr-defined]
        ax.zaxis.set_major_formatter(ScalarFormatter(useOffset=True))  # type: ignore[attr-defined]


class _Axes:

    def __init__(
        self,
        shape: tuple[int, int] = (1, 1),
        grid_layout: Sequence[Sequence[int | tuple[int, int]]] | None = None,
        *,
        fig: Figure | None = None,
        axs: Axes_plt | Sequence[Axes_plt] | _Array | None = None,
        subplot_kw: dict[str, typing.Any] | None = None,
        gridspec_kw: dict[str, typing.Any] | None = None,
        **fig_kw: typing.Any,
    ) -> None:
        """Create a new figure with (possibly) subplots, or wrap existing ones.

        When both ``fig`` and ``axs`` are None, a fresh figure is built via
        ``plt.subplots`` (or ``plt.figure`` + a custom gridspec when ``grid_layout``
        is given). Otherwise, the existing figure/axes are wrapped.

        Args:
            shape: ``(nrows, ncols)`` for the subplot grid.
            grid_layout: Optional list of span specs ``[[row_range, col_range], ...]``,
                where each range is either an int ``i`` (interpreted as ``(i, i+1)``)
                or a ``(start, end)`` tuple. Example: ``[[(0, 2), (0, 2)], [(0, 1), (2, 4)]]``.
            fig: Existing figure to wrap (mutually exclusive with ``axs``).
            axs: Existing axes (single, sequence, or 2D array) to wrap.
            subplot_kw: Forwarded to ``plt.subplots``.
            gridspec_kw: GridSpec spacing parameters (``wspace``, ``hspace``,
                ``width_ratios``, ``height_ratios``, etc.).
            **fig_kw: Forwarded to ``plt.figure``. Also accepts the shortcuts:

                * ``sharex`` / ``sharey``: Share the x or y axis across subplots.
                * ``projection``: Subplot projection (None → ``'rectilinear'``).
                * ``layout``: One of ``'constrained'``, ``'compressed'``, ``'tight'``,
                  ``'none'``, a ``LayoutEngine`` instance, or None.

        Example:
            >>> import numpy as np
            >>> from liron_utils import graphics as gr
            >>>
            >>> axes = gr.Axes(shape=(2, 3))
            >>> t = np.linspace(0, 10, 1001)
            >>> axes.plot(t, np.sin(t))
            >>> axes.set_props(sup_title="abc", show_fig=True)
        """
        self.fig: Figure | None = fig
        self.axs: _Array = np.atleast_2d(typing.cast(typing.Any, axs))

        if fig is None and axs is None:
            sharex = fig_kw.pop("sharex", False)
            sharey = fig_kw.pop("sharey", False)
            projection = fig_kw.pop("projection", None)
            layout = fig_kw.pop("layout", None)
            self._build_new_figure(
                _FigureBuildConfig(
                    shape=shape,
                    grid_layout=grid_layout,
                    share=(sharex, sharey),
                    subplot_kw={"projection": projection} | (subplot_kw or {}),
                    gridspec_kw=gridspec_kw,
                    fig_kw={"layout": layout} | fig_kw,
                ),
            )
        elif fig is not None:
            self.axs = np.atleast_2d(typing.cast(typing.Any, fig.axes))
        elif axs is not None:
            self.fig = self.axs[0, 0].figure

        self.func_animation: matplotlib.animation.FuncAnimation | None = None

    def _build_new_figure(self, cfg: "_FigureBuildConfig") -> None:
        """Create ``self.fig`` and ``self.axs`` from a build config.

        Routes through ``plt.subplots`` for plain grids, or through
        :meth:`_build_custom_grid` for span layouts. Also applies layout padding.

        Args:
            cfg: Bundle of figure-build options.
        """
        cfg.fig_kw = merge_kwargs(fig_kw=cfg.fig_kw)["fig_kw"]
        padding = _extract_layout_padding(cfg.fig_kw)

        if cfg.grid_layout is None:
            nrows, ncols = cfg.shape
            sharex, sharey = cfg.share
            self.fig, self.axs = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                sharex=typing.cast(typing.Any, sharex),
                sharey=typing.cast(typing.Any, sharey),
                squeeze=False,
                subplot_kw=cfg.subplot_kw,
                gridspec_kw=cfg.gridspec_kw,
                **cfg.fig_kw,
            )
        else:
            self._build_custom_grid(cfg)

        assert self.fig is not None
        _apply_constrained_layout_padding(self.fig, padding)

    def _build_custom_grid(self, cfg: "_FigureBuildConfig") -> None:
        """Build a figure with a custom span gridspec from ``cfg.grid_layout``.

        Args:
            cfg: Build config; ``cfg.grid_layout`` must be non-None.
        """
        assert cfg.grid_layout is not None
        assert cfg.subplot_kw is not None
        gridspec_kw = cfg.gridspec_kw if cfg.gridspec_kw is not None else {}
        share_bool = (_normalize_share_flag(cfg.share[0]), _normalize_share_flag(cfg.share[1]))

        nrows, ncols = cfg.shape
        self.fig = plt.figure(**cfg.fig_kw)
        gs = self.fig.add_gridspec(nrows=nrows, ncols=ncols, **gridspec_kw)

        axs_arr: _Array = np.ones(shape=cfg.shape, dtype=Axes_plt)
        ax_share = self._add_span_subplots(axs_arr, gs, cfg, share_bool=share_bool)
        self._fill_remaining_subplots(axs_arr, gs, cfg.subplot_kw, ax_share=ax_share, share_bool=share_bool)

        self.axs = axs_arr

    def _add_span_subplots(
        self,
        axs_arr: _Array,
        gs: typing.Any,
        cfg: "_FigureBuildConfig",
        share_bool: tuple[bool, bool],
    ) -> Axes_plt | None:
        """Create the span subplots defined by ``cfg.grid_layout``.

        Args:
            axs_arr: Output 2D array; cells covered by a span are set to 0 and the
                top-left cell of each span receives the new ``Axes_plt``. Mutated in place.
            gs: GridSpec instance from ``self.fig.add_gridspec``.
            cfg: Build config (uses ``grid_layout`` and ``subplot_kw``).
            share_bool: Normalized ``(sharex, sharey)`` booleans.

        Returns:
            The first created axis (used as the share-target for the rest), or None
            if ``cfg.grid_layout`` is empty.
        """
        assert self.fig is not None
        assert cfg.grid_layout is not None
        sharex_bool, sharey_bool = share_bool
        ax_share: Axes_plt | None = None
        for row_range, col_range in cfg.grid_layout:
            r0, r1 = _as_range(row_range)
            c0, c1 = _as_range(col_range)

            axs_arr[r0:r1, c0:c1] = 0
            axs_arr[r0, c0] = self.fig.add_subplot(
                gs[r0:r1, c0:c1],
                sharex=ax_share if sharex_bool else None,
                sharey=ax_share if sharey_bool else None,
                **cfg.subplot_kw,
            )

            if ax_share is None:
                ax_share = axs_arr[r0, c0]
        return ax_share

    def _fill_remaining_subplots(
        self,
        axs_arr: _Array,
        gs: typing.Any,
        subplot_kw: dict[str, typing.Any],
        *,
        ax_share: Axes_plt | None,
        share_bool: tuple[bool, bool],
    ) -> None:
        """Fill the gridspec cells not covered by any span with single-cell subplots.

        Args:
            axs_arr: Output 2D array; cells still equal to 1 receive a new ``Axes_plt``.
                Mutated in place.
            gs: GridSpec instance from ``self.fig.add_gridspec``.
            subplot_kw: Forwarded to ``Figure.add_subplot``.
            ax_share: First axis to share with (or None to skip sharing).
            share_bool: Normalized ``(sharex, sharey)`` booleans.
        """
        assert self.fig is not None
        sharex_bool, sharey_bool = share_bool
        for r, c in np.argwhere(axs_arr == 1):
            axs_arr[r, c] = self.fig.add_subplot(
                gs[r, c],
                sharex=ax_share if sharex_bool else None,
                sharey=ax_share if sharey_bool else None,
                **subplot_kw,
            )

    def __getitem__(self, item: typing.Any) -> "_Axes":
        """Return a deep-copied ``_Axes`` whose ``axs`` is sliced by ``item``.

        Args:
            item: Anything accepted by ``np.ndarray.__getitem__`` on ``self.axs``.

        Returns:
            A new ``_Axes`` sharing ``self.fig`` but holding the sliced axes array.
        """
        out = copy.deepcopy(self)
        out.axs = np.atleast_2d(self.axs[item])
        return out

    def _vectorize(
        self,
        ax: Axes_plt | None = None,
        **vec_params: typing.Any,
    ) -> Callable[[Callable[..., _R]], Callable[..., _Array]]:
        """Return a decorator that runs ``func`` either on one axis or across the grid.

        When ``ax`` is given, ``func`` is called once with that specific axis. Otherwise
        ``func`` is called once per cell of ``self.axs`` (skipping non-axes cells),
        with the per-axis slice of ``vec_params``; results are collected into a
        matching object-dtype ndarray.

        Args:
            ax: Optional single axis to bind ``func`` to.
            **vec_params: Per-call kwargs. List-valued entries are de-interleaved
                (via ``dl_to_ld``) to one mapping per axis; scalar entries are broadcast.

        Returns:
            A decorator wrapping ``func`` into a callable that returns an object-dtype
            ndarray of per-axis results.
        """

        def decorator(func: Callable[..., _R]) -> Callable[..., _Array]:
            @functools.wraps(func)
            def wrapper(*args: typing.Any, **kwargs: typing.Any) -> _Array:
                if ax is not None:
                    return typing.cast(_Array, func(ax, *args, **vec_params, **kwargs))

                m, n = self.axs.shape

                params_list = dl_to_ld(vec_params)
                if params_list is None or len(params_list) != m * n:
                    params_list = list(np.repeat(typing.cast(typing.Any, vec_params), m * n))

                out: _Array = np.empty((m, n), dtype=object)

                for i in range(m):
                    for j in range(n):
                        if isinstance(self.axs[i, j], Axes_plt):
                            out[i, j] = func(
                                self.axs[i, j],
                                *args,
                                **params_list[j * m + i],
                                **kwargs,
                            )

                return out

            return wrapper

        return decorator

    def _merge_kwargs(
        self,
        key: str,
        **kwargs: typing.Any,
    ) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
        """Return a decorator that overrides one kwarg with defaults merged from ``DefaultKwargs``.

        Args:
            key: Name of the kwarg in the wrapped function whose value is replaced
                with ``DefaultKwargs[key.upper()] | kwargs``.
            **kwargs: Caller-supplied overrides applied on top of the defaults.

        Returns:
            A decorator wrapping ``func`` so each call uses the merged kwargs.
        """

        def decorator(func: Callable[_P, _R]) -> Callable[_P, _R]:
            @functools.wraps(func)
            def wrapper(*args: _P.args, **outer_kwargs: _P.kwargs) -> _R:  # pylint: disable=unused-argument
                kwargs_merged = merge_kwargs(**{key: kwargs})[key]
                return func(*args, **kwargs_merged)

            return wrapper

        return decorator

    def draw_xy_lines(self, **xy_lines_kw: typing.Any) -> None:
        """Draw bold ``x=0`` and ``y=0`` lines on each axis to highlight the origin.

        3D plots and axes containing images are skipped. Each axis is annotated
        with ``axis_lines_drawn = True`` to make the operation idempotent.

        Args:
            **xy_lines_kw: Forwarded to ``ax.axhline`` and ``ax.axvline``.
        """

        @self._merge_kwargs("xy_lines_kw", **xy_lines_kw)
        @self._vectorize()
        def _draw_xy_lines(ax: Axes_plt, **kw: typing.Any) -> None:
            # Don't draw axis lines for 3D plots and images
            if hasattr(ax, "axis_lines_drawn") or hasattr(ax, "zaxis") or len(ax.images) > 0:
                return

            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            ax.axhline(**kw)
            ax.axvline(**kw)

            ax.set_xlim(*xlim, auto=True)
            ax.set_ylim(*ylim, auto=True)

            ax.axis_lines_drawn = True  # type: ignore[attr-defined]

        _draw_xy_lines()

    def sup_title(self, title: str) -> None:
        """Set the figure-wide super-title.

        Args:
            title: Super-title text.
        """
        # todo: for unknown reason, color is not automatically inherited from text.color
        assert self.fig is not None
        self.fig.suptitle(title, color=plt.rcParams["text.color"])

    def ax_axis(self, axis: bool | str) -> None:
        """Set the axis visibility / scaling mode on each axis.

        Args:
            axis: Forwarded to ``ax.axis``. Common values include ``"on"``/``"off"``,
                ``"equal"``, ``"scaled"``, ``"tight"``, ``"auto"``, ``"image"``,
                ``"square"``, or a bool.
        """

        @self._vectorize(axis=axis)
        def _ax_axis(ax: Axes_plt, axis: bool | str) -> None:
            ax.axis(axis)

        _ax_axis()

    def ax_spines(self, spines: str | list[str] | bool) -> None:
        """Show or hide axis spines (boundaries) on each axis.

        Args:
            spines: ``True`` shows all four spines, ``False`` hides them, a string
                or list of strings selects specific ones from
                ``["left", "bottom", "top", "right"]``.

        Raises:
            ValueError: If ``spines`` is not one of (str, list, bool).
        """

        @self._vectorize(spines=spines)
        def _ax_spines(ax: Axes_plt, spines: str | list[str] | bool) -> None:
            locs = np.array(["left", "bottom", "top", "right"])

            if isinstance(spines, str):
                spines = [spines]

            if isinstance(spines, Iterable):
                locs = np.array(spines)
                idx: list[bool] = [True] * locs.size
            elif isinstance(spines, bool):
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

    def ax_ticks(
        self,
        ticks: bool | list[typing.Any] | dict[typing.Any, typing.Any] | list[dict[typing.Any, typing.Any]] | None,
        labels: bool | list[typing.Any] | None,
    ) -> None:
        """Set ticks (and labels) on each axis along x, y, and (for 3D) z.

        Args:
            ticks: ``True``/``None`` shows the default ticks, ``False`` hides them, a
                per-dim list selects per-axis values, a dict ``{position: label}``
                applies to the x-axis only.
            labels: ``True``/``None`` keeps current labels, ``False`` hides them, a
                per-dim list provides explicit labels. When ``ticks`` is a dict,
                ``labels`` must not be given.
        """

        @self._vectorize(ticks=ticks, labels=labels)
        def _ax_ticks(
            ax: Axes_plt,
            ticks: bool | list[typing.Any] | dict[typing.Any, typing.Any] | list[dict[typing.Any, typing.Any]] | None,
            labels: bool | list[typing.Any] | None,
        ) -> None:
            if len(ax.get_shared_x_axes().get_siblings(ax)) > 1 or len(ax.get_shared_y_axes().get_siblings(ax)) > 1:
                # todo: fix in future
                warnings.warn("Axis shares x or y axis with siblings. Changing ticks would affect all siblings.")

            ndim = 3 if hasattr(ax, "get_zticks") else 2
            ticks_norm, labels_norm = _normalize_ticks_labels(ticks, labels, ndim)

            assert len(ticks_norm) <= ndim, "len(ticks) must be <= graph dimensionality."
            assert len(labels_norm) <= ndim, "len(labels) must be <= graph dimensionality."

            d = _collect_current_ticks_labels(ax, ndim)
            _filter_ticks_within_limits(ax, d, ndim)
            _apply_tick_label_overrides(d, ticks_norm, labels_norm)
            _set_ticks_on_axes(ax, d, ndim)

        _ax_ticks()

    def ax_title(self, title: str) -> None:
        """Set the per-axes title on each axis.

        Args:
            title: Title text (vectorizable across the axes grid).
        """

        @self._vectorize(title=title)
        def _ax_title(ax: Axes_plt, title: str) -> None:
            ax.set_title(title)

        _ax_title()

    def ax_labels(self, labels: list[str]) -> None:
        """Set the x/y/z axis labels on each axis.

        Args:
            labels: 1-, 2-, or 3-element list — entries are applied to ``set_xlabel``,
                ``set_ylabel``, and ``set_zlabel`` in order.
        """

        @self._vectorize(labels=labels)
        def _ax_labels(ax: Axes_plt, labels: list[str]) -> None:
            labels_arr = np.atleast_1d(labels)
            ax.set_xlabel(labels_arr[0])
            if np.size(labels_arr) >= 2:
                ax.set_ylabel(labels_arr[1])
            if np.size(labels_arr) >= 3:
                ax.set_zlabel(labels_arr[2])  # type: ignore[attr-defined]

        _ax_labels()

    def ax_limits(self, limits: list[float]) -> None:
        """Set the x/y/z axis limits on each axis.

        Args:
            limits: List of ``[min, max]`` pairs, one per axis dimension. Example:
                ``[[0, 2], [-1, 1]]`` sets ``xlim=[0, 2]`` and ``ylim=[-1, 1]``.
        """

        @self._vectorize(limits=limits)
        def _ax_limits(ax: Axes_plt, limits: list[float]) -> None:
            limits_arr = np.atleast_2d(np.array(limits, dtype=object))
            ax.set_xlim(limits_arr[0])
            if len(limits_arr) >= 2:
                ax.set_ylim(limits_arr[1])
            if len(limits_arr) >= 3:
                ax.set_zlim(limits_arr[2])  # type: ignore[attr-defined]

        _ax_limits()

    def ax_view(self, view: list[float]) -> None:
        """Set the 3D view angle on each axis.

        Args:
            view: ``[elev, azim]`` pair forwarded to ``Axes3D.view_init``.
        """

        @self._vectorize(view=view)
        def _ax_view(ax: Axes_plt, view: list[float]) -> None:
            ax.view_init(view[0], view[1])  # type: ignore[attr-defined]

        _ax_view()

    def ax_grid(self, grid: bool) -> None:
        """Show or hide the grid on each axis.

        Args:
            grid: True shows the grid, False hides it; None toggles.
        """

        @self._vectorize(grid=grid)
        def _ax_grid(ax: Axes_plt, grid: bool) -> None:
            ax.grid(grid)

        _ax_grid()

    def ax_legend(self, legend: bool | list[str], legend_loc: str | None) -> None:
        """Show, hide, or set the legend on each axis.

        Args:
            legend: ``True`` reuses the labels already attached to artists; ``False``
                removes an existing legend; a list of strings sets explicit labels.
            legend_loc: Matplotlib legend location string (e.g. ``"best"``,
                ``"upper right"``) or ``None`` for the default.
        """

        @self._vectorize(legend=legend, legend_loc=legend_loc)
        def _ax_legend(ax: Axes_plt, legend: bool | list[str], legend_loc: str | None) -> None:
            if legend is False:
                existing = ax.get_legend()
                if existing is not None:
                    existing.remove()
            elif legend is True:
                handles, labels = ax.get_legend_handles_labels()
                if len(handles) > 0:
                    ax.legend(handles, labels, loc=legend_loc)
            else:
                ax.legend(legend, loc=legend_loc)

        _ax_legend()

    def ax_colorbar(
        self,
        axs: bool | Axes_plt | typing.Sequence[typing.Any] | _Array,
        **colorbar_kw: typing.Any,
    ) -> None:
        """Attach a colorbar to one or more axes that contain an image.

        Args:
            axs: ``False`` is a no-op; ``True`` finds every axis on the figure with
                an image and gives each its own colorbar; a single axis or array/list
                of axes adds one colorbar per entry; a list of lists creates a
                shared colorbar across each inner list.
            **colorbar_kw: Forwarded to ``Figure.colorbar``.

        Raises:
            AssertionError: If any provided axis has no image plotted.
        """
        if axs is False:
            return
        assert self.fig is not None
        axs_list: list[typing.Any]
        if axs is True:
            axs_list = [[ax] for ax in self.fig.axes if len(ax.images) > 0]
        elif isinstance(axs, Axes_plt):
            axs_list = [axs]
        elif isinstance(axs, np.ndarray):
            axs_list = typing.cast(list[typing.Any], axs.tolist())
        else:
            axs_list = list(axs)

        for i in range(len(axs_list)):  # pylint: disable=consider-using-enumerate
            if isinstance(axs_list[i], Axes_plt):
                axs_list[i] = [axs_list[i]]

            assert isinstance(axs_list[i], Iterable), "'axs' must be a list of lists."
            for j in range(len(axs_list[i])):
                assert len(axs_list[i][j].images) > 0, "At least one axis has no image plotted."

        for ax_common in axs_list:
            mappable = ax_common[0].images[0]
            self.fig.colorbar(mappable=mappable, ax=ax_common, **colorbar_kw)

    def ax_face_color(self, face_color: typing.Any) -> None:
        """Set the axes face color on each axis.

        Args:
            face_color: Any matplotlib color spec (hex string, RGB tuple, etc.).
        """

        @self._vectorize(face_color=face_color)
        def _ax_face_color(ax: Axes_plt, face_color: typing.Any) -> None:
            ax.set_facecolor(face_color)

        _ax_face_color()

    def save_fig(self, file_name: str | None = None, **savefig_kw: typing.Any) -> str:
        """Save the figure (or animation) to disk.

        ``.gif`` outputs require :meth:`plot_animation` to have been called first;
        any other extension is saved via ``Figure.savefig``.

        Args:
            file_name: Output path. None or a bare filename routes through
                :func:`get_savefig_file_name` to derive a default directory.
            **savefig_kw: Forwarded to ``Figure.savefig`` or ``FuncAnimation.save``.

        Returns:
            The full path the file was saved to.

        Raises:
            AssertionError: For ``.gif`` outputs when no animation has been created.
        """
        file_name = get_savefig_file_name(file_name, mkdir=True)

        ext = os.path.splitext(file_name)[-1]

        if ext == ".gif":
            assert isinstance(
                self.func_animation,
                matplotlib.animation.FuncAnimation,
            ), "Call plot_animation() before saving gif."
            self.func_animation.save(file_name, **savefig_kw)
        else:
            assert self.fig is not None
            self.fig.savefig(file_name, **savefig_kw)

        return file_name

    def show_fig(self) -> None:
        """Show the wrapped figure (delegates to ``Figure.show``)."""
        assert self.fig is not None
        self.fig.show()

    def show(self) -> None:
        """Alias for :meth:`show_fig`."""
        self.show_fig()

    def set_props(
        self,
        save_file_name: str | bool = False,
        colorbar_kw: dict[str, typing.Any] | None = None,
        xy_lines_kw: dict[str, typing.Any] | None = None,
        save_fig_kw: dict[str, typing.Any] | None = None,
        **set_props_kw: typing.Any,
    ) -> str | None:
        """Apply post-plot figure/axes properties in a single call.

        Each ``set_props_kw`` value is either a scalar (applied to all axes) or a
        list/array (one entry per axis). Parameters marked vectorizable below also
        accept per-axis lists.

        Args:
            save_file_name: ``False`` — don't save; ``True`` — save to ``MAIN_FILE_DIR``
                with an auto-generated name; ``str`` — explicit path.
            colorbar_kw: Forwarded to ``Figure.colorbar``.
            xy_lines_kw: Forwarded to ``ax.axhline`` / ``ax.axvline`` via :meth:`draw_xy_lines`.
            save_fig_kw: Forwarded to ``Figure.savefig`` via :meth:`save_fig`.
            **set_props_kw: Property overrides. Vectorizable entries are marked ``*``:

                * ``sup_title``: Figure suptitle.
                * ``*ax_title``: Per-axis title.
                * ``*axis``: Axis visibility / scaling mode.
                * ``*spines``: Axis spines visibility.
                * ``*ticks``: Axis ticks.
                * ``*tick_labels``: Axis tick labels.
                * ``*labels``: x/y/z labels.
                * ``*limits``: Axis limits ``[[xmin, xmax], [ymin, ymax], ...]``.
                * ``*view``: 3D view angles.
                * ``*grid``: Show grid.
                * ``*legend``: Show / set legend (bool or list of label strings).
                * ``*legend_loc``: Legend location string.
                * ``colorbar``: Add colorbar to image axes.
                * ``*xy_lines``: Draw ``x=0`` and ``y=0`` lines.
                * ``*face_color``: Axes face color.
                * ``show_fig``: Show figure when done.
                * ``open_dir``: Open the save directory after saving.
                * ``close_fig``: Close the figure after saving / showing.

        Returns:
            The saved file path when ``save_file_name`` is truthy, else None.
        """
        if colorbar_kw is None:
            colorbar_kw = {}
        if xy_lines_kw is None:
            xy_lines_kw = {}
        if save_fig_kw is None:
            save_fig_kw = {}

        kw = merge_kwargs(set_props_kw=set_props_kw)["set_props_kw"]

        def caller(func: Callable[..., typing.Any], *args: typing.Any, **kwargs: typing.Any) -> None:
            if args[0] is None:
                return
            func(*args, **kwargs)

        caller(self.sup_title, kw["sup_title"])
        caller(self.ax_title, kw["ax_title"])
        caller(self.ax_axis, kw["axis"])
        caller(self.ax_spines, kw["spines"])
        caller(self.ax_labels, kw["labels"])
        caller(self.ax_view, kw["view"])
        caller(self.ax_grid, kw["grid"])
        self.ax_legend(kw["legend"], kw["legend_loc"])
        self.ax_colorbar(kw["colorbar"], **colorbar_kw)
        caller(self.ax_limits, kw["limits"])
        caller(self.ax_ticks, kw["ticks"], kw["tick_labels"])

        if kw["xy_lines"]:
            self.draw_xy_lines(**xy_lines_kw)

        caller(self.ax_face_color, kw["face_color"])

        file_name: str | None = None
        if save_file_name is not False:
            if save_file_name is True:
                save_file_name = None  # type: ignore[assignment]
            file_name = self.save_fig(file_name=save_file_name, **save_fig_kw)  # type: ignore[arg-type]

        if kw["show_fig"]:
            self.show_fig()
            if kw["open_dir"] and file_name is not None:
                open_file(os.path.split(file_name)[0])

        if kw["close_fig"]:
            assert self.fig is not None
            plt.close(self.fig)

        return file_name


def new_figure(
    nrows: int = 1,
    ncols: int = 1,
    sharex: bool = False,
    sharey: bool = False,
    projection: str | None = None,
    *,
    squeeze: bool = True,
    subplot_kw: dict[str, typing.Any] | None = None,
    gridspec_kw: dict[str, typing.Any] | None = None,
    **figure_kw: typing.Any,
) -> tuple[Figure, Axes_plt]:
    """Thin wrapper around ``plt.subplots`` that injects ``projection`` into ``subplot_kw``.

    Args:
        nrows: Number of subplot rows.
        ncols: Number of subplot columns.
        sharex: Share the x-axis across subplots.
        sharey: Share the y-axis across subplots.
        projection: Default subplot projection (``None`` → ``'rectilinear'``).
        squeeze: Forwarded to ``plt.subplots``.
        subplot_kw: Forwarded to ``plt.subplots``.
        gridspec_kw: Forwarded to ``plt.subplots``.
        **figure_kw: Forwarded to ``plt.subplots``.

    Returns:
        ``(fig, axs)`` from ``plt.subplots``.
    """
    if subplot_kw is None:
        subplot_kw = {}
    subplot_kw = {"projection": projection} | subplot_kw

    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        sharex=sharex,
        sharey=sharey,
        squeeze=squeeze,
        subplot_kw=subplot_kw,
        gridspec_kw=gridspec_kw,
        **figure_kw,
    )

    return fig, axs


def draw_xy_lines(ax: Axes_plt, **axis_lines_kw: typing.Any) -> None:
    """Standalone shim around :meth:`_Axes.draw_xy_lines`.

    Args:
        ax: Target axes.
        **axis_lines_kw: Forwarded to ``_Axes.draw_xy_lines``.
    """
    _Axes(axs=ax).draw_xy_lines(**axis_lines_kw)


def save_fig(fig: Figure | None = None, file_name: str | None = None, **savefig_kw: typing.Any) -> None:
    """Standalone shim around :meth:`_Axes.save_fig`.

    Args:
        fig: Target figure (defaults to ``plt.gcf()``).
        file_name: Output path; see :meth:`_Axes.save_fig`.
        **savefig_kw: Forwarded to ``_Axes.save_fig``.
    """
    if fig is None:
        fig = plt.gcf()
    _Axes(fig=fig).save_fig(file_name, **savefig_kw)


def set_props(
    ax: Axes_plt | None = None,
    save_file_name: str | bool = False,
    save_fig_kw: dict[str, typing.Any] | None = None,
    **set_props_kw: typing.Any,
) -> None:
    """Standalone shim around :meth:`_Axes.set_props`.

    Args:
        ax: Target axes (defaults to ``plt.gca()``).
        save_file_name: See :meth:`_Axes.set_props`.
        save_fig_kw: Forwarded to ``_Axes.set_props``.
        **set_props_kw: Forwarded to ``_Axes.set_props``.
    """
    if ax is None:
        ax = plt.gca()
    _Axes(axs=ax).set_props(save_file_name=save_file_name, save_fig_kw=save_fig_kw, **set_props_kw)


def get_savefig_file_name(
    file_name: str | None = None,
    time_dir: bool = False,
    mkdir: bool = False,
) -> str:
    """Build the full path for saving a figure.

    Args:
        file_name: If it has no directory component, the path resolves under
            ``<MAIN_FILE_DIR>/figs``. If None, an auto-generated ``"fig <ts>"`` name
            is used.
        time_dir: If True, append a timestamped subdirectory.
        mkdir: Create the directory if it doesn't exist.

    Returns:
        Resolved absolute (or near-absolute) file path.
    """
    if file_name is None or os.path.dirname(file_name) == "":
        dir_name = os.path.join(MAIN_FILE_DIR, "figs")
    else:
        dir_name = os.path.dirname(file_name)

    if time_dir:
        dir_name = os.path.join(dir_name, TIME_STR)

    if mkdir and not os.path.exists(dir_name):
        mkdirs(dir_name)

    if file_name is None:
        file_name = os.path.join(dir_name, f"fig {get_time_str()}")
    elif os.path.dirname(file_name) == "":
        file_name = os.path.join(dir_name, file_name)

    return file_name
