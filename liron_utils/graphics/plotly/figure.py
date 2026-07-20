import os
import typing
from collections.abc import Callable, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ...signal_processing.base import interp1
from ...uncertainties_math import to_numpy
from ..common import COLORS
from ..common.files import get_savefig_file_name
from ..common.fitting import curve_fit_confidence_band, curve_fit_prep_data
from ..common.spectra import (
    fft_data,
    frequency_response_data,
    impulse_response_data,
    periodogram_data,
    si_prefix_scale,
    spectrogram_data,
    spectrum_display_data,
)
from .templates import COLORWAY

_SpanRange = int | tuple[int, int]
_SpecRow = list[dict[str, typing.Any] | None]
_Array1D = np.ndarray[tuple[int], np.dtype[typing.Any]]


def _as_range(value: _SpanRange) -> tuple[int, int]:
    """Promote a single int row/col index to a 0-based half-open ``(start, end)`` range."""
    if isinstance(value, int):
        return value, value + 1
    return value


def _span_layout_to_specs(
    shape: tuple[int, int],
    span_layout: Sequence[Sequence[_SpanRange]],
    subplot_type: str,
) -> list[_SpecRow]:
    """Convert an mpl-style span layout to a ``make_subplots`` specs grid.

    Args:
        shape: ``(nrows, ncols)``.
        span_layout: List of ``[row_range, col_range]`` span specs; each range is an
            int ``i`` (interpreted as ``(i, i + 1)``) or a 0-based half-open
            ``(start, end)`` tuple.
        subplot_type: Subplot type applied to every cell (``"xy"``, ``"scene"``, ...).

    Returns:
        Specs grid for ``make_subplots``: span-covered cells are None, span anchors
        carry ``rowspan``/``colspan``, remaining cells are plain single subplots.
    """
    rows, cols = shape
    specs: list[_SpecRow] = [[{"type": subplot_type} for _ in range(cols)] for _ in range(rows)]
    for row_range, col_range in span_layout:
        r0, r1 = _as_range(row_range)
        c0, c1 = _as_range(col_range)
        for r in range(r0, r1):
            for c in range(c0, c1):
                specs[r][c] = None
        specs[r0][c0] = {"type": subplot_type, "rowspan": r1 - r0, "colspan": c1 - c0}
    return specs


class Figure(go.Figure):
    """A ``plotly.graph_objects.Figure`` with a subplot grid and domain plot helpers.

    All native plotly methods work directly (``add_scatter(row=, col=)``,
    ``update_layout``, ``show``, ...). Rows/cols in method calls are plotly's
    native 1-based indices; ``span_layout`` uses 0-based half-open ranges
    (mirroring ``graphics.mpl``'s ``grid_layout``).

    Example:
        >>> from liron_utils.graphics.plotly import Figure
        >>>
        >>> fig = Figure(shape=(2, 1), shared_xaxes=True)
        >>> fig.add_scatter(x=[0, 1], y=[1, 0], row=2, col=1)
        >>> fig.update_layout(title="Example")
        >>> fig.show()
    """

    def __init__(
        self,
        shape: tuple[int, int] = (1, 1),
        *,
        span_layout: Sequence[Sequence[_SpanRange]] | None = None,
        subplot_type: str = "xy",
        shared_xaxes: bool = False,
        shared_yaxes: bool = False,
        specs: list[_SpecRow] | None = None,
        **make_subplots_kw: typing.Any,
    ) -> None:
        """Create a figure holding an ``nrows × ncols`` subplot grid.

        Args:
            shape: ``(nrows, ncols)`` for the subplot grid.
            span_layout: Optional span specs (see :func:`_span_layout_to_specs`).
                Example: ``[[(0, 2), (0, 2)], [0, (2, 3)]]``.
            subplot_type: Type for all cells (``"xy"``, ``"scene"``, ``"polar"``, ...).
                Ignored for cells covered by an explicit ``specs``.
            shared_xaxes: Share x-axes across subplots (forwarded to ``make_subplots``).
            shared_yaxes: Share y-axes across subplots (forwarded to ``make_subplots``).
            specs: Explicit ``make_subplots`` specs grid; overrides ``span_layout``.
            **make_subplots_kw: Forwarded to ``plotly.subplots.make_subplots``
                (``subplot_titles``, ``row_heights``, ``column_widths``,
                ``horizontal_spacing``, ...).
        """
        rows, cols = shape
        if specs is None:
            if span_layout is not None:
                specs = _span_layout_to_specs(shape, span_layout, subplot_type)
            else:
                specs = [[{"type": subplot_type} for _ in range(cols)] for _ in range(rows)]
        base = make_subplots(
            rows=rows,
            cols=cols,
            specs=specs,
            shared_xaxes=shared_xaxes,
            shared_yaxes=shared_yaxes,
            **make_subplots_kw,
        )
        super().__init__(base)
        # make_subplots stores the grid on the instance it returns; without copying it,
        # row=/col= addressing on this subclass instance fails.
        self._grid_ref = base._grid_ref
        self._grid_str = base._grid_str

    def save(self, file_name: str | None = None, **write_kw: typing.Any) -> str:
        """Save the figure to disk, dispatching on the file extension.

        ``.html`` (or no extension) saves an interactive page via ``write_html``;
        any other extension (``.png``, ``.pdf``, ``.svg``, ...) is rendered via
        ``write_image`` (requires kaleido).

        Args:
            file_name: Output path. None or a bare filename resolves under
                ``<MAIN_FILE_DIR>/figs`` with an auto-generated name.
            **write_kw: Forwarded to ``write_html`` / ``write_image``.

        Returns:
            The full path the file was saved to.
        """
        file_name = get_savefig_file_name(file_name, mkdir=True)
        ext = os.path.splitext(file_name)[-1].lower()
        if ext == "":
            file_name += ".html"
            ext = ".html"
        if ext == ".html":
            self.write_html(file_name, **write_kw)
        else:
            self.write_image(file_name, **write_kw)
        return file_name

    def plot(
        self,
        x: _Array1D,
        y: _Array1D | None = None,
        z: _Array1D | None = None,
        *,
        row: int = 1,
        col: int = 1,
        **scatter_kw: typing.Any,
    ) -> "Figure":
        """Add a 2D line trace (or 3D when ``z`` is given).

        Args:
            x: 1D x data. If ``y`` is None, ``x`` is treated as y and the x-axis
                becomes ``range(len(x))``.
            y: 1D y data.
            z: 1D z data; when given, a ``Scatter3d`` is added (requires a
                ``"scene"`` subplot).
            row: Target subplot row (1-based).
            col: Target subplot column (1-based).
            **scatter_kw: Forwarded to ``go.Scatter`` / ``go.Scatter3d``.

        Returns:
            self (chainable).
        """
        if y is None:
            x, y = typing.cast(_Array1D, np.arange(len(x))), x
        scatter_kw = {"mode": "lines"} | scatter_kw
        trace: typing.Any
        if z is not None:
            trace = go.Scatter3d(x=x, y=y, z=z, **scatter_kw)
        else:
            trace = go.Scatter(x=x, y=y, **scatter_kw)
        self.add_trace(trace, row=row, col=col)
        return self

    def plot_surf(
        self,
        x: _Array1D | np.ndarray[tuple[int, int], np.dtype[typing.Any]],
        y: _Array1D | np.ndarray[tuple[int, int], np.dtype[typing.Any]],
        z: (
            np.ndarray[tuple[int, int], np.dtype[typing.Any]]
            | Callable[[typing.Any, typing.Any], np.ndarray[tuple[int, int], np.dtype[typing.Any]]]
        ),
        *,
        row: int = 1,
        col: int = 1,
        **surface_kw: typing.Any,
    ) -> "Figure":
        """Plot a 3D surface ``z = f(x, y)`` (requires a ``"scene"`` subplot).

        Args:
            x: x-coordinates — 1D of length M (meshgrid applied) or 2D meshgrid.
            y: y-coordinates — 1D of length N (meshgrid applied) or 2D meshgrid.
            z: 2D z-values, or a callable ``f(x_grid, y_grid) -> z_grid``.
            row: Target subplot row (1-based).
            col: Target subplot column (1-based).
            **surface_kw: Forwarded to ``go.Surface``.

        Returns:
            self (chainable).
        """
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        if x_arr.ndim == 1:
            x_grid, y_grid = np.meshgrid(x_arr, y_arr)
        else:
            x_grid, y_grid = x_arr, y_arr
        z_grid = np.asarray(z(x_grid, y_grid)) if callable(z) else np.asarray(z)
        if z_grid.shape != x_grid.shape and z_grid.shape == tuple(reversed(x_grid.shape)):
            z_grid = z_grid.T

        self.add_trace(go.Surface(x=x_grid, y=y_grid, z=z_grid, **surface_kw), row=row, col=col)
        return self

    def plot_contour(  # pylint: disable=too-many-locals
        self,
        x: _Array1D | np.ndarray[tuple[int, int], np.dtype[typing.Any]],
        y: _Array1D | np.ndarray[tuple[int, int], np.dtype[typing.Any]],
        z: np.ndarray[tuple[int, int], np.dtype[typing.Any]],
        contours: int | tuple[float, float, float] | None = None,
        *,
        row: int = 1,
        col: int = 1,
        **contour_kw: typing.Any,
    ) -> "Figure":
        """Plot labeled contour lines for the scalar field ``z = f(x, y)``.

        Args:
            x: x-coordinates — 1D of length M, or a 2D meshgrid (its first row is used).
            y: y-coordinates — 1D of length N, or a 2D meshgrid (its first column is used).
            z: 2D z-values of shape (N, M).
            contours: Number of levels (int), or explicit uniform levels as
                ``(start, end, size)``. None lets plotly pick.
            row: Target subplot row (1-based).
            col: Target subplot column (1-based).
            **contour_kw: Forwarded to ``go.Contour``.

        Returns:
            self (chainable).
        """
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        x_1d = x_arr[0, :] if x_arr.ndim == 2 else x_arr
        y_1d = y_arr[:, 0] if y_arr.ndim == 2 else y_arr

        contours_spec: dict[str, typing.Any] = {"showlabels": True}
        if isinstance(contours, int):
            # ncontours is honored while autocontour (default True) is on and contours.size is unset.
            contour_kw = {"ncontours": contours} | contour_kw
        elif contours is not None:
            start, end, size = contours
            contours_spec |= {"start": start, "end": end, "size": size}

        self.add_trace(
            go.Contour(x=x_1d, y=y_1d, z=z, contours=contours_spec, **contour_kw),
            row=row,
            col=col,
        )
        return self

    def plot_vlines(
        self,
        x: _Array1D | float = 0,
        *,
        label: str | None = None,
        row: int | str | None = None,
        col: int | str | None = None,
        **vline_kw: typing.Any,
    ) -> "Figure":
        """Add one or more vertical lines (as shapes) spanning the y-axis.

        Args:
            x: Line positions — a scalar or a 1D array.
            label: Legend label, attached to the last line only.
            row: Target subplot row; None applies to all subplots.
            col: Target subplot column; None applies to all subplots.
            **vline_kw: Forwarded to ``add_vline`` (e.g. ``line_color``, ``line_dash``).

        Returns:
            self (chainable).
        """
        x_arr = typing.cast(_Array1D, np.atleast_1d(x))
        for i, xx in enumerate(x_arr):
            last = i == x_arr.shape[0] - 1
            self.add_vline(
                x=float(xx),
                row=row,
                col=col,
                name=label if last else None,
                showlegend=label is not None and last,
                **vline_kw,
            )
        return self

    def plot_hlines(
        self,
        y: _Array1D | float = 0,
        *,
        label: str | None = None,
        row: int | str | None = None,
        col: int | str | None = None,
        **hline_kw: typing.Any,
    ) -> "Figure":
        """Add one or more horizontal lines (as shapes) spanning the x-axis.

        Args:
            y: Line positions — a scalar or a 1D array.
            label: Legend label, attached to the last line only.
            row: Target subplot row; None applies to all subplots.
            col: Target subplot column; None applies to all subplots.
            **hline_kw: Forwarded to ``add_hline``.

        Returns:
            self (chainable).
        """
        y_arr = typing.cast(_Array1D, np.atleast_1d(y))
        for i, yy in enumerate(y_arr):
            last = i == y_arr.shape[0] - 1
            self.add_hline(
                y=float(yy),
                row=row,
                col=col,
                name=label if last else None,
                showlegend=label is not None and last,
                **hline_kw,
            )
        return self

    def draw_xy_lines(
        self,
        *,
        row: int | str | None = None,
        col: int | str | None = None,
        **line_kw: typing.Any,
    ) -> "Figure":
        """Draw bold ``x=0`` and ``y=0`` lines to highlight the origin.

        Args:
            row: Target subplot row; None applies to all subplots.
            col: Target subplot column; None applies to all subplots.
            **line_kw: Forwarded to ``add_hline`` / ``add_vline``.

        Returns:
            self (chainable).
        """
        line_kw = {"line_color": COLORS.DARK_GREY, "line_width": 2} | line_kw
        self.add_hline(y=0, row=row, col=col, **line_kw)
        self.add_vline(x=0, row=row, col=col, **line_kw)
        return self

    def plot_errorbar(
        self,
        x: _Array1D,
        y: _Array1D | None = None,
        xerr: _Array1D | None = None,
        yerr: _Array1D | None = None,
        *,
        row: int = 1,
        col: int = 1,
        **scatter_kw: typing.Any,
    ) -> "Figure":
        """Add a marker trace with error bars.

        If ``y`` is None, ``x`` is treated as the y data and the x-axis becomes
        ``range(len(x))``. ``x``/``y`` may be uncertainties arrays, in which case
        the errors are taken from the embedded uncertainties.

        Args:
            x: 1D x data (or y data when ``y`` is None).
            y: 1D y data.
            xerr: 1D errors in x. Ignored if ``x`` is an uncertainties array.
            yerr: 1D errors in y. Ignored if ``y`` is an uncertainties array.
            row: Target subplot row (1-based).
            col: Target subplot column (1-based).
            **scatter_kw: Forwarded to ``go.Scatter``.

        Returns:
            self (chainable).
        """
        if y is None:
            assert xerr is None and yerr is None, "If y is not given, xerr and yerr should not be given."
            x, y = typing.cast(_Array1D, np.arange(len(x))), x

        x, xerr = typing.cast(tuple[_Array1D, _Array1D | None], to_numpy(x, xerr))
        y, yerr = typing.cast(tuple[_Array1D, _Array1D | None], to_numpy(y, yerr))

        scatter_kw = {"mode": "markers", "marker": {"size": 8}} | scatter_kw
        error_bar_style = {"type": "data", "color": COLORS.RED_E, "thickness": 1.4}
        trace = go.Scatter(
            x=x,
            y=y,
            error_x=error_bar_style | {"array": xerr} if xerr is not None else None,
            error_y=error_bar_style | {"array": yerr} if yerr is not None else None,
            **scatter_kw,
        )
        self.add_trace(trace, row=row, col=col)
        return self

    def plot_filled_error(
        self,
        x: _Array1D,
        y: _Array1D | None = None,
        *,
        yerr: _Array1D | None = None,
        n_std: float = 2,
        y_low: _Array1D | None = None,
        y_high: _Array1D | None = None,
        row: int = 1,
        col: int = 1,
        **scatter_kw: typing.Any,
    ) -> "Figure":
        """Add a y±n_std·yerr confidence band as a filled area (two traces).

        Either ``(y, yerr)`` or ``(y_low, y_high)`` must be given.

        Args:
            x: 1D x data.
            y: 1D center values; required when ``y_low``/``y_high`` are None.
            yerr: 1D errors in y; the band is ``y ± n_std·yerr``.
            n_std: Number of standard deviations spanned by the band.
            y_low: 1D lower bound; required when ``y`` is None.
            y_high: 1D upper bound; required when ``y`` is None.
            row: Target subplot row (1-based).
            col: Target subplot column (1-based).
            **scatter_kw: Forwarded to the upper (filled) ``go.Scatter``.

        Returns:
            self (chainable).

        Raises:
            AssertionError: If neither ``(y, yerr)`` nor ``(y_low, y_high)`` is provided.
        """
        x = typing.cast(_Array1D, to_numpy(x)[0])
        if y is None:
            assert y_low is not None and y_high is not None, "(y, yerr) or (y_low, y_high) should be given."
        else:
            assert y_low is None and y_high is None, "(y, yerr) or (y_low, y_high) should be given."
            y, yerr = typing.cast(tuple[_Array1D, _Array1D | None], to_numpy(y, yerr))
            assert yerr is not None, "yerr should be given."
            y_low = typing.cast(_Array1D, y - n_std * yerr)
            y_high = typing.cast(_Array1D, y + n_std * yerr)

        fillcolor = scatter_kw.pop("fillcolor", "rgba(187, 187, 187, 0.4)")  # LIGHT_GRAY at 0.4 alpha
        band_style = {"mode": "lines", "line": {"width": 0}, "hoverinfo": "skip"}
        self.add_trace(go.Scatter(x=x, y=y_low, showlegend=False, **band_style), row=row, col=col)
        self.add_trace(
            go.Scatter(
                x=x,
                y=y_high,
                fill="tonexty",
                fillcolor=fillcolor,
                showlegend=scatter_kw.pop("showlegend", False),
                **band_style,
                **scatter_kw,
            ),
            row=row,
            col=col,
        )
        return self

    def plot_data_and_curve_fit(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        x: _Array1D,
        y: _Array1D,
        fit_fcn: Callable[..., _Array1D],
        *,
        xerr: _Array1D | None = None,
        yerr: _Array1D | None = None,
        p_opt: _Array1D | None = None,
        p_cov: np.ndarray[tuple[int, int], np.dtype[typing.Any]] | None = None,
        n_std: float = 2,
        interp_factor: int = 20,
        curve_fit_scatter_kw: dict[str, typing.Any] | None = None,
        row: int = 1,
        col: int = 1,
        **errorbar_kw: typing.Any,
    ) -> "Figure":
        """Scatter the data with errorbars and overlay a smooth curve fit with a confidence band.

        The mid curve uses ``fit_fcn(x, *p_opt)`` on a denser x-axis (``interp_factor`` ×
        original sample count). When both ``p_opt`` and ``p_cov`` are given, a filled
        ±n_std·σ confidence band is added.

        Args:
            x: 1D x data (uncertainties arrays supported).
            y: 1D y data (uncertainties arrays supported).
            fit_fcn: Model function ``f(x, *params)``.
            xerr: 1D errors in x.
            yerr: 1D errors in y.
            p_opt: 1D best-fit parameters (e.g. from ``scipy.optimize.curve_fit``).
            p_cov: Covariance of the fit parameters.
            n_std: Standard deviations spanned by the confidence band.
            interp_factor: x-axis upsampling factor for the fit curve.
            curve_fit_scatter_kw: Forwarded to the fit-line ``plot`` call.
            row: Target subplot row (1-based).
            col: Target subplot column (1-based).
            **errorbar_kw: Forwarded to ``plot_errorbar`` for the data scatter.

        Returns:
            self (chainable).
        """
        errorbar_kw = {"name": "Data"} | errorbar_kw
        curve_fit_scatter_kw = {"name": "Curve fit"} | (curve_fit_scatter_kw or {})

        x, y, xerr, yerr, p_opt = curve_fit_prep_data(x, y, xerr, yerr, p_opt)
        self.plot_errorbar(x, y, xerr=xerr, yerr=yerr, row=row, col=col, **errorbar_kw)

        if p_opt is not None:
            x_interp = typing.cast(_Array1D, interp1(x, interp_factor * len(x)))
            self.plot(x_interp, fit_fcn(x_interp, *p_opt), row=row, col=col, **curve_fit_scatter_kw)
            if p_cov is not None:
                fit_low, fit_high = curve_fit_confidence_band(fit_fcn, x_interp, p_opt, p_cov, n_std)
                self.plot_filled_error(x_interp, y_low=fit_low, y_high=fit_high, row=row, col=col)
        return self

    def plot_data_and_lin_reg(
        self,
        x: _Array1D,
        y: _Array1D,
        reg: typing.Any = None,
        *,
        xerr: _Array1D | None = None,
        yerr: _Array1D | None = None,
        reg_scatter_kw: dict[str, typing.Any] | None = None,
        row: int = 1,
        col: int = 1,
        **errorbar_kw: typing.Any,
    ) -> "Figure":
        """Scatter the data with errorbars and overlay a linear regression line.

        Args:
            x: 1D x data (uncertainties arrays supported).
            y: 1D y data (uncertainties arrays supported).
            reg: Output of ``scipy.stats.linregress`` (line + slope/stderr/R² label).
            xerr: 1D errors in x.
            yerr: 1D errors in y.
            reg_scatter_kw: Forwarded to the regression-line ``plot`` call.
            row: Target subplot row (1-based).
            col: Target subplot column (1-based).
            **errorbar_kw: Forwarded to ``plot_errorbar`` for the data scatter.

        Returns:
            self (chainable).
        """
        errorbar_kw = {"name": "Data"} | errorbar_kw

        x, xerr = typing.cast(tuple[_Array1D, _Array1D | None], to_numpy(x, xerr))
        y, yerr = typing.cast(tuple[_Array1D, _Array1D | None], to_numpy(y, yerr))
        self.plot_errorbar(x, y, xerr=xerr, yerr=yerr, row=row, col=col, **errorbar_kw)

        if reg is not None:
            reg_scatter_kw = {
                "name": f"{errorbar_kw['name']} linreg: slope={reg.slope:.3f}±{reg.stderr:.3f}, "
                f"R²={reg.rvalue ** 2:.3f}",
            } | (reg_scatter_kw or {})
            y_reg = typing.cast(_Array1D, reg.slope * x + reg.intercept)
            self.plot(x, y_reg, row=row, col=col, **reg_scatter_kw)
        return self

    def _set_spectrum_axis_titles(self, fs: float | None, ylabel: str, *, row: int, col: int) -> None:
        """Set frequency/quantity axis titles on one subplot."""
        xlabel = "Frequency [normalized]" if fs == 1.0 else "Frequency [Hz]"
        self.update_xaxes(title_text=xlabel, row=row, col=col)
        self.update_yaxes(title_text=ylabel, row=row, col=col)

    def plot_fft(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        x: _Array1D,
        fs: float = 1.0,
        n: int | None = None,
        *,
        one_sided: bool = True,
        normalize: bool = False,
        input_time: bool = True,
        which: str = "power",
        db: bool = False,
        eps: float = 1e-20,
        row: int = 1,
        col: int = 1,
        **scatter_kw: typing.Any,
    ) -> tuple[_Array1D, _Array1D]:
        """Plot the magnitude/power/phase spectrum of an FFT.

        Args:
            x: 1D input signal. Complex inputs always yield two-sided plots.
            fs: Sampling frequency in Hz. ``fs == 1.0`` produces a normalized axis label.
            n: FFT length. If None, uses ``len(x)``.
            one_sided: If True, plot only positive frequencies (auto-disabled for complex x).
            normalize: If True, scale the spectrum so its peak is 1.
            input_time: If True, treat ``x`` as time-domain and apply FFT.
            which: One of ``"amp"``, ``"power"``, ``"phase"``.
            db: If True (and ``which`` is amp/power), use a dB y-axis.
            eps: Floor added before ``log10`` to avoid log(0).
            row: Target subplot row (1-based).
            col: Target subplot column (1-based).
            **scatter_kw: Forwarded to the line trace.

        Returns:
            ``(spectrum, freqs)``.
        """
        spectrum, freqs = fft_data(x, fs=fs, n=n, one_sided=one_sided, normalize=normalize, input_time=input_time)
        ydata, ylabel = spectrum_display_data(spectrum, which=which, db=db, eps=eps)
        self.plot(freqs, ydata, row=row, col=col, **scatter_kw)
        self._set_spectrum_axis_titles(fs, ylabel, row=row, col=col)
        return spectrum, freqs

    def plot_periodogram(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        x: _Array1D,
        fs: float = 1.0,
        n: int | None = None,
        *,
        window: str = "boxcar",
        one_sided: bool = True,
        normalize: bool = False,
        which: str = "power",
        db: bool = False,
        eps: float = 1e-20,
        row: int = 1,
        col: int = 1,
        **scatter_kw: typing.Any,
    ) -> tuple[_Array1D, _Array1D]:
        """Plot the PSD estimate of a signal via ``scipy.signal.periodogram``.

        Args:
            x: 1D input signal. Complex inputs always yield two-sided plots.
            fs: Sampling frequency in Hz.
            n: FFT length; ``None`` uses ``len(x)``.
            window: Window name accepted by ``scipy.signal.periodogram``.
            one_sided: If True, plot only positive frequencies (auto-disabled for complex x).
            normalize: If True, scale the PSD so its peak is 1.
            which: One of ``"amp"``, ``"power"``, ``"phase"``.
            db: If True (and ``which`` is amp/power), use a dB y-axis.
            eps: Floor added before ``log10`` to avoid log(0).
            row: Target subplot row (1-based).
            col: Target subplot column (1-based).
            **scatter_kw: Forwarded to the line trace.

        Returns:
            ``(psd, freqs)``.
        """
        psd, freqs = periodogram_data(x, fs=fs, n=n, window=window, one_sided=one_sided, normalize=normalize)
        ydata, ylabel = spectrum_display_data(
            typing.cast(_Array1D, np.sqrt(psd)),
            which=which,
            db=db,
            eps=eps,
        )
        self.plot(freqs, ydata, row=row, col=col, **scatter_kw)
        self._set_spectrum_axis_titles(fs, ylabel, row=row, col=col)
        return psd, freqs

    def plot_frequency_response(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        b: _Array1D | float,
        a: _Array1D | float = 1,
        fs: float | None = 1.0,
        num_freq_points: int = 512,
        *,
        one_sided: bool = True,
        which: str = "amp",
        db: bool = False,
        eps: float = 1e-20,
        normalize: bool = False,
        row: int = 1,
        col: int = 1,
        **scatter_kw: typing.Any,
    ) -> tuple[_Array1D, _Array1D]:
        """Plot the frequency response of an LTI system given its ``(b, a)`` coefficients.

        Args:
            b: 1D numerator coefficients (or a scalar).
            a: 1D denominator coefficients (``a == 1`` for FIR systems).
            fs: Sampling frequency in Hz. ``None`` selects continuous-time.
            num_freq_points: Number of frequency points to evaluate.
            one_sided: If True, plot only positive frequencies.
            which: One of ``"amp"``, ``"power"``, ``"phase"``.
            db: If True (and ``which`` is amp/power), use a dB y-axis.
            eps: Floor added before ``log10`` to avoid log(0).
            normalize: If True, scale the response so its peak magnitude is 1.
            row: Target subplot row (1-based).
            col: Target subplot column (1-based).
            **scatter_kw: Forwarded to the line trace.

        Returns:
            ``(h, freqs)``.
        """
        h, freqs = frequency_response_data(b, a, fs=fs, num_freq_points=num_freq_points, one_sided=one_sided)
        if normalize:
            h = typing.cast(_Array1D, h / np.abs(h).max())
        ydata, ylabel = spectrum_display_data(h, which=which, db=db, eps=eps)
        self.plot(freqs, ydata, row=row, col=col, **scatter_kw)
        self._set_spectrum_axis_titles(fs, ylabel, row=row, col=col)
        return h, freqs

    def plot_impulse_response(
        self,
        b: _Array1D | float,
        a: _Array1D | float = 1,
        dt: float | None = 1,
        t: _Array1D | None = None,
        *,
        n: int | None = None,
        row: int = 1,
        col: int = 1,
        **scatter_kw: typing.Any,
    ) -> tuple[_Array1D, _Array1D]:
        """Plot the impulse response of an LTI system given its ``(b, a)`` coefficients.

        Discrete-time responses are drawn as a stem plot (markers plus vertical
        segments — plotly has no native stem); continuous-time as a line.

        Args:
            b: 1D numerator coefficients (or a scalar).
            a: 1D denominator coefficients (``a == 1`` for FIR systems).
            dt: Sample period. If None, treats the system as continuous-time and requires ``t``.
            t: 1D time grid. If None for discrete-time, ``n`` is used to build one.
            n: Sample count for the discrete-time response; required when ``t`` is None.
            row: Target subplot row (1-based).
            col: Target subplot column (1-based).
            **scatter_kw: Forwarded to the marker/line trace.

        Returns:
            ``(h, t_out)``.
        """
        h, t_out, is_discrete = impulse_response_data(b, a, dt=dt, t=t, n=n)
        if is_discrete:
            color = scatter_kw.pop("color", COLORWAY[0])
            segments_x = np.column_stack([t_out, t_out, np.full_like(t_out, np.nan)]).ravel()
            segments_y = np.column_stack([np.zeros_like(h), h, np.full_like(h, np.nan)]).ravel()
            self.add_trace(
                go.Scatter(x=segments_x, y=segments_y, mode="lines", line={"color": color}, showlegend=False),
                row=row,
                col=col,
            )
            self.add_trace(
                go.Scatter(x=t_out, y=h, mode="markers", marker={"color": color}, **scatter_kw),
                row=row,
                col=col,
            )
        else:
            self.plot(t_out, h, row=row, col=col, **scatter_kw)
        return h, t_out

    def plot_specgram(
        self,
        y: _Array1D,
        fs: float = 1.0,
        *,
        nfft: int = 4096,
        window: str = "blackmanharris",
        overlap_fraction: float = 0.85,
        row: int = 1,
        col: int = 1,
        **heatmap_kw: typing.Any,
    ) -> tuple[np.ndarray[tuple[int, int], np.dtype[typing.Any]], _Array1D, _Array1D]:
        """Plot a spectrogram of a 1D time-domain signal as a heatmap.

        Args:
            y: 1D time-domain signal.
            fs: Sample rate in Hz; ``fs != 1`` triggers SI-prefix scaling on the frequency axis.
            nfft: Segment/FFT length.
            window: Window name accepted by ``scipy.signal.get_window``.
            overlap_fraction: Segment overlap as a fraction of ``nfft``.
            row: Target subplot row (1-based).
            col: Target subplot column (1-based).
            **heatmap_kw: Forwarded to ``go.Heatmap``.

        Returns:
            ``(spec, freqs, times)`` — dB spectrogram matrix and its axes (freqs unscaled).
        """
        spec, freqs, times = spectrogram_data(y, fs=fs, nfft=nfft, window=window, overlap_fraction=overlap_fraction)

        scale, prefix = si_prefix_scale(float(freqs[-1])) if fs != 1 else (0, "")
        heatmap_kw = {"colorbar": {"title": {"text": "Power [dB]"}}} | heatmap_kw
        self.add_trace(go.Heatmap(x=times, y=freqs / 10**scale, z=spec, **heatmap_kw), row=row, col=col)
        self.update_xaxes(title_text="Time [sec]", row=row, col=col)
        self.update_yaxes(title_text=f"Frequency [{prefix}Hz]", showgrid=False, row=row, col=col)
        return spec, freqs, times

    def plot_animation(
        self,
        data: np.ndarray[tuple[int, int, int], np.dtype[typing.Any]],
        *,
        trace_type: str = "auto",
        titles: Sequence[str] | Callable[[int], str] | None = None,
        frame_duration: int = 100,
        row: int = 1,
        col: int = 1,
        **trace_kw: typing.Any,
    ) -> "Figure":
        """Build an interactive animation (frames + play/pause button + slider).

        Args:
            data: Per-frame data — ``[n_frames, h, w]`` for images (Heatmap frames),
                ``[n_frames, 2, n_pts]`` for ``(x, y)`` line data (Scatter frames).
            trace_type: ``"heatmap"``, ``"lines"``, or ``"auto"``. ``"auto"`` infers
                from the shape and raises for the ambiguous ``h == 2`` case.
            titles: Per-frame figure titles — a sequence indexed by frame, or a
                callable ``(i) -> str``.
            frame_duration: Per-frame duration in milliseconds during playback.
            row: Target subplot row (1-based).
            col: Target subplot column (1-based).
            **trace_kw: Forwarded to the per-frame trace constructor.

        Returns:
            self (chainable).

        Raises:
            ValueError: If ``data`` is not 3D, or the shape is ambiguous with
                ``trace_type="auto"``.
        """
        data_arr = np.asarray(data)
        if data_arr.ndim != 3:
            raise ValueError(f"data must be 3D ([n_frames, h, w] or [n_frames, 2, n_pts]); got ndim={data_arr.ndim}.")
        n_frames = data_arr.shape[0]

        if trace_type == "auto":
            if data_arr.shape[1] == 2:
                raise ValueError("data with second dimension 2 is ambiguous; pass trace_type='heatmap' or 'lines'.")
            trace_type = "heatmap"
        if trace_type not in ("heatmap", "lines"):
            raise ValueError(f"trace_type must be one of 'auto', 'heatmap', 'lines'. Got: {trace_type}")

        def make_trace(frame: typing.Any) -> typing.Any:
            if trace_type == "lines":
                return go.Scatter(x=frame[0], y=frame[1], mode="lines", **trace_kw)
            return go.Heatmap(z=frame, **trace_kw)

        titles_list: list[str] | None
        if callable(titles):
            titles_list = [titles(i) for i in range(n_frames)]
        else:
            titles_list = list(titles) if titles is not None else None

        self.add_trace(make_trace(data_arr[0]), row=row, col=col)
        trace_index = len(self.data) - 1

        self.frames = [
            go.Frame(
                data=[make_trace(data_arr[i])],
                traces=[trace_index],
                name=str(i),
                layout={"title": {"text": titles_list[i]}} if titles_list is not None else None,
            )
            for i in range(n_frames)
        ]

        self.update_layout(
            updatemenus=[
                {
                    "type": "buttons",
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [
                                None,
                                {"frame": {"duration": frame_duration, "redraw": True}, "fromcurrent": True},
                            ],
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                        },
                    ],
                },
            ],
            sliders=[
                {
                    "steps": [
                        {
                            "label": str(i),
                            "method": "animate",
                            "args": [[str(i)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                        }
                        for i in range(n_frames)
                    ],
                },
            ],
        )
        return self
