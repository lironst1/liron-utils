# pylint: disable=no-value-for-parameter,too-many-lines

import typing
from collections.abc import Callable, Iterable

import matplotlib.animation
import matplotlib.cm
import matplotlib.collections
import matplotlib.colors
import numpy as np
import scipy.signal
from matplotlib.axes import Axes as Axes_plt
from matplotlib.figure import Figure

from ...signal_processing.base import interp1
from ...uncertainties_math import to_numpy
from ..common.fitting import curve_fit_confidence_band, curve_fit_prep_data
from ..common.spectra import (
    fft_data,
    impulse_response_data,
    periodogram_data,
    spectrum_display_data,
)
from .axes import _Axes

_N = typing.TypeVar("_N", bound=int)
_K = typing.TypeVar("_K", bound=int)

# Shape-parameterized aliases — thread the same TypeVar across params to express "same length".
_Vec = np.ndarray[tuple[_N], np.dtype[typing.Any]]
_Mat = np.ndarray[tuple[_N, _K], np.dtype[typing.Any]]

# Fixed-dimensionality aliases (use when same-length constraints aren't expressed).
_Array1D = np.ndarray[tuple[int], np.dtype[typing.Any]]
_Array2D = np.ndarray[tuple[int, int], np.dtype[typing.Any]]
_Array3D = np.ndarray[tuple[int, int, int], np.dtype[typing.Any]]

# Catch-all for opaque/arbitrary-dim arrays (used for object-dtype outputs from `_vectorize`).
_Array = np.ndarray[typing.Any, np.dtype[typing.Any]]

# TODO:
#   - add option to call gr.Axes.plot(ax, x, y) with ax=plt.Axes (add decorator to outer plotting functions),
#     and change class name to be Plot
#   - add standalone functions out of class gr.plot
#   - PyWavelet.dwt (wavelet transform) and plot_wavelet


class Axes(_Axes):

    def plot(
        self,
        x: _Vec[_N],
        y: _Vec[_N] | None = None,
        z: _Vec[_N] | None = None,
        **plot_kw: typing.Any,
    ) -> _Array:
        """Plot 2D curve y=f(x) (or 3D when z is given) on each axis of the grid.

        Args:
            x: 1D x-axis data of length N. If y is None, x is treated as y and the
                x-axis becomes range(len(x)).
            y: 1D y-axis data of length N.
            z: 1D z-axis data of length N for 3D plots.
            **plot_kw: Forwarded to matplotlib.axes.Axes.plot.

        Returns:
            Object-dtype ndarray with one list[Line2D] per axis.
        """

        @self._merge_kwargs("plot_kw", **plot_kw)
        @self._vectorize(x=x, y=y, z=z)
        def _plot(
            ax: Axes_plt,
            x: _Vec[_N],
            y: _Vec[_N] | None = None,
            z: _Vec[_N] | None = None,
            **plot_kw: typing.Any,
        ) -> list[typing.Any]:
            args: list[_Vec[_N]] = [x]
            if y is not None:
                args += [y]
            if z is not None:
                args += [z]
            return ax.plot(*args, **plot_kw)

        return _plot()

    def plot_vlines(
        self,
        x: _Array1D | float = 0,
        ymin: float = 0,
        ymax: float = 1,
        **plot_kw: typing.Any,
    ) -> _Array:
        """Plot one or more vertical lines on each axis.

        Args:
            x: x positions of the lines — a scalar or a 1D array.
            ymin: Lower y endpoint in data coordinates.
            ymax: Upper y endpoint in data coordinates.
            **plot_kw: Forwarded to matplotlib.axes.Axes.axvline. The optional
                ``label`` kwarg is only attached to the last drawn line.

        Returns:
            Object-dtype ndarray with one list[Line2D] per axis.
        """

        @self._merge_kwargs("plot_kw", **plot_kw)
        @self._vectorize(x=x, ymin=ymin, ymax=ymax)
        def _plot_vlines(
            ax: Axes_plt,
            x: _Array1D | float = 0,
            ymin: float = 0,
            ymax: float = 1,
            **plot_kw: typing.Any,
        ) -> list[typing.Any]:
            x_arr = typing.cast(_Array1D, np.atleast_1d(x))

            label: str | None = None
            if "label" in plot_kw:
                label = plot_kw.pop("label")

            lines: list[typing.Any] = []
            for i, xx in enumerate(x_arr):
                line = ax.axvline(
                    x=xx,
                    ymin=ymin,
                    ymax=ymax,
                    label=label if i == x_arr.shape[0] - 1 else "_nolabel_",
                    **plot_kw,
                )
                lines.append(line)

            return lines

        return _plot_vlines()

    def plot_hlines(
        self,
        y: _Array1D | float = 0,
        xmin: float = 0,
        xmax: float = 1,
        **plot_kw: typing.Any,
    ) -> _Array:
        """Plot one or more horizontal lines on each axis.

        Args:
            y: y positions of the lines — a scalar or a 1D array.
            xmin: Left x endpoint in data coordinates.
            xmax: Right x endpoint in data coordinates.
            **plot_kw: Forwarded to matplotlib.axes.Axes.axhline. The optional
                ``label`` kwarg is only attached to the last drawn line.

        Returns:
            Object-dtype ndarray with one list[Line2D] per axis.
        """

        @self._merge_kwargs("plot_kw", **plot_kw)
        @self._vectorize(y=y, xmin=xmin, xmax=xmax)
        def _plot_hlines(
            ax: Axes_plt,
            y: _Array1D | float = 0,
            xmin: float = 0,
            xmax: float = 1,
            **plot_kw: typing.Any,
        ) -> list[typing.Any]:
            y_arr = typing.cast(_Array1D, np.atleast_1d(y))

            label: str | None = None
            if "label" in plot_kw:
                label = plot_kw.pop("label")

            lines: list[typing.Any] = []
            for i, yy in enumerate(y_arr):
                line = ax.axhline(
                    y=yy,
                    xmin=xmin,
                    xmax=xmax,
                    label=label if i == y_arr.shape[0] - 1 else "_nolabel_",
                    **plot_kw,
                )
                lines.append(line)

            return lines

        return _plot_hlines()

    def plot_errorbar(
        self,
        x: _Vec[_N],
        y: _Vec[_N] | None = None,
        xerr: _Vec[_N] | None = None,
        yerr: _Vec[_N] | None = None,
        **errorbar_kw: typing.Any,
    ) -> _Array:
        """Plot y=f(x) with error bars on each axis.

        If ``y`` is None, ``x`` is treated as the y-axis data and the x-axis
        becomes ``range(len(x))``. All four arrays share length N.

        Args:
            x: 1D x-axis data of length N. May also be an uncertainties array; in
                that case ``xerr`` is taken from the embedded uncertainties.
            y: 1D y-axis data of length N. May also be an uncertainties array.
            xerr: 1D errors in x, length N.
            yerr: 1D errors in y, length N.
            **errorbar_kw: Forwarded to matplotlib.axes.Axes.errorbar.

        Returns:
            Object-dtype ndarray with one ErrorbarContainer per axis.
        """

        @self._merge_kwargs("errorbar_kw", **errorbar_kw)
        @self._vectorize(x=x, y=y, xerr=xerr, yerr=yerr)
        def _plot_errorbar(
            ax: Axes_plt,
            x: _Vec[_N],
            y: _Vec[_N] | None = None,
            xerr: _Vec[_N] | None = None,
            yerr: _Vec[_N] | None = None,
            **errorbar_kw: typing.Any,
        ) -> typing.Any:
            if y is None:
                assert xerr is None and yerr is None, "If y is not given, xerr and yerr should not be given."
                y = x
                x = typing.cast(_Vec[_N], np.arange(len(y)))

            x, xerr = typing.cast(tuple[_Vec[_N], _Vec[_N] | None], to_numpy(x, xerr))
            y, yerr = typing.cast(tuple[_Vec[_N], _Vec[_N] | None], to_numpy(y, yerr))

            return ax.errorbar(x, y, xerr=xerr, yerr=yerr, **errorbar_kw)

        return _plot_errorbar()

    def plot_filled_error(
        self,
        ax: Axes_plt,
        x: _Vec[_N],
        y: _Vec[_N] | None = None,
        *,
        yerr: _Vec[_N] | None = None,
        n_std: float = 2,
        y_low: _Vec[_N] | None = None,
        y_high: _Vec[_N] | None = None,
        **fill_between_kw: typing.Any,
    ) -> _Array:
        """Plot a y±n_std·yerr confidence band as a filled area.

        Either ``(y, yerr)`` or ``(y_low, y_high)`` must be given. All arrays share length N.

        Args:
            ax: Axes to draw on.
            x: 1D x-axis data of length N.
            y: 1D y-axis center values, length N; required when ``y_low``/``y_high`` are None.
            yerr: 1D errors in y, length N; the band is drawn at ``y ± n_std·yerr``.
            n_std: Number of standard deviations spanned by the band.
            y_low: 1D lower bound of the band, length N; required when ``y`` is None.
            y_high: 1D upper bound of the band, length N; required when ``y`` is None.
            **fill_between_kw: Forwarded to matplotlib.axes.Axes.fill_between.

        Returns:
            Object-dtype ndarray with one PolyCollection per axis.

        Raises:
            AssertionError: If neither ``(y, yerr)`` nor ``(y_low, y_high)`` is provided.
        """

        @self._merge_kwargs("fill_between_kw", **fill_between_kw)
        @self._vectorize(
            ax=ax,
            x=x,
            y=y,
            yerr=yerr,
            n_std=n_std,
            y_low=y_low,
            y_high=y_high,
        )
        def _plot_filled_error(
            ax: Axes_plt,
            x: _Vec[_N],
            y: _Vec[_N] | None = None,
            *,
            yerr: _Vec[_N] | None = None,
            n_std: float = 2,
            y_low: _Vec[_N] | None = None,
            y_high: _Vec[_N] | None = None,
            **fill_between_kw: typing.Any,
        ) -> typing.Any:
            x = typing.cast(_Vec[_N], to_numpy(x)[0])
            if y is None:
                assert y_low is not None and y_high is not None, "(y, yerr) or (y_low, y_high) should be given."
            else:
                assert y_low is None and y_high is None, "(y, yerr) or (y_low, y_high) should be given."
                y, yerr = typing.cast(tuple[_Vec[_N], _Vec[_N] | None], to_numpy(y, yerr))
                assert yerr is not None, "yerr should be given."
                y_low = typing.cast(_Vec[_N], y - n_std * yerr)
                y_high = typing.cast(_Vec[_N], y + n_std * yerr)

            return ax.fill_between(x, y_low, y_high, **fill_between_kw)

        return _plot_filled_error()

    def plot_data_and_curve_fit(  # pylint: disable=too-many-arguments
        self,
        x: _Vec[_N],
        y: _Vec[_N],
        fit_fcn: Callable[..., _Array1D],
        *,
        xerr: _Vec[_N] | None = None,
        yerr: _Vec[_N] | None = None,
        p_opt: _Vec[_K] | None = None,
        p_cov: _Mat[_K, _K] | None = None,
        n_std: float = 2,
        interp_factor: int = 20,
        curve_fit_plot_kw: dict[str, typing.Any] | None = None,
        **errorbar_kw: typing.Any,
    ) -> _Array:
        """Scatter the data with errorbars and overlay a smooth curve fit with a confidence band.

        The data is shown via ``plot_errorbar``. The mid curve uses ``fit_fcn(x, *p_opt)``
        evaluated on a denser x-axis (``interp_factor`` × original sample count). When both
        ``p_opt`` and ``p_cov`` are provided, a filled ±n_std·σ confidence band is added.

        Args:
            x: 1D x-axis data of length N.
            y: 1D y-axis data of length N.
            fit_fcn: Model function ``f(x, *params)``; the first argument is the independent variable.
            xerr: 1D errors in x, length N. May be omitted if x is an uncertainties array.
            yerr: 1D errors in y, length N. May be omitted if y is an uncertainties array.
            p_opt: 1D best-fit parameters of length K (typically scipy.optimize.curve_fit output).
            p_cov: K×K covariance of the fit parameters (typically scipy.optimize.curve_fit output).
            n_std: Number of standard deviations spanned by the confidence band.
            interp_factor: x-axis upsampling factor used to draw the smooth fit curve.
            curve_fit_plot_kw: Forwarded to the inner ``ax.plot`` call for the fit line.
            **errorbar_kw: Forwarded to ``plot_errorbar`` for the data scatter.

        Returns:
            Object-dtype ndarray (one entry per axis) — the result of the inner vectorize.

        Example:
            >>> import numpy as np
            >>> import scipy.optimize
            >>> from liron_utils.graphics import mpl as gr
            >>>
            >>> N = 101
            >>> x = np.linspace(0, 10, N)
            >>> yerr = 5 * np.random.randn(N)
            >>> y = 2 * x ** 2 + 4 * x + 5 + yerr
            >>>
            >>> def fit_fcn(x, a, b, c):
            ...     return a * x ** 2 + b * x + c
            >>>
            >>> p_opt, p_cov = scipy.optimize.curve_fit(fit_fcn, x, y)
            >>> axes = gr.Axes()
            >>> axes.plot_data_and_curve_fit(x, y, fit_fcn, yerr=yerr, p_opt=p_opt, p_cov=p_cov)
            >>> axes.show_fig()
        """

        @self._merge_kwargs("errorbar_kw", **errorbar_kw)
        @self._vectorize(
            x=x,
            y=y,
            fit_fcn=fit_fcn,
            xerr=xerr,
            yerr=yerr,
            p_opt=p_opt,
            p_cov=p_cov,
            n_std=n_std,
            interp_factor=interp_factor,
            curve_fit_plot_kw=curve_fit_plot_kw,
        )
        def _plot_data_and_curve_fit(  # pylint: disable=too-many-arguments
            ax: Axes_plt,
            x: _Vec[_N],
            y: _Vec[_N],
            fit_fcn: Callable[..., _Array1D],
            *,
            xerr: _Vec[_N] | None = None,
            yerr: _Vec[_N] | None = None,
            p_opt: _Vec[_K] | None = None,
            p_cov: _Mat[_K, _K] | None = None,
            n_std: float = 2,
            interp_factor: int = 20,
            curve_fit_plot_kw: dict[str, typing.Any] | None = None,
            **errorbar_kw: typing.Any,
        ) -> None:
            errorbar_kw = {"label": "Data", "zorder": -1} | errorbar_kw

            if curve_fit_plot_kw is None:
                curve_fit_plot_kw = {}
            curve_fit_plot_kw = {"label": "Curve fit"} | curve_fit_plot_kw

            x, y, xerr, yerr, p_opt = curve_fit_prep_data(x, y, xerr, yerr, p_opt)

            self.plot_errorbar(x, y, xerr=xerr, yerr=yerr, **errorbar_kw)

            x_interp = interp1(x, interp_factor * len(x))
            if p_opt is not None:
                ax.plot(x_interp, fit_fcn(x_interp, *p_opt), **curve_fit_plot_kw)
                if p_cov is not None:
                    fit_low, fit_high = curve_fit_confidence_band(fit_fcn, x_interp, p_opt, p_cov, n_std)
                    self.plot_filled_error(ax=ax, x=x_interp, y_low=fit_low, y_high=fit_high)

        return _plot_data_and_curve_fit()

    def plot_data_and_lin_reg(
        self,
        x: _Vec[_N],
        y: _Vec[_N],
        reg: typing.Any = None,
        *,
        xerr: _Vec[_N] | None = None,
        yerr: _Vec[_N] | None = None,
        reg_plot_kw: dict[str, typing.Any] | None = None,
        **errorbar_kw: typing.Any,
    ) -> _Array:
        """Scatter the data with errorbars and overlay a linear regression line.

        Args:
            x: 1D x-axis data of length N.
            y: 1D y-axis data of length N.
            reg: Output of ``scipy.stats.linregress`` (used to draw the regression line and
                label it with slope/stderr/R²).
            xerr: 1D errors in x, length N. May be omitted if x is an uncertainties array.
            yerr: 1D errors in y, length N. May be omitted if y is an uncertainties array.
            reg_plot_kw: Forwarded to the inner ``plot`` call for the regression line.
            **errorbar_kw: Forwarded to ``plot_errorbar`` for the data scatter.

        Returns:
            Object-dtype ndarray (one entry per axis) — the result of the inner vectorize.

        Example:
            >>> import numpy as np
            >>> from scipy.stats import linregress
            >>> from liron_utils.graphics import mpl as gr
            >>>
            >>> N = 100
            >>> x = np.arange(N)
            >>> y = 2 * x + np.random.randn(N)
            >>> reg = linregress(x, y)
            >>> axes = gr.Axes()
            >>> axes.plot_data_and_lin_reg(x, y, reg)
            >>> axes.show_fig()
        """

        @self._merge_kwargs("errorbar_kw", **errorbar_kw)
        @self._vectorize(x=x, y=y, reg=reg, xerr=xerr, yerr=yerr, reg_plot_kw=reg_plot_kw)
        def _plot_data_and_lin_reg(
            ax: Axes_plt,  # pylint: disable=unused-argument
            x: _Vec[_N],
            y: _Vec[_N],
            reg: typing.Any = None,
            *,
            xerr: _Vec[_N] | None = None,
            yerr: _Vec[_N] | None = None,
            reg_plot_kw: dict[str, typing.Any] | None = None,
            **errorbar_kw: typing.Any,
        ) -> None:
            errorbar_kw = {"label": "Data"} | errorbar_kw

            if reg_plot_kw is None:
                reg_plot_kw = {}
            reg_plot_kw = {
                "label": rf"{errorbar_kw['label']} linreg: slope={reg.slope:.3f}$\pm${reg.stderr:.3f}, "
                rf"$R^2$={reg.rvalue ** 2:.3f}",
            } | reg_plot_kw

            x, xerr = to_numpy(x, xerr)
            y, yerr = to_numpy(y, yerr)

            self.plot_errorbar(x, y, xerr=xerr, yerr=yerr, **errorbar_kw)  # TODO: change to _plot_errorbar

            if reg is not None:
                y_reg = typing.cast(_Array, np.asarray([reg.slope * xi + reg.intercept for xi in x]))
                self.plot(x, y_reg, **reg_plot_kw)

        return _plot_data_and_lin_reg()

    def plot_line_collection(
        self,
        x: _Vec[_N],
        y: _Vec[_N],
        arr: _Vec[_N],
        colorbar_kw: dict[str, typing.Any] | None = None,
        **LineCollection_kw: typing.Any,
    ) -> _Array:
        """Plot a line whose segments are colored according to a third array.

        Args:
            x: 1D x-coordinates of the line vertices (length N).
            y: 1D y-coordinates of the line vertices (length N).
            arr: 1D per-vertex values of length N, mapped to colors via the LineCollection's norm/cmap.
            colorbar_kw: Forwarded to ``Figure.colorbar``.
            **LineCollection_kw: Forwarded to ``matplotlib.collections.LineCollection``.

        Returns:
            Object-dtype ndarray with one ``(LineCollection, Colorbar)`` tuple per axis.
        """

        @self._vectorize(x=x, y=y, arr=arr)
        def _plot_line_collection(
            ax: Axes_plt,
            x: _Vec[_N],
            y: _Vec[_N],
            arr: _Vec[_N],
            colorbar_kw: dict[str, typing.Any] | None = None,
            **LineCollection_kw: typing.Any,
        ) -> tuple[typing.Any, typing.Any]:
            if colorbar_kw is None:
                colorbar_kw = {}

            points = np.array([x, y]).T.reshape((-1, 1, 2))
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            norm = matplotlib.colors.Normalize(vmin=np.min(arr), vmax=np.max(arr))

            lc = matplotlib.collections.LineCollection(
                typing.cast(typing.Any, segments),
                norm=norm,
                **LineCollection_kw,
            )
            lc.set_array(arr)

            line = ax.add_collection(lc)
            cbar = ax.figure.colorbar(line, ax=ax, **colorbar_kw)

            return line, cbar

        return _plot_line_collection(colorbar_kw=colorbar_kw, **LineCollection_kw)

    def _plot_spectrum(
        self,
        ax: Axes_plt,
        spectrum: _Vec[_N],
        freqs: _Vec[_N],
        fs: float = 1.0,
        *,
        db: bool = False,
        eps: float = 1e-20,
        which: str = "power",
        **plot_kw: typing.Any,
    ) -> list[typing.Any]:
        """Plot a complex spectrum as amplitude, power, or phase (linear or dB).

        Args:
            ax: Axes to draw on.
            spectrum: 1D complex spectrum of length N.
            freqs: 1D frequency axis of length N matching ``spectrum``.
            fs: Sampling frequency in Hz. ``fs == 1.0`` is treated as "normalized" for the x-label.
            db: If True (and ``which`` is amp/power), convert the y-axis to dB.
            eps: Floor added before ``log10`` to avoid log(0).
            which: One of ``"amp"``, ``"power"``, ``"phase"``.
            **plot_kw: Forwarded to ``ax.plot``.

        Returns:
            The list of ``Line2D`` returned by matplotlib.

        Raises:
            ValueError: If ``which`` is not one of ``"amp"``, ``"power"``, ``"phase"``.
        """
        ydata, ylabel = spectrum_display_data(spectrum, which=which, db=db, eps=eps)

        line: list[typing.Any] = ax.plot(freqs, ydata, **plot_kw)

        if ax.get_xlabel() == "":
            ax.set_xlabel("Frequency [normalized]" if fs == 1.0 else "Frequency [Hz]")
        if ax.get_ylabel() == "":
            ax.set_ylabel(ylabel)

        return line

    def plot_fft(
        self,
        x: _Array1D,
        fs: float | None = 1.0,
        n: int | None = None,
        *,
        one_sided: bool = True,
        normalize: bool = False,
        input_time: bool = True,
        plot_spectrum_kw: dict[str, typing.Any] | None = None,
        **plot_kw: typing.Any,
    ) -> _Array:
        """Plot the magnitude/power/phase spectrum of an FFT.

        If ``input_time`` is True, ``x`` is FFT-transformed first; otherwise ``x`` is
        treated as already-transformed frequency-domain data.

        Args:
            x: 1D input signal. Complex inputs always yield two-sided plots.
            fs: Sampling frequency in Hz. ``fs == 1.0`` produces a normalized frequency axis.
            n: FFT length. If None, uses ``len(x)``.
            one_sided: If True, plot only the positive frequencies (auto-disabled for complex x).
            normalize: If True, scale the spectrum so its peak magnitude is 1.
            input_time: If True, treat ``x`` as time-domain and apply FFT.
                If False, treat ``x`` as already in the frequency domain.
            plot_spectrum_kw: Forwarded to ``_plot_spectrum`` (controls ``db``, ``which``, ``eps``).
            **plot_kw: Forwarded to ``ax.plot``.

        Returns:
            Object-dtype ndarray with one ``((spectrum, freqs), list[Line2D])`` tuple per axis.
        """

        @self._merge_kwargs("plot_kw", **plot_kw)
        @self._vectorize(
            x=x,
            fs=fs,
            n=n,
            one_sided=one_sided,
            normalize=normalize,
            input_time=input_time,
            plot_spectrum_kw=plot_spectrum_kw,
        )
        def _plot_fft(
            ax: Axes_plt,
            x: _Array1D,
            fs: float = 1.0,
            n: int | None = None,
            *,
            one_sided: bool = True,
            normalize: bool = False,
            input_time: bool = True,
            plot_spectrum_kw: dict[str, typing.Any] | None = None,
            **plot_kw: typing.Any,
        ) -> tuple[tuple[_Array1D, _Array1D], list[typing.Any]]:
            spectrum, freqs = fft_data(
                x,
                fs=fs,
                n=n,
                one_sided=one_sided,
                normalize=normalize,
                input_time=input_time,
            )

            if plot_spectrum_kw is None:
                plot_spectrum_kw = {}

            line = self._plot_spectrum(
                ax=ax,
                spectrum=spectrum,
                freqs=freqs,
                fs=fs,
                **plot_spectrum_kw,
                **plot_kw,
            )

            return (spectrum, freqs), line

        return _plot_fft()

    def plot_periodogram(
        self,
        x: _Array1D,
        fs: float = 1.0,
        n: int | None = None,
        *,
        window: str = "boxcar",
        one_sided: bool = True,
        normalize: bool = False,
        plot_spectrum_kw: dict[str, typing.Any] | None = None,
        **plot_kw: typing.Any,
    ) -> _Array:
        """Plot the PSD estimate of a signal via ``scipy.signal.periodogram``.

        Args:
            x: 1D input signal. Complex inputs always yield two-sided plots.
            fs: Sampling frequency in Hz.
            n: FFT length; ``None`` uses ``len(x)``.
            window: Window function name accepted by ``scipy.signal.periodogram``.
            one_sided: If True, return only the positive frequencies (auto-disabled for complex x).
            normalize: If True, scale the PSD so its peak is 1.
            plot_spectrum_kw: Forwarded to ``_plot_spectrum`` (controls ``db``, ``which``, ``eps``).
            **plot_kw: Forwarded to ``ax.plot``.

        Returns:
            Object-dtype ndarray with one ``((psd, freqs), list[Line2D])`` tuple per axis.
        """

        @self._merge_kwargs("plot_kw", **plot_kw)
        @self._vectorize(
            x=x,
            fs=fs,
            n=n,
            window=window,
            one_sided=one_sided,
            normalize=normalize,
            plot_spectrum_kw=plot_spectrum_kw,
        )
        def _plot_periodogram(
            ax: Axes_plt,
            x: _Array1D,
            fs: float = 1.0,
            n: int | None = None,
            *,
            window: str = "boxcar",
            one_sided: bool = True,
            normalize: bool = False,
            plot_spectrum_kw: dict[str, typing.Any] | None = None,
            **plot_kw: typing.Any,
        ) -> tuple[tuple[_Array1D, _Array1D], list[typing.Any]]:
            psd, freqs = periodogram_data(x, fs=fs, n=n, window=window, one_sided=one_sided, normalize=normalize)

            if plot_spectrum_kw is None:
                plot_spectrum_kw = {}

            line = self._plot_spectrum(
                ax=ax,
                spectrum=typing.cast(_Array1D, np.sqrt(psd)),
                freqs=freqs,
                fs=fs,
                **plot_spectrum_kw,
                **plot_kw,
            )

            return (psd, freqs), line

        return _plot_periodogram()

    def plot_impulse_response(
        self,
        b: _Array1D | float,
        a: _Array1D | float = 1,
        dt: float | None = 1,
        t: _Array1D | None = None,
        *,
        n: int | None = None,
        **plot_kw: typing.Any,
    ) -> _Array:
        """Plot the impulse response of an LTI system given its ``(b, a)`` coefficients.

        Args:
            b: 1D numerator coefficients (or a scalar).
            a: 1D denominator coefficients (set ``a == 1`` for FIR systems).
            dt: Sample period. If None, treats the system as continuous-time and requires ``t``.
            t: 1D time grid for the response. If None for discrete-time, ``n`` is used to build one.
            n: Sample count for the discrete-time response; required when ``t`` is None.
            **plot_kw: Forwarded to ``ax.plot`` (continuous) or ``ax.stem`` (discrete).

        Returns:
            Object-dtype ndarray with one ``((h, t_out), line/stem)`` per axis.

        Raises:
            ValueError: For continuous-time when ``t`` is not given; for discrete-time when neither
                ``t`` nor ``n`` is given.
        """

        @self._merge_kwargs("plot_kw", **plot_kw)
        @self._vectorize(b=b, a=a, dt=dt, t=t, n=n)
        def _plot_impulse_response(
            ax: Axes_plt,
            b: _Array1D | float,
            a: _Array1D | float = 1,
            dt: float | None = 1,
            t: _Array1D | None = None,
            *,
            n: int | None = None,
            **plot_kw: typing.Any,
        ) -> tuple[tuple[_Array1D, _Array1D], typing.Any]:
            h, t_out, is_discrete = impulse_response_data(b, a, dt=dt, t=t, n=n)
            line = ax.stem(t_out, h, **plot_kw) if is_discrete else ax.plot(t_out, h, **plot_kw)
            return (h, t_out), line

        return _plot_impulse_response()

    def plot_frequency_response(  # pylint: disable=too-many-arguments
        self,
        b: _Array1D | float,
        a: _Array1D | float = 1,
        fs: float | None = 1.0,
        num_freq_points: int = 512,
        *,
        one_sided: bool = True,
        db: bool = False,
        eps: float = 1e-20,
        which: str = "amp",
        normalize: bool = False,
        **plot_kw: typing.Any,
    ) -> _Array:
        """Plot the frequency response of an LTI system given its ``(b, a)`` coefficients.

        For ``fs is None`` the system is treated as continuous-time and ``scipy.signal.freqs``
        is used; otherwise ``scipy.signal.freqz`` is used.

        Args:
            b: 1D numerator coefficients (or a scalar).
            a: 1D denominator coefficients (set ``a == 1`` for FIR systems).
            fs: Sampling frequency in Hz. ``None`` selects continuous-time.
            num_freq_points: Number of frequency points to evaluate.
            one_sided: If True, plot only positive frequencies.
            db: If True (and ``which`` is amp/power), use a dB y-axis.
            eps: Floor added before ``log10`` to avoid log(0).
            which: One of ``"amp"``, ``"power"``, ``"phase"``.
            normalize: If True, scale the response so its peak magnitude is 1.
            **plot_kw: Forwarded to the inner ``plot_fft`` call.

        Returns:
            Object-dtype ndarray with one ``((h, freqs), list[Line2D])`` per axis.
        """

        @self._merge_kwargs("plot_kw", **plot_kw)
        @self._vectorize(
            b=b,
            a=a,
            fs=fs,
            num_freq_points=num_freq_points,
            one_sided=one_sided,
            db=db,
            eps=eps,
            which=which,
            normalize=normalize,
        )
        def _plot_frequency_response(  # pylint: disable=too-many-arguments
            ax: Axes_plt,
            b: _Array1D | float,
            a: _Array1D | float = 1,
            fs: float | None = 1.0,
            num_freq_points: int = 512,
            *,
            one_sided: bool = True,
            db: bool = False,
            eps: float = 1e-20,
            which: str = "amp",
            normalize: bool = False,
            **plot_kw: typing.Any,
        ) -> tuple[tuple[_Array1D, _Array1D], typing.Any]:
            wh: tuple[typing.Any, typing.Any]
            if fs is None:  # Continuous-time system
                wh = scipy.signal.freqs(
                    b=typing.cast(typing.Any, b),
                    a=typing.cast(typing.Any, a),
                    worN=num_freq_points,
                )
            else:  # Discrete-time system
                wh = scipy.signal.freqz(b=b, a=a, fs=2 * np.pi * fs, worN=num_freq_points, whole=not one_sided)

            h_arr = typing.cast(_Array1D, wh[1])
            freqs = typing.cast(_Array1D, np.asarray(wh[0]) / (2 * np.pi) - 0.5)

            _, line = Axes(axs=ax).plot_fft(  # pylint: disable=invalid-sequence-index
                x=h_arr,
                fs=fs,
                n=2 * num_freq_points if one_sided else num_freq_points,
                one_sided=one_sided,
                db=db,
                eps=eps,
                which=which,
                normalize=normalize,
                input_time=False,
                **plot_kw,
            )[0, 0]

            return (h_arr, freqs), line

        return _plot_frequency_response()

    def plot_specgram(
        self,
        y: _Array1D,
        fs: float,
        **specgram_kw: typing.Any,
    ) -> _Array:
        """Plot a spectrogram of a 1D time-domain signal on each axis.

        Args:
            y: 1D time-domain signal.
            fs: Sample rate in Hz; ``fs != 1`` triggers automatic SI-prefix scaling on the frequency axis.
            **specgram_kw: Forwarded to ``matplotlib.axes.Axes.specgram``.

        Returns:
            Object-dtype ndarray; each entry is ``(spectrum, freqs, times, AxesImage)`` from matplotlib.
        """

        @self._merge_kwargs("specgram_kw", **specgram_kw)
        @self._vectorize(y=y, fs=fs)
        def _plot_specgram(
            ax: Axes_plt,
            y: _Array1D,
            fs: float,
            **specgram_kw: typing.Any,
        ) -> tuple[typing.Any, ...]:
            specgram_out = ax.specgram(
                y,
                Fs=fs,
                **specgram_kw,
            )  # todo: add option for log frequency mapping using librosa.feature.melspectrogram()
            _, freqs, _, im = specgram_out

            scaling: dict[int, str] = {
                0: "",
                3: "K",
                6: "M",
                9: "G",
                12: "T",
            }  # todo: use siprefix.si_format
            scale = 0
            if fs != 1:
                scale = 3 * (int(np.log10(freqs[-1])) // 3)

                extent = im.get_extent()
                extent = (
                    extent[0],
                    extent[1],
                    extent[2] / 10**scale,
                    extent[3] / 10**scale,
                )
                im.set_extent(extent=extent)

            if ax.get_title() == "":
                ax.set_title("Spectrogram")
            if ax.get_xlabel() == "":
                ax.set_xlabel("Time [sec]")
            if ax.get_ylabel() == "":
                ax.set_ylabel(f"Frequency [{scaling[scale]}Hz]")

            ax.figure.colorbar(im, ax=ax, label="Power [dB]")
            ax.grid(False)

            return specgram_out

        return _plot_specgram()

    @typing.overload
    def plot_surf(
        self,
        x: _Array1D,
        y: _Array1D,
        z: _Array2D | Callable[[_Array2D, _Array2D], _Array2D],
        **plot_surface_kw: typing.Any,
    ) -> _Array: ...

    @typing.overload
    def plot_surf(
        self,
        x: _Array2D,
        y: _Array2D,
        z: _Array2D | Callable[[_Array2D, _Array2D], _Array2D],
        **plot_surface_kw: typing.Any,
    ) -> _Array: ...

    def plot_surf(
        self,
        x: _Array1D | _Array2D,
        y: _Array1D | _Array2D,
        z: _Array2D | Callable[[_Array2D, _Array2D], _Array2D],
        **plot_surface_kw: typing.Any,
    ) -> _Array:
        """Plot a 3D surface ``z = f(x, y)`` on each axis (requires ``projection='3d'``).

        Args:
            x: x-coordinates — 1D of length M (``meshgrid`` is applied) or 2D meshgrid (M×N).
            y: y-coordinates — 1D of length N (``meshgrid`` is applied) or 2D meshgrid (M×N).
            z: 2D array of z-values (M×N), or a callable ``f(x_grid, y_grid) -> z_grid``.
            **plot_surface_kw: Forwarded to ``mpl_toolkits.mplot3d.Axes3D.plot_surface``.

        Returns:
            Object-dtype ndarray (one ``Poly3DCollection`` per axis).

        Raises:
            AssertionError: If the underlying axes were not created with ``projection='3d'``.
        """

        @self._merge_kwargs("plot_surface_kw", **plot_surface_kw)
        @self._vectorize(x=x, y=y, z=z)
        def _plot_surf(
            ax: Axes_plt,
            x: _Array1D | _Array2D,
            y: _Array1D | _Array2D,
            z: _Array2D | Callable[[_Array2D, _Array2D], _Array2D],
            **plot_surface_kw: typing.Any,
        ) -> None:
            assert hasattr(ax, "plot_surface"), (
                "Axes does not have a plot_surface attribute. "
                "make sure that you created an axes with projection='3d'"
            )

            x_grid: _Array2D
            y_grid: _Array2D
            if x.ndim == 1:
                grids = typing.cast(list[_Array2D], np.meshgrid(x, y))
                x_grid, y_grid = grids[0], grids[1]
            else:
                x_grid, y_grid = typing.cast(_Array2D, x), typing.cast(_Array2D, y)
            z_grid: _Array2D = z(x_grid, y_grid) if callable(z) else z
            if z_grid.shape != x_grid.shape and z_grid.shape == tuple(np.flip(x_grid.shape)):
                z_grid = z_grid.T

            ax.plot_surface(x_grid, y_grid, z_grid, **plot_surface_kw)

        # ax.figure.colorbar(matplotlib.cm.ScalarMappable(), ax=ax)

        return _plot_surf()

    @typing.overload
    def plot_contour(
        self,
        x: _Array1D,
        y: _Array1D,
        z: _Array2D,
        contours: int | _Array1D,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> _Array: ...

    @typing.overload
    def plot_contour(
        self,
        x: _Array2D,
        y: _Array2D,
        z: _Array2D,
        contours: int | _Array1D,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> _Array: ...

    def plot_contour(
        self,
        x: _Array1D | _Array2D,
        y: _Array1D | _Array2D,
        z: _Array2D,
        contours: int | _Array1D,
        *args: typing.Any,
        **kwargs: typing.Any,
    ) -> _Array:
        """Plot labeled contour lines for the scalar field ``z = f(x, y)``.

        Args:
            x: x-coordinates — 1D of length M, or 2D meshgrid (M×N).
            y: y-coordinates — 1D of length N, or 2D meshgrid (M×N).
            z: 2D array of z-values (M×N).
            contours: Number of contour levels (``int``) or an explicit 1D array of levels.
            *args: Forwarded to ``matplotlib.axes.Axes.contour``.
            **kwargs: Forwarded to ``matplotlib.axes.Axes.contour``.

        Returns:
            Object-dtype ndarray (one ``QuadContourSet`` per axis).
        """

        @self._vectorize(x=x, y=y, z=z, contours=contours)
        def _plot_contour(
            ax: Axes_plt,
            x: _Array1D | _Array2D,
            y: _Array1D | _Array2D,
            z: _Array2D,
            contours: int | _Array1D,
            *args: typing.Any,
            **kwargs: typing.Any,
        ) -> typing.Any:
            cs = ax.contour(x, y, z, contours, *args, **kwargs)
            ax.clabel(cs, inline=True, fontsize=10)
            return cs

        return _plot_contour(*args, **kwargs)

    def plot_animation(  # pylint: disable=keyword-arg-before-vararg
        self,
        axs: Axes_plt | _Array,
        func: Callable[[int], typing.Any] | None = None,
        n_frames: int | None = None,
        *args: typing.Any,
        data: list[_Array3D] | None = None,
        data_instance: list[typing.Any] | None = None,
        titles: list[typing.Any] | Callable[[int], str] | None = None,
        **kwargs: typing.Any,
    ) -> None:
        """Build a ``matplotlib.animation.FuncAnimation`` over one or more axes.

        Either ``(func, n_frames)`` must be given, or ``(data, data_instance)`` — in the
        latter case a default per-frame updater is used that calls ``set_data`` on each
        artist handle.

        Args:
            axs: A single axes, or an array of axes (all must share the same figure).
            func: Per-frame callable returning the list of redrawn artists. If None, a default
                updater built from ``data`` and ``data_instance`` is used.
            n_frames: Number of frames. If None, ``len(data[0])`` is used.
            *args: Forwarded to ``matplotlib.animation.FuncAnimation``.
            data: Per-axis 3D data sources of shape ``[n_frames, ...]``:
                ``[n_frames, h, w]`` for images, ``[n_frames, 2, n_pts]`` for line plots.
            data_instance: Per-axis artist handles (image / line / etc.) updated each frame.
            titles: Per-frame axis titles — list indexed by frame, or callable ``(i) -> str``.
            **kwargs: Forwarded to ``matplotlib.animation.FuncAnimation``.

        Raises:
            AssertionError: If neither ``(func, n_frames)`` nor ``(data, data_instance)``
                is given, or if axes are from different figures.

        Example:
            >>> import numpy as np
            >>> from liron_utils.graphics import mpl as gr
            >>>
            >>> images = np.random.random((10, 32, 32))
            >>> axes = gr.Axes()
            >>> axes.plot_animation(axes.axs[0, 0], data=[images], data_instance=[...])
            >>> axes.save_fig("test.gif")
        """
        assert (func is not None and n_frames is not None) or (
            data is not None and data_instance is not None
        ), "Either (func, n_frames) or (data, data_instance) should be given."

        axs_list: list[Axes_plt]
        if isinstance(axs, Axes_plt):
            axs_list = [axs]
            data = typing.cast(list[_Array3D], [data])
            data_instance = [data_instance]
        else:
            axs_list = list(np.asarray(axs).flat)

        assert (
            axs_list[i].figure == axs_list[i + 1].figure for i in range(len(axs_list) - 1)
        ), "All axes should be of the same figure."

        if func is None:
            assert data is not None and data_instance is not None
            assert len(axs_list) == len(data) == len(data_instance), "Number of axes should equal number of data sets."
            n_frames = len(data[0])

        if titles is not None:
            kwargs = {"blit": False} | kwargs

        def update_data(i: int) -> list[typing.Any]:
            assert data is not None and data_instance is not None
            for idx_ax, ax in enumerate(axs_list):
                h = data_instance[idx_ax]
                if isinstance(h, Iterable):
                    for j, hh in enumerate(h):
                        hh.set_data(data[idx_ax][i][j])
                else:
                    h.set_data(data[idx_ax][i])

                if titles is not None:
                    ax.set_title(titles[i])  # type: ignore[index]

            return data_instance

        if func is None:
            func = update_data

        fig_for_anim = axs_list[0].figure
        assert isinstance(fig_for_anim, Figure), "Animation requires a Figure (not SubFigure)."
        self.func_animation = matplotlib.animation.FuncAnimation(
            fig_for_anim,
            func,
            n_frames,
            *args,
            **kwargs,
        )
