import typing
from collections import namedtuple
from collections.abc import Callable

import numpy as np
import scipy.optimize
import scipy.signal
import scipy.stats
from scipy import odr
from sklearn.metrics import r2_score
from uncertainties import ufloat

from ..graphics import mpl as gr
from ..uncertainties_math import from_numpy, to_numpy, val
from .stats import chi_squared_test

# pylint: disable=duplicate-code
_N = typing.TypeVar("_N", bound=int)
_K = typing.TypeVar("_K", bound=int)
_Vec = np.ndarray[tuple[_N], np.dtype[typing.Any]]
_Mat = np.ndarray[tuple[_N, _K], np.dtype[typing.Any]]
_Array1D = np.ndarray[tuple[int], np.dtype[typing.Any]]
# pylint: enable=duplicate-code

linear_regression = scipy.stats.linregress

LinearRegression = namedtuple("LinearRegression", ["h", "eval", "slope", "intercept", "r2"])
# Backwards-compatible alias for the previous public namedtuple name.
Linear_regression = LinearRegression  # noqa: N816  pylint: disable=invalid-name


def _build_linear_regression_result(out: typing.Any, x: _Vec[_N], y: _Vec[_N]) -> LinearRegression:
    """Wrap an ODR output into a :class:`LinearRegression` namedtuple with a predictor closure.

    Args:
        out: Output of ``scipy.odr.ODR.run()``.
        x: 1D x-axis data of length N.
        y: 1D y-axis data of length N.

    Returns:
        :class:`LinearRegression` with slope/intercept as ``ufloat``s, an ``eval``
        closure, and the R² score on the input data.
    """
    slope = ufloat(out.beta[0], out.sd_beta[0])
    intercept = ufloat(out.beta[1], out.sd_beta[1])

    def predict(x_in: typing.Any) -> typing.Any:
        return slope * x_in + intercept

    r2 = r2_score(y, val(predict(x)))
    return LinearRegression(h=out, eval=predict, slope=slope, intercept=intercept, r2=r2)


def linear_fit(
    x: _Vec[_N],
    y: _Vec[_N],
    xerr: _Vec[_N] | None = None,
    yerr: _Vec[_N] | None = None,
    beta0: _Array1D | None = None,
    **kwargs: typing.Any,
) -> LinearRegression:
    """Orthogonal-distance linear regression with optional x/y uncertainties.

    Args:
        x: 1D x-axis data of length N. May be an uncertainties array, in which
            case ``xerr`` is taken from its uncertainties and the keyword is ignored.
        y: 1D y-axis data of length N. May be an uncertainties array (see ``yerr``).
        xerr: 1D uncertainties in x, length N. Disregarded if ``x`` is uncertainties.
        yerr: 1D uncertainties in y, length N. Disregarded if ``y`` is uncertainties.
        beta0: Initial parameter values ``(slope, intercept)``. Defaults to ``(0, 0)``.
        **kwargs: Forwarded to ``scipy.odr.ODR``.

    Returns:
        :class:`LinearRegression` (see :func:`_build_linear_regression_result`).
    """
    x, xerr = typing.cast(tuple[_Vec[_N], _Vec[_N] | None], to_numpy(x, xerr))
    y, yerr = typing.cast(tuple[_Vec[_N], _Vec[_N] | None], to_numpy(y, yerr))

    if xerr is not None:
        xerr[xerr == 0] = np.nan
    if yerr is not None:
        yerr[yerr == 0] = np.nan

    if beta0 is None:
        beta0 = typing.cast(_Array1D, np.array([0, 0]))  # [slope, intercept]

    def model(params: typing.Any, x_in: typing.Any) -> typing.Any:
        return params[0] * x_in + params[1]

    data = odr.RealData(x, y, xerr, yerr)
    out = odr.ODR(data, odr.Model(model), beta0, **kwargs).run()
    return _build_linear_regression_result(out, x, y)


def curve_fit(
    fit_fcn: Callable[..., _Array1D],
    x: _Vec[_N],
    y: _Vec[_N],
    p0: _Array1D | None = None,
    *,
    xerr: _Vec[_N] | None = None,
    yerr: _Vec[_N] | None = None,
    plot: bool = False,
    plot_data_and_curve_fit_kw: dict[str, typing.Any] | None = None,
    **kwargs: typing.Any,
) -> tuple[typing.Any, ...]:
    """Non-linear least-squares fit of ``fit_fcn`` to ``(x, y)`` assuming ``y = f(x, *params) + eps``.

    Args:
        fit_fcn: Model function ``f(x, *params)`` returning a 1D array. The first
            argument is the independent variable.
        x: 1D x-axis data of length N.
        y: 1D y-axis data of length N — nominally ``f(x, *params)``.
        p0: 1D initial parameter guess of length K. If ``None``, scipy initializes to 1s.
        xerr: 1D uncertainties in x, length N (used only for plotting).
        yerr: 1D uncertainties in y, length N. ``None``, a 1D vector of standard
            deviations, or a 2D covariance matrix (passed through to scipy).
        plot: If True, draw the data with errorbars and the curve fit and return
            the ``Axes`` object as a fourth element of the result tuple.
        plot_data_and_curve_fit_kw: Forwarded to :meth:`gr.Axes.plot_data_and_curve_fit`.
        **kwargs: Forwarded to ``scipy.optimize.curve_fit``.

    Returns:
        ``(p_opt, p_cov, chi2_red)`` — or ``(p_opt, p_cov, chi2_red, axes)`` when
        ``plot=True``. ``p_opt`` is the K-length parameter vector with embedded
        uncertainties from ``sqrt(diag(p_cov))``; ``p_cov`` is the K×K covariance;
        ``chi2_red`` is the reduced χ² (or ``None`` when ``yerr is None``).
    """
    if plot_data_and_curve_fit_kw is None:
        plot_data_and_curve_fit_kw = {}

    x, xerr = typing.cast(tuple[_Vec[_N], _Vec[_N] | None], to_numpy(x, xerr))
    y, yerr = typing.cast(tuple[_Vec[_N], _Vec[_N] | None], to_numpy(y, yerr))
    p0 = val(p0)

    if xerr is not None and xerr.size == 1:
        xerr = typing.cast(_Vec[_N], np.repeat(xerr, len(x)))
    if yerr is not None and yerr.size == 1:
        yerr = typing.cast(_Vec[_N], np.repeat(yerr, len(y)))

    p_opt, p_cov = scipy.optimize.curve_fit(fit_fcn, x, y, p0=p0, sigma=yerr, **kwargs)

    if yerr is not None:
        chi2_red = chi_squared_test(f_exp=fit_fcn(x, *p_opt), f_obs=y, f_obs_err=yerr, ddof=len(p_opt), reduced=True)
    else:
        chi2_red = None

    p_opt = from_numpy(p_opt, np.sqrt(np.diag(p_cov)))

    if plot:
        axes = gr.Axes()
        axes.plot_data_and_curve_fit(
            x,
            y,
            fit_fcn,
            xerr=xerr,
            yerr=yerr,
            p_opt=p_opt,
            p_cov=p_cov,
            **plot_data_and_curve_fit_kw,
        )
        return p_opt, p_cov, chi2_red, axes

    return p_opt, p_cov, chi2_red


def _quadratic(x: typing.Any, a: float, b: float, c: float) -> typing.Any:
    """Evaluate ``a·x² + b·x + c``."""
    return a * x**2 + b * x + c


def _refine_single_peak(
    peak: int,
    *,
    x: _Vec[_N],
    y: _Vec[_N],
    xerr: _Vec[_N] | None,
    yerr: _Vec[_N] | None,
    n_fit_points: int,
) -> tuple[typing.Any, typing.Any]:
    """Quadratic fit around a single peak.

    Args:
        peak: Index of the peak in ``y``.
        x: 1D x-axis data of length N.
        y: 1D y-axis data of length N.
        xerr: 1D uncertainties in x, length N.
        yerr: 1D uncertainties in y, length N.
        n_fit_points: Number of points on each side of the peak to include in the fit.

    Returns:
        ``(x_peak, y_peak)`` as uncertainties scalars (refined peak location and value).

    Raises:
        ValueError: If the quadratic fit returns ``a == 0`` (degenerate parabola).
    """
    window = slice(max(0, peak - n_fit_points), min(len(y), peak + n_fit_points + 1))
    x0 = float(np.mean(x[window]))

    a, b, c = _shift_quadratic_origin(
        curve_fit(
            fit_fcn=_quadratic,
            x=typing.cast(_Vec[_N], x[window] - x0),
            y=typing.cast(_Vec[_N], y[window]),
            xerr=typing.cast(_Vec[_N], xerr[window]) if xerr is not None else None,
            yerr=typing.cast(_Vec[_N], yerr[window]) if yerr is not None else None,
            p0=typing.cast(_Array1D, np.asarray([0, 0, y[peak]])),
        )[0],
        x0,
    )

    if val(a) == 0:
        raise ValueError("Quadratic fit failed: a=0")

    x_peak = -b / (2 * a)
    return x_peak, _quadratic(x_peak, a, b, c)


def _shift_quadratic_origin(
    params_centered: typing.Sequence[typing.Any],
    x0: float,
) -> tuple[typing.Any, typing.Any, typing.Any]:
    """Convert ``(a, b, c)`` from a quadratic in ``x - x0`` to one in the original ``x``.

    Args:
        params_centered: Coefficients ``(a, b, c)`` of the centered quadratic.
        x0: Origin offset that was subtracted from x before fitting.

    Returns:
        ``(a, b, c)`` for the equivalent quadratic in the original ``x``.
    """
    a_c, b_c, c_c = params_centered
    return a_c, b_c - 2 * a_c * x0, a_c * x0**2 - b_c * x0 + c_c


def _quadratic_peak_refinement(
    peaks: _Array1D,
    *,
    x: _Vec[_N],
    y: _Vec[_N],
    xerr: _Vec[_N] | None,
    yerr: _Vec[_N] | None,
    n_fit_points: int,
) -> tuple[typing.Any, typing.Any]:
    """Quadratic refinement of integer peak positions using ``2*n_fit_points + 1`` samples around each peak.

    Args:
        peaks: 1D integer indices of peaks in ``y``.
        x: 1D x-axis data of length N.
        y: 1D y-axis data of length N.
        xerr: 1D uncertainties in x, length N.
        yerr: 1D uncertainties in y, length N.
        n_fit_points: Number of points on each side of every peak to include in the fit.

    Returns:
        ``(x_peaks, y_peaks)`` as uncertainties arrays, one entry per input peak.
    """
    x_peaks = from_numpy(np.zeros(len(peaks)), 0)
    y_peaks = from_numpy(np.zeros(len(peaks)), 0)
    for i, peak in enumerate(peaks):
        x_peaks[i], y_peaks[i] = _refine_single_peak(peak, x=x, y=y, xerr=xerr, yerr=yerr, n_fit_points=n_fit_points)
    return x_peaks, y_peaks


def find_peaks(
    y: _Vec[_N],
    yerr: _Vec[_N] | None = None,
    x: _Vec[_N] | None = None,
    xerr: _Vec[_N] | None = None,
    n_fit_points: int | None = None,
    *,
    plot: bool = False,
    **kwargs: typing.Any,
) -> tuple[typing.Any, ...]:
    """Find peaks in ``y`` and optionally refine them with a quadratic fit.

    Args:
        y: 1D signal values of length N. May be an uncertainties array.
        yerr: 1D uncertainties in y, length N.
        x: 1D x-axis values of length N. If ``None``, ``range(len(y))`` is used.
        xerr: 1D uncertainties in x, length N.
        n_fit_points: If given, fit a quadratic around each peak using this many
            points on each side and report the refined ``(x, y)`` of every peak.
        plot: If True, draw the data and the peaks and return the ``Axes`` object
            as a fifth element of the result tuple.
        **kwargs: Forwarded to ``scipy.signal.find_peaks``.

    Returns:
        ``(x_peaks, y_peaks, peaks, properties)`` — or ``(x_peaks, y_peaks, peaks,
        properties, axes)`` when ``plot=True``. ``peaks`` are the integer indices
        returned by scipy; ``x_peaks``/``y_peaks`` are the per-peak coordinates
        (possibly with uncertainties or quadratic refinement).
    """
    x, xerr = typing.cast(tuple[_Vec[_N] | None, _Vec[_N] | None], to_numpy(x, xerr))
    y, yerr = typing.cast(tuple[_Vec[_N], _Vec[_N] | None], to_numpy(y, yerr))
    if x is None:
        x = typing.cast(_Vec[_N], np.arange(len(y)))
    if xerr is not None and xerr.size == 1:
        xerr = typing.cast(_Vec[_N], np.repeat(xerr, len(x)))
    if yerr is not None and yerr.size == 1:
        yerr = typing.cast(_Vec[_N], np.repeat(yerr, len(y)))

    peaks, properties = scipy.signal.find_peaks(val(y), **kwargs)

    x_peaks: typing.Any
    y_peaks: typing.Any
    if n_fit_points is None:
        x_peaks = x[peaks]
        y_peaks = y[peaks]
        if xerr is not None:
            x_peaks = from_numpy(x_peaks, xerr[peaks])
        if yerr is not None:
            y_peaks = from_numpy(y_peaks, yerr[peaks])
    else:
        x_peaks, y_peaks = _quadratic_peak_refinement(peaks, x=x, y=y, xerr=xerr, yerr=yerr, n_fit_points=n_fit_points)

    if plot:
        axes = gr.Axes()
        axes.plot_errorbar(x, y, xerr=xerr, yerr=yerr)
        axes.plot_errorbar(x_peaks, y_peaks, marker="o", color="black", linestyle="none")
        return x_peaks, y_peaks, peaks, properties, axes

    return x_peaks, y_peaks, peaks, properties
