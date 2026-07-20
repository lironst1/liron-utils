import typing
from collections.abc import Callable

import numpy as np

from ...uncertainties_math import to_numpy

_N = typing.TypeVar("_N", bound=int)
_K = typing.TypeVar("_K", bound=int)

_Vec = np.ndarray[tuple[_N], np.dtype[typing.Any]]
_Mat = np.ndarray[tuple[_N, _K], np.dtype[typing.Any]]


def curve_fit_prep_data(
    x: _Vec[_N],
    y: _Vec[_N],
    xerr: _Vec[_N] | None,
    yerr: _Vec[_N] | None,
    p_opt: _Vec[_K] | None,
) -> tuple[_Vec[_N], _Vec[_N], _Vec[_N] | None, _Vec[_N] | None, _Vec[_K] | None]:
    """Convert ``(x, y, xerr, yerr, p_opt)`` to numpy (via to_numpy) and sort by x.

    Args:
        x: 1D x-axis data of length N; uncertainties arrays are unpacked into ``(x, xerr)``.
        y: 1D y-axis data of length N; uncertainties arrays are unpacked into ``(y, yerr)``.
        xerr: 1D errors in x, length N. Ignored if x is an uncertainties array.
        yerr: 1D errors in y, length N. Ignored if y is an uncertainties array.
        p_opt: 1D fit parameters of length K; uncertainties unpacked to nominal values.

    Returns:
        ``(x, y, xerr, yerr, p_opt)`` as plain numpy arrays, sorted by x.
    """
    x, xerr = typing.cast(tuple[_Vec[_N], _Vec[_N] | None], to_numpy(x, xerr))
    y, yerr = typing.cast(tuple[_Vec[_N], _Vec[_N] | None], to_numpy(y, yerr))
    p_opt = typing.cast(_Vec[_K] | None, to_numpy(p_opt)[0])
    idx = np.argsort(x)
    x, y = typing.cast(_Vec[_N], x[idx]), typing.cast(_Vec[_N], y[idx])
    if xerr is not None:
        xerr = typing.cast(_Vec[_N], xerr[idx])
    if yerr is not None:
        yerr = typing.cast(_Vec[_N], yerr[idx])
    return x, y, xerr, yerr, p_opt


def curve_fit_confidence_band(
    fit_fcn: Callable[..., _Vec[_N]],
    x_interp: _Vec[_N],
    p_opt: _Vec[_K],
    p_cov: _Mat[_K, _K],
    n_std: float,
) -> tuple[_Vec[_N], _Vec[_N]]:
    """Compute the lower/upper fit envelope by perturbing each parameter ±n_std·σ.

    For each parameter ``p_opt[i]``, evaluate ``fit_fcn`` at ``p_opt[i] ± n_std·σ_i``
    (with σ_i from the diagonal of ``p_cov``) and take the element-wise min/max
    across all parameter perturbations to produce the band edges.

    Args:
        fit_fcn: Model function ``f(x, *params)`` that returns a 1D vector matching x.
        x_interp: 1D x-axis values of length N at which to evaluate the band.
        p_opt: 1D best-fit parameter values of length K.
        p_cov: K×K covariance matrix; its diagonal gives the per-parameter variances.
        n_std: Number of standard deviations spanning the band.

    Returns:
        ``(fit_low, fit_high)`` as 1D arrays of length N.
    """
    p_err = np.sqrt(np.diag(p_cov))
    fit_low = np.full(x_interp.size, np.inf)
    fit_high = np.full(x_interp.size, -np.inf)
    for i, (mid, err) in enumerate(zip(p_opt, p_err)):
        p_opt_i = p_opt.copy()
        p_opt_i[i] = mid - n_std * err
        low = fit_fcn(x_interp, *p_opt_i)
        p_opt_i[i] = mid + n_std * err
        high = fit_fcn(x_interp, *p_opt_i)
        fit_low = np.minimum(fit_low, np.minimum(low, high))
        fit_high = np.maximum(fit_high, np.maximum(low, high))
    return typing.cast(_Vec[_N], fit_low), typing.cast(_Vec[_N], fit_high)
