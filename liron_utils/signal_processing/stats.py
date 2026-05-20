import typing
from collections import namedtuple

import numpy as np
import scipy.stats

from ..uncertainties_math import to_numpy, val

_N = typing.TypeVar("_N", bound=int)

_Vec = np.ndarray[tuple[_N], np.dtype[typing.Any]]

PowerDivergenceResult = namedtuple("PowerDivergenceResult", ("statistic", "pvalue"))


def chi_squared_test(
    f_exp: _Vec[_N],
    f_obs: _Vec[_N],
    f_obs_err: _Vec[_N] | None = None,
    *,
    ddof: int = 0,
    reduced: bool = False,
) -> PowerDivergenceResult:
    """Chi-squared statistic and p-value for the goodness-of-fit test (ref. [1]).

    Same as ``scipy.stats.chisquare(f_obs, f_exp, ddof)`` for the non-reduced case.

    Args:
        f_exp: 1D expected (theoretical) values of length N. May be an
            uncertainties array, in which case its uncertainties are used in
            the reduced computation.
        f_obs: 1D observed values of length N.
        f_obs_err: 1D uncertainties of length N on the observed values. Required
            when ``reduced=True``.
        ddof: Degrees of freedom adjustment (e.g. number of fitted parameters).
        reduced: If True, compute the reduced chi-squared (ref. [2]).
            - chi2 >> 1: poor model fit.
            - chi2 > 1: fit has not fully captured the data (or σ underestimated).
            - chi2 ~ 1: agreement between observations and estimates matches σ.
            - chi2 < 1: overfitting (model fitting noise, or σ overestimated).

    Returns:
        :class:`PowerDivergenceResult` with ``statistic`` (chi²) and ``pvalue``.

    References:
        [1] https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test
        [2] https://en.wikipedia.org/wiki/Reduced_chi-squared_statistic
    """
    f_exp, f_exp_err = typing.cast(tuple[_Vec[_N], _Vec[_N] | None], to_numpy(f_exp))
    f_obs, f_obs_err = typing.cast(tuple[_Vec[_N], _Vec[_N] | None], to_numpy(f_obs, f_obs_err))

    if reduced:
        assert f_obs_err is not None, "f_exp_err, f_obs_err must be provided."
        if f_exp_err is None:
            f_exp_err = np.zeros_like(f_obs_err)
        sigma2 = np.sqrt(f_exp_err**2 + f_obs_err**2)
        chi2 = np.sum((f_obs - f_exp) ** 2 / sigma2**2)
    else:
        f_exp /= f_exp.sum()
        f_obs /= f_obs.sum()
        chi2 = np.sum((f_obs - f_exp) ** 2 / f_exp)

    dof = len(f_exp) - 1 - ddof
    p_value = scipy.stats.chi2.sf(val(chi2), dof)

    return PowerDivergenceResult(chi2, p_value)
