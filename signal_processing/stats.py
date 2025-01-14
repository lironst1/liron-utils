import numpy as np
import scipy.stats

from .filter import movstd
from ..uncertainties_math import to_numpy


def chi_squared(f_exp, f_obs,
		f_exp_err=None, f_obs_err=None,
		ddof: int = 0,
		N: int = None):
	"""
	Compute the chi-squared statistic and p-value for the goodness-of-fit test.

    Parameters
    ----------
    f_exp :     array-like
        Expected (theoretical) values.
    f_obs :     array-like
        Observed values.
    f_exp_err : array-like, optional
    	Uncertainties in expected values
    f_obs_err : array-like, optional
        Uncertainties in observed values
    ddof :      int, optional
        Degrees of freedom adjustment (e.g., number of fitted parameters). Default is 0.
    N :         int, optional
    	Window length for the moving standard deviation.

    Returns
    -------
    chi2 : float
        Chi-squared statistic.
    p_value : float
        p-value for the goodness-of-fit test.
	"""
	assert len(f_exp) == len(f_obs), "f_exp and f_obs must have the same length."

	dof = len(f_exp) - 1 - ddof

	f_exp, f_exp_err = to_numpy(f_exp, f_exp_err)
	f_obs, f_obs_err = to_numpy(f_obs, f_obs_err)
	if f_exp_err is None:
		f_exp_err = 0
	if f_obs_err is None:
		assert N is not None, "f_obs_err must be provided if N is not None."
		f_obs_err = movstd(f_obs, N=N)

	chi2 = np.sum(((f_obs - f_exp) ** 2) / (f_exp_err ** 2 + f_obs_err ** 2))

	p_value = scipy.stats.chi2.sf(chi2, dof)

	return chi2, p_value


def reduced_chi_squared(f_exp, f_obs,
		f_exp_err=None, f_obs_err=None,
		ddof: int = 0,
		N: int = None):
	chi2, p_value = chi_squared(f_exp=f_exp, f_obs=f_obs,
			f_exp_err=f_exp_err, f_obs_err=f_obs_err, ddof=ddof, N=N)

	dof = len(f_exp) - 1 - ddof
	reduced_chi2 = chi2 / dof

	return reduced_chi2, p_value
