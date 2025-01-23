import numpy as np
import scipy.stats

from ..uncertainties_math import val, to_numpy


def chi_squared_test(f_exp, f_obs,
		ddof: int = 0,
		reduced: bool = False):
	"""
	Compute the chi-squared statistic and p-value for the goodness-of-fit test.
	Same as scipy.stats.chisquare(f_obs, f_exp, ddof)

    Parameters
    ----------
    f_exp :     array-like
        Expected (theoretical) values.
    f_obs :     array-like
        Observed values.
    ddof :      int, optional
        Degrees of freedom adjustment (e.g., number of fitted parameters). Default is 0.
    reduced :   bool, optional
    	Return the reduced chi-squared statistic. Default is False.

    Returns
    -------
    chi2 : float
        Chi-squared statistic.
    p_value : float
        p-value for the goodness-of-fit test.
	"""

	f_exp = val(f_exp)
	f_obs = val(f_obs)

	chi2 = np.sum((f_obs - f_exp) ** 2 / f_exp)

	dof = len(f_exp) - 1 - ddof
	if dof <= 0:
		raise ValueError("Degrees of freedom must be greater than 0.")

	p_value = scipy.stats.chi2.sf(val(chi2), dof)

	if reduced:
		chi2 /= dof

	return chi2, p_value
