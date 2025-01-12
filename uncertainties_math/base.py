import numpy as np
import uncertainties
from uncertainties import unumpy
from uncertainties import ufloat
from uncertainties import umath


# f"{a:.2fP}" -> "a±0.01"  (P = pretty-print ±)

def __repr__(self):
	# Not putting spaces around "+/-" helps with arrays of
	# Variable, as each value with an uncertainty is a
	# block of signs (otherwise, the standard deviation can be
	# mistaken for another element of the array).

	std_dev = self.std_dev  # Optimization, since std_dev is calculated

	# A zero standard deviation is printed because otherwise,
	# ufloat_fromstr() does not correctly parse back the value
	# ("1.23" is interpreted as "1.23(1)"):

	if std_dev:
		std_dev_str = repr(std_dev)
	else:
		std_dev_str = '0'

	return "%r±%s" % (self.nominal_value, std_dev_str)


uncertainties.core.AffineScalarFunc.__repr__ = __repr__


def is_unumpy(arr):
	try:
		return any([isinstance(arr[i], uncertainties.core.AffineScalarFunc) for i in range(len(arr))])
	except TypeError:
		return False


def is_ufloat(x):
	return hasattr(x, "std_dev")


def to_numpy(x, xerr=None):
	"""
	Convert unumpy->numpy and ufloat->float. If already numpy, return as is.
	Args:
		x ():           unumpy or ufloat. In this case, don't need to provide xerr as it is already contained in x
		xerr ():        Needed only in case x is numpy instead of unumpy

	Returns:

	"""

	def unumpy_to_numpy(arr):
		val = unumpy.nominal_values(arr)
		dev = unumpy.std_devs(arr)
		if not np.any(dev):
			dev = None
		return val, dev

	def ufloat_to_float(x):
		val = uncertainties.nominal_value(x)
		dev = uncertainties.std_dev(x)
		if dev == 0:
			dev = None
		return val, dev

	if is_unumpy(x):
		x, xerr = unumpy_to_numpy(x)

	elif is_ufloat(x):
		x, xerr = ufloat_to_float(x)

	return x, xerr


def from_numpy(x, xerr=0):
	if type(x) is float:
		return ufloat(x, xerr)
	else:
		return unumpy.uarray(x, xerr)


def val(x, xerr=None):
	return to_numpy(x, xerr)[0]


def dev(x, xerr=None):
	return to_numpy(x, xerr)[1]


def make_independent(x, xerr=None):
	x, xerr = to_numpy(x, xerr)

	return unumpy.uarray(x, xerr)
