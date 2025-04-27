import pandas as pd


class dict_(dict):
	__getattr__ = dict.get
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__


def DL_to_LD(DL: dict) -> list:
	"""
	Convert dict of lists to list of dicts

	Args:
		DL ():      Dict of lists

	Returns:

	"""

	try:
		LD = pd.DataFrame(DL).to_dict(orient="records")
	except ValueError:  # If dict values are scalars
		LD = None

	return LD


def LD_to_DL(LD: list) -> dict:
	"""
	Convert list of dicts to dict of lists

	Args:
		LD ():      List of dicts

	Returns:

	"""

	DL = pd.DataFrame(LD).to_dict(orient="list")
	return DL
