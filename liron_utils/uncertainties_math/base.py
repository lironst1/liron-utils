"""Helpers for converting between ``numpy`` arrays and ``uncertainties`` arrays.

Notes:
    * Printing — ``f"{a:.2fP}" -> "a±0.01"`` (``P`` selects pretty-print ±).
    * Math — ``uncertainties.umath`` and ``uncertainties.unumpy`` mirror the
      ``math``/``numpy`` APIs for scalar and array ``ufloat`` inputs.
"""

import typing

import numpy as np
import uncertainties
from uncertainties import ufloat, unumpy

_Array = np.ndarray[typing.Any, np.dtype[typing.Any]]


def __repr__(self: uncertainties.AffineScalarFunc) -> str:
    """Render a ufloat compactly as ``"nominal±std_dev"``.

    Args:
        self: An ``AffineScalarFunc`` (ufloat) instance.

    Returns:
        Compact, space-free representation suitable for arrays of ufloats.
    """
    # Not putting spaces around "+/-" helps with arrays of Variable,
    # as each value with an uncertainty is a block of signs.
    std_dev = self.std_dev
    std_dev_str = repr(std_dev) if std_dev else "0"
    return f"{self.nominal_value!r}±{std_dev_str}"


try:
    uncertainties.core.AffineScalarFunc.__repr__ = __repr__  # type: ignore[method-assign]
except ImportError:
    pass


def is_unumpy(arr: typing.Any) -> bool:
    """Return True iff ``arr`` is iterable and contains at least one ``AffineScalarFunc``."""
    try:
        return any(isinstance(a, uncertainties.AffineScalarFunc) for a in arr)
    except TypeError:
        return False


def is_ufloat(x: typing.Any) -> bool:
    """Return True iff ``x`` quacks like a ufloat (has a ``std_dev`` attribute)."""
    return hasattr(x, "std_dev")


def _unumpy_to_numpy(arr: typing.Any) -> tuple[_Array, _Array]:
    """Split an ``unumpy`` array into ``(nominal_values, std_devs)``."""
    nominal = unumpy.nominal_values(arr)
    std = unumpy.std_devs(arr)
    return nominal, std


def _ufloat_to_float(x: uncertainties.AffineScalarFunc) -> tuple[float, float]:
    """Split a scalar ufloat into ``(nominal, std_dev)``."""
    return uncertainties.nominal_value(x), uncertainties.std_dev(x)


@typing.overload
def to_numpy(
    x: typing.Any,
    xerr: None = None,
) -> tuple[typing.Any, typing.Any]: ...


@typing.overload
def to_numpy(
    x: typing.Any,
    xerr: typing.Any,
) -> tuple[typing.Any, typing.Any]: ...


def to_numpy(
    x: typing.Any,
    xerr: typing.Any = None,
) -> tuple[typing.Any, typing.Any]:
    """Convert unumpy/ufloat inputs to ``(values, errors)`` numpy arrays.

    Args:
        x: ``unumpy`` array, ``ufloat`` scalar, or already-numpy data. Uncertainty
            is already embedded when ``x`` is a ufloat/unumpy input, so ``xerr``
            is unused in that case.
        xerr: Pre-existing standard deviations; meaningful only when ``x`` is a
            plain numpy array.

    Returns:
        ``(x_values, x_errors)``; ``x_errors`` may be ``None``.
    """
    if is_unumpy(x):
        x, xerr = _unumpy_to_numpy(x)
    elif is_ufloat(x):
        x, xerr = _ufloat_to_float(typing.cast(uncertainties.AffineScalarFunc, x))
    return x, xerr


def from_numpy(
    x: typing.Any,
    xerr: float | _Array = 0,
) -> typing.Any:
    """Combine ``(values, errors)`` into a ufloat / unumpy array.

    Args:
        x: Scalar value or numpy array of nominal values.
        xerr: Matching standard deviation(s).

    Returns:
        A ``ufloat`` when ``x`` is scalar; otherwise a ``unumpy.uarray``.
    """
    if np.isscalar(x):
        x_float = float(typing.cast(typing.Any, x))
        xerr_float = float(typing.cast(typing.Any, xerr)) if np.isscalar(xerr) else 0.0
        return ufloat(x_float, xerr_float)
    return typing.cast(typing.Any, unumpy.uarray(x, xerr))


def val(x: typing.Any, xerr: typing.Any = None) -> typing.Any:
    """Return the nominal value(s) of ``x`` (see ``to_numpy``)."""
    return to_numpy(x, xerr)[0]


def dev(x: typing.Any, xerr: typing.Any = None) -> typing.Any:
    """Return the standard deviation(s) of ``x`` (see ``to_numpy``)."""
    return to_numpy(x, xerr)[1]


def uncertainty(x: typing.Any, xerr: typing.Any = None) -> typing.Any:
    """Return the relative uncertainty ``dev(x) / val(x)``."""
    return dev(x, xerr) / val(x, xerr)


def make_independent(x: typing.Any, xerr: typing.Any = None) -> typing.Any:
    """Re-wrap ``x`` so that its entries are statistically independent ufloats.

    Args:
        x: ufloat/unumpy input or numpy values.
        xerr: Standard deviations when ``x`` is plain numpy.

    Returns:
        Fresh ``unumpy.uarray`` whose correlations have been dropped.
    """
    x, xerr = to_numpy(x, xerr)
    return typing.cast(typing.Any, unumpy.uarray(x, xerr))


def histogram(
    x: typing.Any,
    xerr: typing.Any = None,
    **kwargs: typing.Any,
) -> tuple[typing.Any, _Array]:
    """Compute a histogram of ``x`` with per-bin uncertainty propagation.

    Args:
        x: Sample values, possibly as a ufloat/unumpy array (with embedded errors).
        xerr: Standard deviations when ``x`` is plain numpy.
        **kwargs: Forwarded to ``numpy.histogram``; ``density`` is honored locally
            by normalizing the resulting uncertainties array.

    Returns:
        ``(hist_with_uncertainties, bins)``.
    """
    x, xerr = to_numpy(x, xerr)

    density: bool = kwargs.pop("density", False)

    hist, bins = np.histogram(x, **kwargs)

    hist_err = np.zeros(len(hist))
    for i in range(len(hist)):
        in_bin = (bins[i] <= x) & (x < bins[i + 1])
        hist_err[i] = np.sqrt(np.sum(xerr[in_bin] ** 2))

    result = from_numpy(hist, hist_err)

    if density:
        result /= np.sum(result)

    return result, typing.cast(_Array, bins)
