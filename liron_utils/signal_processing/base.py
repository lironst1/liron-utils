import typing

import numpy as np
import scipy.interpolate
import scipy.signal
import scipy.special

_N = typing.TypeVar("_N", bound=int)

# Shape-parameterized aliases — thread the same TypeVar across params to express "same length".
_Vec = np.ndarray[tuple[_N], np.dtype[typing.Any]]

# Fixed-dimensionality aliases (use when same-length constraints aren't expressed).
_Array1D = np.ndarray[tuple[int], np.dtype[typing.Any]]
_Array2D = np.ndarray[tuple[int, int], np.dtype[typing.Any]]

# Catch-all for opaque/arbitrary-dim arrays.
_Array = np.ndarray[typing.Any, np.dtype[typing.Any]]

meshgrid = np.mgrid
gamma = scipy.special.gamma
windows = scipy.signal.windows
sliding_window = np.lib.stride_tricks.sliding_window_view


def array(*args: typing.Any) -> typing.Any:
    """Convert any data type to ``numpy.ndarray``.

    Args:
        *args: One or more array-like inputs.

    Returns:
        A single ndarray if one argument was given, otherwise a list of ndarrays.
        ``None`` inputs are passed through unchanged.

    Examples:
        >>> x = range(5)  # x = [0, 1, 2, 3, 4]
        >>> y = array(x)  # y = np.array([0, 1, 2, 3, 4])
    """
    args_list = list(args)
    for i, arg in enumerate(args_list):
        if arg is not None:
            args_list[i] = np.asarray(arg).squeeze()
    if len(args_list) == 1:
        return args_list[0]
    return args_list


def sigmoid(x: _Array, supremum: float = 1, k: float = 1, x0: float = 0) -> _Array:
    """Sigmoid/logistic function with parameters ``supremum``, ``k``, ``x0``.

    The standard sigmoid corresponds to ``supremum=1, k=1, x0=0``.

    Args:
        x: Points at which to evaluate the sigmoid.
        supremum: Supremum / maximal asymptotic value.
        k: Steepness.
        x0: Center.

    Returns:
        The sigmoid evaluated element-wise on ``x``.
    """
    out = supremum / (1 + np.exp(-k * (x - x0)))
    return typing.cast(_Array, out)


def buffer(y: _Vec[_N], frame: int, overlap: int) -> _Array2D:
    """Buffer a 1D array into segments with overlap, mirroring MATLAB's ``buffer``.

    Args:
        y: 1D input array of length N.
        frame: Length of each segment.
        overlap: Overlap (in samples) between every two adjacent segments.

    Returns:
        2D buffered array with one segment per column.

    Raises:
        AssertionError: If ``y`` is not 1D.
    """
    assert y.ndim == 1, "Input array must be 1D"

    hop = frame - overlap
    nframes = int(np.ceil(y.size / hop))

    y_padded = np.concatenate([np.zeros(overlap), y, np.zeros(frame)])
    indices = np.sum(meshgrid[0:frame, 0 : nframes * hop : hop], axis=0)
    out = np.take(y_padded, indices, axis=0)
    return typing.cast(_Array2D, out)


def unbuffer(y: _Array, frame: int, overlap: int) -> _Array1D:
    """Inverse of :func:`buffer`: reassemble a buffered matrix into a 1D array.

    Args:
        y: Buffered matrix (2D), or a 1D array that will be tiled to ``(frame, ...)``.
        frame: Length of each segment.
        overlap: Overlap (in samples) between every two adjacent segments.

    Returns:
        1D unbuffered array.
    """
    if y.ndim == 1:
        y = np.tile(y, (frame, 1))
    assert y.ndim == 2
    frame = y.shape[0]
    out = y[min(overlap + 1, frame) - 1 :, :]
    return np.reshape(out.T, -1)


def interp1(y: _Array, n: int, axis: int = -1, **kwargs: typing.Any) -> _Array:
    """1D interpolation or extrapolation along an axis.

    Args:
        y: Input array.
        n: Desired length of output array along ``axis``.
        axis: Axis along which to interpolate.
        **kwargs: Forwarded to ``scipy.interpolate.interp1d``.

    Returns:
        Array with ``axis`` resampled to length ``n``.
    """
    if y.shape[axis] == n:
        return y

    h = scipy.interpolate.interp1d(np.linspace(0, n - 1, y.shape[axis]), y, axis=axis, **kwargs)
    out = h(range(n))
    return typing.cast(_Array, out)


def rms(x: _Array, **kwargs: typing.Any) -> _Array:
    """Root mean square along an axis.

    Args:
        x: Input array.
        **kwargs: Forwarded to ``numpy.mean`` (e.g. ``axis``, ``keepdims``).

    Returns:
        Root-mean-square of ``x``.
    """
    return typing.cast(_Array, np.sqrt(np.mean(x**2, **kwargs)))


def rescale(x: _Array, lower: float = 0, upper: float = 1) -> _Array:
    """Linearly rescale ``x`` to the interval ``[lower, upper]``.

    Args:
        x: Input array (cast to float).
        lower: Lower bound of the target interval.
        upper: Upper bound of the target interval.

    Returns:
        Rescaled array with the same shape as ``x``.
    """
    x = x.astype(float)
    x -= np.min(x)
    x /= np.max(x)
    x *= upper - lower
    x += lower
    return x
