import typing

import numba
import numpy as np
import scipy.fft
import scipy.linalg

_Array1D = np.ndarray[tuple[int], np.dtype[typing.Any]]
_Array = np.ndarray[typing.Any, np.dtype[typing.Any]]

# Fourier Transform
fft = scipy.fft.fft
fft2 = scipy.fft.fft2
fftn = scipy.fft.fftn
fftshift = scipy.fft.fftshift
ifft = scipy.fft.ifft
ifft2 = scipy.fft.ifft2
ifftn = scipy.fft.ifftn
ifftshift = scipy.fft.ifftshift
dft_matrix = scipy.linalg.dft  # in 'scipy.linalg._special_matrices'
fftfreq = scipy.fft.fftfreq

# todo: Fractional Fourier Transform: pip install git+ssh://git@github.com/audiolabs/python_frft.git#egg=frft


def nextpow2(a: typing.Any) -> typing.Any:
    """Exponent of the next higher power of two.

    Returns the exponents ``p`` for the smallest powers of two that satisfy ``2**p > a``.

    Args:
        a: Scalar or array-like.

    Returns:
        Scalar (when ``a`` is scalar) or integer ndarray with the same shape as ``a``.
    """
    if np.isscalar(a):
        if a == 0:
            p: typing.Any = 0
        else:
            p = int(np.ceil(np.log2(a)))
    else:
        a = np.asarray(a)
        p = np.zeros(a.shape, dtype=int)
        idx = a != 0
        p[idx] = np.ceil(np.log2(a[idx]))

    return p


@numba.njit(fastmath=True, cache=True, parallel=False)
def _dft_1d(x: _Array1D, freqs: _Array1D, fs: float) -> _Array1D:
    """Compute the DFT of a 1D real/complex signal ``x`` at arbitrary frequencies.

    Args:
        x: 1D signal of length N.
        freqs: 1D frequencies at which to evaluate the DFT.
        fs: Sampling frequency in Hz.

    Returns:
        1D complex array with the same length as ``freqs``.
    """
    n_samples = x.size
    out = np.zeros(freqs.size, dtype=np.complex128)

    for k in range(freqs.size):
        f = freqs[k]
        s = 0.0 + 0.0j
        for n in range(n_samples):
            s += x[n] * np.exp(-2j * np.pi * f * n / fs)
        out[k] = s
    return out


def dft(x: _Array, freqs: _Array1D, fs: float, axis: int = -1) -> _Array:
    """Discrete Fourier Transform at specified frequencies.

    Equivalent to::

        n = np.arange(len(x))
        Exponent = np.exp(-2j * np.pi * np.outer(freqs, n) / fs)
        return Exponent @ x

    Args:
        x: Input signal (any dim).
        freqs: 1D frequencies at which to compute the DFT.
        fs: Sampling frequency in Hz.
        axis: Axis of ``x`` along which to compute the DFT.

    Returns:
        Array with the same shape as ``x`` except that ``axis`` has length ``len(freqs)``.
    """
    x = np.asarray(x)
    freqs = typing.cast(_Array1D, np.asarray(freqs))

    # Move axis to last position
    x_moved = np.moveaxis(x, axis, -1)

    # Prepare output
    out_shape = x_moved.shape[:-1] + (freqs.size,)
    out = np.empty(out_shape, dtype=np.complex128)

    # Loop over all outer indices
    it = np.nditer(np.zeros(x_moved.shape[:-1]), flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        out[idx] = _dft_1d(x=x_moved[idx], freqs=freqs, fs=fs)
        it.iternext()

    # Move the frequency axis back to its original position
    return typing.cast(_Array, np.moveaxis(out, -1, axis))
