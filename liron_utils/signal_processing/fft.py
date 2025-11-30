import typing

import numba
import numpy as np
import scipy.fft

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


def nextpow2(a):
    """
    Exponent of next higher power of 2. Returns the exponents for the smallest powers
    of two that satisfy 2**p > a

    Parameters
    ----------
    a : array_like

    Returns
    -------
    p : array_like

    """
    if np.isscalar(a):
        if a == 0:
            p = 0
        else:
            p = int(np.ceil(np.log2(a)))
    else:
        a = np.asarray(a)
        p = np.zeros(a.shape, dtype=int)
        idx = a != 0
        p[idx] = np.ceil(np.log2(a[idx]))

    return p


@numba.njit(fastmath=True, cache=True, parallel=False)
def _dft_1d(x: np.ndarray, freqs: typing.Iterable[float], fs: float) -> np.ndarray:
    """
    Compute the DFT of a 1D real/complex signal x at arbitrary frequencies.
    """
    N = x.size
    out = np.zeros(freqs.size, dtype=np.complex128)

    for k in range(freqs.size):
        f = freqs[k]
        s = 0.0 + 0.0j
        for n in range(N):
            s += x[n] * np.exp(-2j * np.pi * f * n / fs)
        out[k] = s
    return out


def dft(x: np.ndarray, freqs: typing.Iterable[float], fs: float, axis: int = -1) -> np.ndarray:
    """
    Discrete Fourier Transform at specified frequencies.
    Equivalent to:
    >>> n = np.arange(len(x))
    >>> Exponent = np.exp(-2j * np.pi * np.outer(freqs, n) / fs)
    >>> return Exponent @ x

    Args:
        x: Input signal
        freqs: Frequencies at which to compute the DFT
        fs: Sampling frequency
        axis: Axis along which to compute the DFT

    Returns:

    """
    x = np.asarray(x)
    freqs = np.asarray(freqs)

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

    # Move back the frequency axis to the original position
    return np.moveaxis(out, -1, axis)
