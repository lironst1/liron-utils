import numpy as np
from scipy.signal import periodogram, windows
import matplotlib.pyplot as plt


def powerbw(x, fs=1.0, f=None, r=3.01, freq_lims=None, input="time", plot=False, **periodogram_kw):
    """
    Compute the `-r` dB power bandwidth of signal x, mimicking MATLAB's powerbw(x, fs, [], r).

    Parameters
    ----------
    x : array_like
        Input signal (1-D).
    fs : float, optional
        Sampling frequency in Hz. Default = 1.0.
    f : array_like, optional
        Frequency vector (1-D). If None, it will be computed from the input signal.
        Must be provided if input is 'pxx' and ignored otherwise.
    r : float, optional
        Power drop defining the bandwidth (default 3.01 dB).
    freq_lims : (float, float) or None, optional
        Two-element tuple/list [fmin, fmax] defining the frequency range
        over which to compute the reference level. Default = None (use global max).
    input : {'time', 'pxx'}, optional
        Specify the input type: 'time' for time-domain signal, 'pxx' for power spectral density.
    plot : bool, optional
        If True, plot the PSD and shaded bandwidth region.

    Returns
    -------
    bw : float
        Bandwidth in Hz.
    flo, fhi : float
        Lower and upper frequency edges in Hz.
    pwr : float
        Total power within the 3 dB band.
    f, Pxx : ndarray
        Frequency and power-spectral-density vectors.
    """

    # --- Step 1: Periodogram with Kaiser(β=0) == rectangular window
    input = input.lower()
    if input == "time":
        if f is not None:
            raise ValueError("`f` must be None if input is 'time'.")
        periodogram_kw = dict(
                window=windows.kaiser(len(x), beta=0)
        ) | periodogram_kw
        f, Pxx = periodogram(x, fs=fs, detrend=False, scaling='density', return_onesided=False, **periodogram_kw)
        f = np.fft.fftshift(f)
        Pxx = np.fft.fftshift(Pxx)

    elif input == "pxx":
        if f is None:
            raise ValueError("`f` must be provided if input is 'pxx'.")
        elif len(f) != len(x):
            raise ValueError("`f` and `x` must have the same length.")
        Pxx = x

    else:
        raise ValueError(f"Invalid input type: {input}.")

    # --- Step 2: reference level
    if freq_lims is None:
        i_max = np.argmax(Pxx)
        meanP = Pxx[i_max]
    else:
        f1, f2 = freq_lims
        i_max = np.argmin(np.abs(f - np.mean([f1, f2])))

        idx = (f1 <= f) & (f <= f2)
        if not np.any(idx):
            raise ValueError("`freq_lims` outside PSD frequency range.")
        meanP = np.mean(Pxx[idx])

    pref = meanP * 10 ** (-abs(r) / 10)

    # --- Step 3: find crossings on each side
    left = np.where(Pxx[:i_max] <= pref)[0]
    right = np.where(Pxx[i_max:] <= pref)[0] + i_max

    # logarithmic interpolation, like MATLAB’s linterp(log10(P))
    def interp_log(f1, f2, p1, p2, pref):
        return np.interp(np.log10(pref), [np.log10(p1), np.log10(p2)], [f1, f2])

    if len(left) > 0:
        iL = left[-1]
        f_lo = interp_log(f[iL], f[iL + 1], Pxx[iL], Pxx[iL + 1], pref)
    else:
        f_lo = f[0]

    if len(right) > 0:
        iR = right[0]
        f_hi = interp_log(f[iR], f[iR - 1], Pxx[iR], Pxx[iR - 1], pref)
    else:
        f_hi = f[-1]

    # --- Step 4: integrate power in that region
    inside = (f >= f_lo) & (f <= f_hi)
    pwr = np.trapezoid(Pxx[inside], f[inside])

    bw = f_hi - f_lo

    if plot:
        plt.figure()
        plt.plot(f, 10 * np.log10(Pxx), label="PSD")
        plt.axvline(f_lo, color="black", linestyle="--", label="_none")
        plt.axvline(f_hi, color="black", linestyle="--", label=f"bandwidth {bw:.3f} Hz")
        plt.title(f"{r}-dB Bandwidth: {bw:.3f} Hz")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power/Frequency (dB/Hz)")
        plt.legend()
        plt.grid(True)

    return dict(
            bw=bw,
            f_lo=f_lo,
            f_hi=f_hi,
            pwr=pwr,
            f=f,
            Pxx=Pxx,
    )
