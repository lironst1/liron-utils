import numpy as np
from scipy.signal import periodogram, windows
import matplotlib.pyplot as plt


def powerbw(x, fs=1.0, rolloff_db=3.01, plot=False):
    """
    Compute the -3 dB power bandwidth of signal x, mimicking MATLAB's powerbw(x, fs).

    Parameters
    ----------
    x : array_like
        Input signal (1-D).
    fs : float, optional
        Sampling frequency in Hz. Default = 1.0.
    rolloff_db : float, optional
        Power drop defining the bandwidth (default 3.01 dB).

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
    f, Pxx = periodogram(x, fs=fs, window=windows.kaiser(len(x), 0), scaling='density', return_onesided=False)
    f = np.fft.fftshift(f)
    Pxx = np.fft.fftshift(Pxx)

    # --- Step 2: reference (peak) and -3 dB level
    i_max = np.argmax(Pxx)
    pref = Pxx[i_max] * 10 ** (-abs(rolloff_db) / 10)

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
        plt.plot(f, 10 * np.log10(Pxx), label="PSD (dB/Hz)")
        # plt.axhline(10 * np.log10(pref), xmin=f_lo, xmax=f_hi, color="gray", linestyle="--", label="-3 dB level")
        plt.axvline(f_lo, color="black", linestyle="--", label="_none")
        plt.axvline(f_hi, color="black", linestyle="--", label=f"bandwidth {bw:.2f} Hz")
        # ylim = plt.ylim()
        # plt.fill_between(f[inside], *ylim, color="gray", alpha=0.4, label="_none")
        plt.title('Power Spectral Density and -3 dB Bandwidth')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power/Frequency (dB/Hz)')
        plt.legend()
        # plt.ylim(*ylim)
        plt.grid(True)

    return bw, f_lo, f_hi, pwr, f, Pxx
