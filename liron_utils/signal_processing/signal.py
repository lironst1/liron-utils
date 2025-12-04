import typing
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import periodogram, windows


class PowerBWResult:
    def __init__(
        self,
        bw: float,
        f_low: float,
        f_high: float,
        power_in_bw: float,
        power_total: float,
        f: np.ndarray,
        Pxx: np.ndarray,
    ):
        self.bw = bw
        self.f_low = f_low
        self.f_high = f_high
        self.power_in_bw = power_in_bw
        self.power_total = power_total
        self.f = f
        self.Pxx = Pxx


def powerbw(
    x: typing.Sequence[float],
    fs: float = 1.0,
    f: typing.Sequence[float] | None = None,
    r: float = 3.01,
    freq_lims: tuple[float, float] | None = None,
    input: typing.Literal["time", "pxx"] = "time",
    plot: bool = False,
    **periodogram_kw,
) -> PowerBWResult:
    """
    Compute the `-r` dB power bandwidth of signal x, mimicking MATLAB's powerbw(x, fs, [], r).

    Parameters
    ----------
    x : Input signal (1-D).
    fs : Sampling frequency in Hz. Default = 1.0.
    f : Frequency vector (1-D). If None, it will be computed from the input signal.
        Must be provided if input is 'pxx' and ignored otherwise.
    r : Power drop defining the bandwidth (default 3.01 dB).
    freq_lims : Two-element tuple/list [fmin, fmax] defining the frequency range
        over which to compute the reference level. Default = None (use global max).
    input : Specify the input type: 'time' for time-domain signal, 'pxx' for power spectral density.
    plot : If True, plot the PSD and shaded bandwidth region.

    Returns
    -------
    PowerBWResult
        An object with attributes:
        - bw : The computed bandwidth in Hz.
        - f_low : Lower frequency bound of the bandwidth.
        - f_high : Upper frequency bound of the bandwidth.
        - pwr_in : Power within the computed bandwidth.
        - pwr_total : Total power of the signal.
        - f : Frequency vector used in the computation.
        - Pxx : Power spectral density used in the computation.
    """

    # Ignore nans
    idx = ~np.isnan(x)
    f = np.array(f)[idx]
    x = np.array(x)[idx]

    # --- Step 1: Periodogram with Kaiser(β=0) == rectangular window
    input = input.lower()
    if input == "time":
        if f is not None:
            raise ValueError("`f` must be None if input is 'time'.")
        periodogram_kw = dict(window=windows.kaiser(len(x), beta=0)) | periodogram_kw
        f, Pxx = periodogram(
            x,
            fs=fs,
            detrend=False,
            scaling="density",
            return_onesided=False,
            **periodogram_kw,
        )
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
        f_low = interp_log(f[iL], f[iL + 1], Pxx[iL], Pxx[iL + 1], pref)
    else:
        f_low = f[0]

    if len(right) > 0:
        iR = right[0]
        f_high = interp_log(f[iR], f[iR - 1], Pxx[iR], Pxx[iR - 1], pref)
    else:
        f_high = f[-1]

    # --- Step 4: integrate power in that region
    idx_inside = (f >= f_low) & (f <= f_high)
    pwr_in = np.trapezoid(Pxx[idx_inside], f[idx_inside])
    pwr_total = np.trapezoid(Pxx, f)

    bw = f_high - f_low

    if plot:
        plt.figure()
        plt.plot(f, 10 * np.log10(Pxx), label="PSD")
        plt.axvline(f_low, color="black", linestyle="--", label="_none")
        plt.axvline(f_high, color="black", linestyle="--", label=f"bandwidth {bw:.3f} Hz")
        plt.title(f"{r}-dB Bandwidth: {bw:.3f} Hz")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power/Frequency (dB/Hz)")
        plt.legend()
        plt.grid(True)

    return PowerBWResult(
        bw=bw,
        f_low=f_low,
        f_high=f_high,
        power_in_bw=pwr_in,
        power_total=pwr_total,
        f=f,
        Pxx=Pxx,
    )
