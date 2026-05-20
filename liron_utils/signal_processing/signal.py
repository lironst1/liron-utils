import typing

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import periodogram, windows

_Array1D = np.ndarray[tuple[int], np.dtype[typing.Any]]


class PowerBWResult:
    """Result container returned by :func:`powerbw`."""

    def __init__(
        self,
        bw: float,
        f_low: float,
        f_high: float,
        *,
        power_in_bw: float,
        power_total: float,
        f: _Array1D,
        psd: _Array1D,
    ) -> None:
        """Store the bandwidth-search outputs.

        Args:
            bw: Computed ``-r`` dB bandwidth in Hz.
            f_low: Lower frequency bound of the bandwidth.
            f_high: Upper frequency bound of the bandwidth.
            power_in_bw: Power integrated within ``[f_low, f_high]``.
            power_total: Power integrated over the full PSD.
            f: 1D frequency axis used in the computation.
            psd: 1D PSD used in the computation.
        """
        self.bw = bw
        self.f_low = f_low
        self.f_high = f_high
        self.power_in_bw = power_in_bw
        self.power_total = power_total
        self.f = f
        self.psd = psd


def _interp_log(f1: float, f2: float, p1: float, p2: float, pref: float) -> float:
    """Logarithmic interpolation between two PSD samples, mirroring MATLAB's ``linterp(log10(P))``.

    Args:
        f1: First frequency sample.
        f2: Second frequency sample.
        p1: PSD value at ``f1``.
        p2: PSD value at ``f2``.
        pref: PSD level at which to locate the crossing.

    Returns:
        Interpolated frequency where the PSD equals ``pref``.
    """
    return typing.cast(float, np.interp(np.log10(pref), [np.log10(p1), np.log10(p2)], [f1, f2]))


def _compute_psd(
    x_arr: _Array1D,
    f_arr: _Array1D | None,
    fs: float,
    signal_type: str,
    periodogram_kw: dict[str, typing.Any],
) -> tuple[_Array1D, _Array1D]:
    """Compute (or pass-through) the frequency axis and PSD.

    Args:
        x_arr: Either a 1D time-domain signal or a pre-computed PSD.
        f_arr: 1D frequency axis matching ``x_arr``; required only for ``signal_type='pxx'``.
        fs: Sampling frequency in Hz.
        signal_type: ``"time"`` (compute periodogram) or ``"pxx"`` (pass-through).
        periodogram_kw: Forwarded to ``scipy.signal.periodogram`` when ``signal_type='time'``.

    Returns:
        ``(frequencies, psd)`` as 1D arrays, FFT-shifted to be centered at 0 Hz.

    Raises:
        ValueError: For invalid input/argument combinations.
    """
    input_lower = signal_type.lower()
    if input_lower == "time":
        if f_arr is not None:
            raise ValueError("`f` must be None if input is 'time'.")
        periodogram_kw = {"window": windows.kaiser(len(x_arr), beta=0)} | periodogram_kw
        f_out, psd = typing.cast(
            tuple[_Array1D, _Array1D],
            periodogram(
                x_arr,
                fs=fs,
                detrend=False,
                scaling="density",
                return_onesided=False,
                **periodogram_kw,
            ),
        )
        return typing.cast(_Array1D, np.fft.fftshift(f_out)), typing.cast(_Array1D, np.fft.fftshift(psd))

    if input_lower == "pxx":
        if f_arr is None:
            raise ValueError("`f` must be provided if input is 'pxx'.")
        if len(f_arr) != len(x_arr):
            raise ValueError("`f` and `x` must have the same length.")
        return f_arr, x_arr

    raise ValueError(f"Invalid input type: {signal_type}.")


def _reference_level(
    f_out: _Array1D,
    psd: _Array1D,
    freq_lims: tuple[float, float] | None,
) -> tuple[int, float]:
    """Find the reference index (peak or band center) and reference power level.

    Args:
        f_out: 1D frequency axis.
        psd: 1D PSD matching ``f_out``.
        freq_lims: Optional ``(fmin, fmax)`` band over which to average the PSD.
            If ``None``, the global PSD peak is used.

    Returns:
        ``(idx_peak, mean_power)`` — the reference index into ``f_out``/``psd``
        and the reference power level used to define the bandwidth.

    Raises:
        ValueError: If ``freq_lims`` lies entirely outside the PSD frequency range.
    """
    if freq_lims is None:
        idx_peak = int(np.argmax(psd))
        return idx_peak, float(psd[idx_peak])

    f1, f2 = freq_lims
    idx_peak = int(np.argmin(np.abs(f_out - np.mean([f1, f2]))))
    in_band = (f1 <= f_out) & (f_out <= f2)
    if not np.any(in_band):
        raise ValueError("`freq_lims` outside PSD frequency range.")
    return idx_peak, float(np.mean(psd[in_band]))


def _find_peak_band_edges(
    f_out: _Array1D,
    psd: _Array1D,
    idx_peak: int,
    pref: float,
) -> tuple[float, float]:
    """Find ``f_low`` / ``f_high`` where the PSD crosses ``pref`` on each side of the peak.

    Args:
        f_out: 1D frequency axis.
        psd: 1D PSD matching ``f_out``.
        idx_peak: Index of the reference peak in ``f_out``/``psd``.
        pref: PSD level at which to locate the band edges.

    Returns:
        ``(f_low, f_high)``. If no crossing is found on a side, the corresponding
        edge of ``f_out`` is returned.
    """
    left = np.where(psd[:idx_peak] <= pref)[0]
    right = np.where(psd[idx_peak:] <= pref)[0] + idx_peak

    if len(left) > 0:
        idx_left = left[-1]
        f_low = _interp_log(
            float(f_out[idx_left]),
            float(f_out[idx_left + 1]),
            float(psd[idx_left]),
            float(psd[idx_left + 1]),
            float(pref),
        )
    else:
        f_low = float(f_out[0])

    if len(right) > 0:
        idx_right = right[0]
        f_high = _interp_log(
            float(f_out[idx_right]),
            float(f_out[idx_right - 1]),
            float(psd[idx_right]),
            float(psd[idx_right - 1]),
            float(pref),
        )
    else:
        f_high = float(f_out[-1])

    return f_low, f_high


def _compute_band_power(f_out: _Array1D, psd: _Array1D, f_low: float, f_high: float) -> float:
    """Integrate ``psd`` between ``f_low`` and ``f_high``.

    Args:
        f_out: 1D frequency axis.
        psd: 1D PSD matching ``f_out``.
        f_low: Lower frequency bound of the band.
        f_high: Upper frequency bound of the band.

    Returns:
        Integrated power inside ``[f_low, f_high]``.
    """
    idx_inside = (f_out >= f_low) & (f_out <= f_high)
    return float(np.trapezoid(psd[idx_inside], f_out[idx_inside]))


def _compute_total_power(f_out: _Array1D, psd: _Array1D) -> float:
    """Integrate ``psd`` over the full ``f_out``.

    Args:
        f_out: 1D frequency axis.
        psd: 1D PSD matching ``f_out``.

    Returns:
        Total integrated power.
    """
    return float(np.trapezoid(psd, f_out))


def _plot_powerbw(
    f_out: _Array1D,
    psd: _Array1D,
    *,
    f_low: float,
    f_high: float,
    bw: float,
    r: float,
) -> None:
    """Plot the PSD (in dB) with vertical markers at the bandwidth edges.

    Args:
        f_out: 1D frequency axis.
        psd: 1D PSD matching ``f_out``.
        f_low: Lower frequency bound of the bandwidth.
        f_high: Upper frequency bound of the bandwidth.
        bw: Bandwidth (``f_high - f_low``) in Hz, shown in the legend.
        r: Power drop in dB shown in the title.
    """
    plt.figure()
    plt.plot(f_out, 10 * np.log10(psd), label="PSD")
    plt.axvline(f_low, color="black", linestyle="--", label="_none")
    plt.axvline(f_high, color="black", linestyle="--", label=f"bandwidth {bw:.3f} Hz")
    plt.title(f"{r}-dB Bandwidth: {bw:.3f} Hz")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power/Frequency (dB/Hz)")
    plt.legend()
    plt.grid(True)


def _prepare_psd(
    x: typing.Sequence[float],
    f: typing.Sequence[float] | None,
    fs: float,
    signal_type: str,
    periodogram_kw: dict[str, typing.Any],
) -> tuple[_Array1D, _Array1D]:
    """Drop NaNs from ``x`` (and ``f``), then compute or pass-through the PSD.

    Args:
        x: Time-domain signal or pre-computed PSD.
        f: Frequency axis matching ``x``; required only when ``signal_type='pxx'``.
        fs: Sampling frequency in Hz.
        signal_type: ``"time"`` or ``"pxx"``.
        periodogram_kw: Forwarded to :func:`_compute_psd` and onward.

    Returns:
        ``(frequencies, psd)`` after NaN removal.
    """
    x_arr = typing.cast(_Array1D, np.array(x))
    keep = ~np.isnan(x_arr)
    f_arr = typing.cast(_Array1D, np.array(f)[keep]) if f is not None else None
    return _compute_psd(typing.cast(_Array1D, x_arr[keep]), f_arr, fs, signal_type, periodogram_kw)


def powerbw(
    x: typing.Sequence[float],
    fs: float = 1.0,
    f: typing.Sequence[float] | None = None,
    r: float = 3.01,
    *,
    freq_lims: tuple[float, float] | None = None,
    signal_type: typing.Literal["time", "pxx"] = "time",
    plot: bool = False,
    **periodogram_kw: typing.Any,
) -> PowerBWResult:
    """Compute the ``-r`` dB power bandwidth of signal ``x``.

    Mirrors MATLAB's ``powerbw(x, fs, [], r)``.

    Args:
        x: 1D input signal.
        fs: Sampling frequency in Hz.
        f: 1D frequency vector. Required when ``signal_type='pxx'`` and ignored otherwise.
        r: Power drop in dB defining the bandwidth.
        freq_lims: ``(fmin, fmax)`` defining the frequency range over which to
            compute the reference level. If ``None``, the global PSD peak is used.
        signal_type: ``"time"`` (compute periodogram from ``x``) or ``"pxx"`` (treat ``x`` as a PSD).
        plot: If True, plot the PSD with markers at the bandwidth edges.
        **periodogram_kw: Forwarded to ``scipy.signal.periodogram`` when computing a PSD.

    Returns:
        :class:`PowerBWResult` with the bandwidth, band edges, integrated powers
        and the frequency axis / PSD used.
    """
    f_out, psd = _prepare_psd(x, f, fs, signal_type, periodogram_kw)
    idx_peak, mean_power = _reference_level(f_out, psd, freq_lims)
    f_low, f_high = _find_peak_band_edges(f_out, psd, idx_peak, mean_power * 10 ** (-abs(r) / 10))

    if plot:
        _plot_powerbw(f_out, psd, f_low=f_low, f_high=f_high, bw=f_high - f_low, r=r)

    return PowerBWResult(
        bw=f_high - f_low,
        f_low=f_low,
        f_high=f_high,
        power_in_bw=_compute_band_power(f_out, psd, f_low, f_high),
        power_total=_compute_total_power(f_out, psd),
        f=f_out,
        psd=psd,
    )
