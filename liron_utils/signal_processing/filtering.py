# pylint: disable=too-many-lines

import typing

import numpy as np
import scipy.linalg
import scipy.ndimage.filters
import scipy.optimize
import scipy.signal

_Array1D = np.ndarray[tuple[int], np.dtype[typing.Any]]
_Array2D = np.ndarray[tuple[int, int], np.dtype[typing.Any]]
_Array = np.ndarray[typing.Any, np.dtype[typing.Any]]

# Convolution
conv = scipy.signal.convolve
conv2 = scipy.signal.convolve2d
convn = scipy.signal.convolve
deconv = scipy.signal.deconvolve
convolution_matrix = scipy.linalg.convolution_matrix  # in 'scipy.linalg._special_matrices'

# Digital filtering
lfilter = scipy.signal.lfilter  # 1-D digital filtering
filtfilt = scipy.signal.filtfilt  # Zero-phase digital filtering
sosfilt = scipy.signal.sosfilt  # Second-order sections filtering
sosfiltfilt = scipy.signal.sosfiltfilt  # Zero-phase second-order sections filtering
movmedian = scipy.ndimage.filters.median_filter
movmax = scipy.ndimage.filters.maximum_filter
movmin = scipy.ndimage.filters.minimum_filter


def filter2(
    h: _Array2D,
    x: _Array2D,
    shape: str = "full",
) -> _Array2D:
    """2D digital FIR filter via correlation.

    Args:
        h: 2D filter kernel.
        x: 2D input data.
        shape: One of ``"full"`` (full 2D filtered data), ``"valid"`` (only the
            parts computed without zero-padded edges), or ``"same"`` (central
            slice with the same size as ``x``).

    Returns:
        2D filtered data.
    """
    out = scipy.signal.convolve2d(
        typing.cast(typing.Any, x),
        typing.cast(typing.Any, np.rot90(h, 2)),
        mode=typing.cast(typing.Any, shape),
    )
    return typing.cast(_Array2D, out)


def movsum(x: _Array, window_size: int, mode: str = "same", axis: int = -1) -> _Array:
    """Moving sum filter.

    Args:
        x: Input array.
        window_size: Filter size in samples.
        mode: Convolution mode (``"full"``, ``"same"``, or ``"valid"``).
        axis: Axis along which to filter.

    Returns:
        Filtered array (same shape as ``x`` when ``mode='same'``).
    """
    kernel = np.ones(window_size)

    def _convolve(m: _Array) -> _Array:
        result = np.convolve(typing.cast(typing.Any, m), kernel, mode=mode)  # type: ignore[call-overload]
        return typing.cast(_Array, result)

    return typing.cast(_Array, np.apply_along_axis(_convolve, axis=axis, arr=x))


def movmean(x: _Array, window_size: int, mode: str = "same", axis: int = -1) -> _Array:
    """Moving average filter (see :func:`movsum`)."""
    return typing.cast(_Array, movsum(x, window_size, mode=mode, axis=axis) / window_size)


def movrms(x: _Array, window_size: int, mode: str = "same", axis: int = -1) -> _Array:
    """Moving RMS filter (see :func:`movsum`)."""
    return typing.cast(_Array, np.sqrt(movmean(x**2, window_size, mode=mode, axis=axis)))


def movvar(
    x: _Array,
    window_size: int,
    ddof: int = 1,
    mode: str = "same",
    axis: int = -1,
) -> _Array:
    """Moving variance filter (Bessel-corrected by ``ddof``)."""
    out = movmean(x**2, window_size, mode=mode, axis=axis) - movmean(x, window_size, mode=mode, axis=axis) ** 2
    out *= window_size / (window_size - ddof)
    return typing.cast(_Array, out)


def movstd(x: _Array, window_size: int, mode: str = "same", axis: int = -1) -> _Array:
    """Moving standard deviation filter (square root of :func:`movvar`)."""
    return typing.cast(_Array, np.sqrt(movvar(x, window_size, mode=mode, axis=axis)))


def _window_time_domain_metrics(window_samples: _Array1D, fs: float | None) -> dict[str, typing.Any]:
    """Time-domain window metrics: coherent gain and ENBW.

    Args:
        window_samples: 1D window samples.
        fs: Sampling rate; ``None`` returns ENBW only in bins.

    Returns:
        Dict with ``coherent_gain``, ``enbw_bins`` and ``enbw_hz`` (``None`` if ``fs is None``).
    """
    n_samples = window_samples.size
    coherent_gain = window_samples.mean()
    enbw_bins = np.mean(window_samples**2) / (np.mean(window_samples) ** 2)
    enbw_hz = enbw_bins * fs / n_samples if fs is not None else None
    return {
        "coherent_gain": coherent_gain,
        "enbw_bins": enbw_bins,
        "enbw_hz": enbw_hz,
    }


def _first_null_index(magnitude: _Array1D) -> int:
    """Index of the first magnitude null after DC.

    Args:
        magnitude: 1D non-negative magnitude response.

    Returns:
        Index of the first local minimum (falls back to the first
        increasing-derivative bin if no minima are found).
    """
    minima_idx, _ = scipy.signal.find_peaks(-magnitude)
    if minima_idx.size:
        return int(minima_idx[0])
    deriv = np.diff(magnitude)
    return int(np.argmax(deriv > 0) or 1)


def _interpolate_3db_crossing(freqs: _Array1D, magnitude_db: _Array1D, k_null: int) -> float:
    """Interpolate the -3 dB crossing on the dB scale, before the first null.

    Args:
        freqs: 1D frequency axis.
        magnitude_db: 1D magnitude response in dB.
        k_null: Index of the first null.

    Returns:
        Frequency at which the magnitude crosses -3 dB, or ``nan`` if no crossing.
    """
    k_hp = np.where(magnitude_db[: k_null + 1] <= -3.0)[0]
    if not k_hp.size:
        return np.nan
    k2 = k_hp[0]
    k1 = max(k2 - 1, 0)
    y1, y2 = magnitude_db[k1], magnitude_db[k2]
    x1, x2 = freqs[k1], freqs[k2]
    if y2 == y1:
        return float(x2)
    return float(x1 + (x2 - x1) * ((-3.0 - y1) / (y2 - y1)))


def _window_sidelobe_rolloff(
    freqs: _Array1D,
    magnitude_db: _Array1D,
    side_peaks: _Array1D,
    f_null: float,
) -> tuple[float, float]:
    """Linear regression of sidelobe *peak* levels vs ``log10(freq)``.

    Args:
        freqs: 1D frequency axis.
        magnitude_db: 1D magnitude response in dB.
        side_peaks: 1D integer indices of sidelobe peaks in ``freqs``/``magnitude_db``.
        f_null: First null frequency; peaks at or below this are excluded.

    Returns:
        ``(slope_db_per_decade, slope_db_per_octave)``, or ``(nan, nan)`` when
        fewer than two valid sidelobe peaks are available.
    """
    if side_peaks.size < 2:
        return np.nan, np.nan
    fpk = freqs[side_peaks]
    ypk = magnitude_db[side_peaks]
    good = (fpk > f_null) & np.isfinite(ypk) & (ypk < -1)  # exclude main-lobe skirt
    fpk, ypk = fpk[good], ypk[good]
    if fpk.size < 2:
        return np.nan, np.nan
    slope, _ = np.polyfit(np.log10(fpk), ypk, 1)
    return float(slope), float(slope * np.log10(2.0))


def _window_sidelobe_levels(
    magnitude_db: _Array1D,
    side_peaks: _Array1D,
) -> tuple[float, float]:
    """Peak and first sidelobe levels (dB).

    Args:
        magnitude_db: 1D magnitude response in dB.
        side_peaks: 1D integer indices of sidelobe peaks.

    Returns:
        ``(peak_sidelobe_db, first_sidelobe_db)``, or ``(nan, nan)`` if no sidelobes.
    """
    if not side_peaks.size:
        return np.nan, np.nan
    return float(magnitude_db[side_peaks].max()), float(magnitude_db[side_peaks[0]])


def _window_integrated_sidelobe_level(magnitude: _Array1D, k_null: int) -> float:
    """Integrated sidelobe level in dB (power ratio sidelobes / main-lobe).

    Args:
        magnitude: 1D (linear) magnitude response, normalized to peak 1.
        k_null: Index of the first null.

    Returns:
        ``10 * log10(power_sidelobes / power_main)``, or ``nan`` if the main-lobe
        power is non-positive.
    """
    power = magnitude**2
    power_main = power[: k_null + 1].sum()
    power_side = power[k_null + 1 :].sum()
    if power_main <= 0:
        return np.nan
    return float(10 * np.log10(power_side / power_main))


def _window_freqz(
    window_samples: _Array1D,
    num_freq_points: int,
    fs: float | None,
) -> tuple[_Array1D, _Array1D, _Array1D]:
    """Compute the normalized magnitude (linear and dB) and frequency axis of a window.

    Args:
        window_samples: 1D window samples.
        num_freq_points: Number of frequency points evaluated by ``scipy.signal.freqz``.
        fs: Sampling rate; ``None`` returns a Nyquist-normalized axis (Nyquist = 1).

    Returns:
        ``(freqs, magnitude, magnitude_db)`` as 1D arrays of length ``num_freq_points``,
        with ``magnitude`` normalized to peak 1 and ``magnitude_db = 20·log10(magnitude)``.
    """
    w_freq, freq_response = scipy.signal.freqz(window_samples, worN=num_freq_points, whole=False)
    freqs = w_freq / np.pi
    if fs is not None:
        freqs *= fs / 2
    magnitude = np.abs(freq_response)
    magnitude /= magnitude.max()
    magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-300))
    return (
        typing.cast(_Array1D, freqs),
        typing.cast(_Array1D, magnitude),
        typing.cast(_Array1D, magnitude_db),
    )


def _window_frequency_domain_metrics(
    window_samples: _Array1D,
    num_freq_points: int,
    fs: float | None,
) -> dict[str, typing.Any]:
    """Frequency-domain window metrics: response, lobe widths, sidelobe levels, rolloff.

    Args:
        window_samples: 1D window samples.
        num_freq_points: Number of frequency points used by ``scipy.signal.freqz``.
        fs: Sampling rate; ``None`` produces a Nyquist-normalized axis.

    Returns:
        Dict including ``freqs``, ``magnitude_db``, ``k_null``, ``f_null``,
        ``main_lobe_width``, ``main_lobe_width_3db``, ``f_3db``, ``psl_db``,
        ``fsl_db``, ``isl_db``, ``rolloff_db_per_decade``, ``rolloff_db_per_octave``,
        and ``side_peaks``.
    """
    freqs, magnitude, magnitude_db = _window_freqz(window_samples, num_freq_points, fs)
    k_null = _first_null_index(magnitude)
    f_null = freqs[k_null]
    f_3db = _interpolate_3db_crossing(freqs, magnitude_db, k_null)

    peaks_idx, _ = scipy.signal.find_peaks(magnitude)
    side_peaks = typing.cast(_Array1D, peaks_idx[freqs[peaks_idx] > f_null])
    psl_db, fsl_db = _window_sidelobe_levels(magnitude_db, side_peaks)
    rolloff_per_decade, rolloff_per_octave = _window_sidelobe_rolloff(freqs, magnitude_db, side_peaks, f_null)

    return {
        "freqs": freqs,
        "magnitude_db": magnitude_db,
        "k_null": k_null,
        "f_null": f_null,
        "main_lobe_width": 2 * f_null,
        "main_lobe_width_3db": 2 * f_3db if np.isfinite(f_3db) else np.nan,
        "f_3db": f_3db,
        "psl_db": psl_db,
        "fsl_db": fsl_db,
        "isl_db": _window_integrated_sidelobe_level(magnitude, k_null),
        "rolloff_db_per_decade": rolloff_per_decade,
        "rolloff_db_per_octave": rolloff_per_octave,
        "side_peaks": side_peaks,
    }


def _plot_window_time_domain(
    ax: typing.Any,
    window: str | tuple[typing.Any, ...] | _Array1D,
    window_samples: _Array1D,
) -> None:
    """Render the time-domain subplot for window analysis."""
    n_samples = window_samples.size
    ax.plot(np.arange(n_samples), window_samples)
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Time Domain — {_window_label(window, n_samples)}")
    ax.set_ylim(0, window_samples.max() * 1.05)


def _plot_window_frequency_domain(
    ax: typing.Any,
    freq_metrics: dict[str, typing.Any],
    metrics_summary: dict[str, typing.Any],
    fs: float | None,
) -> None:
    """Render the frequency-domain subplot for window analysis."""
    freqs = freq_metrics["freqs"]
    magnitude_db = freq_metrics["magnitude_db"]
    f_null = freq_metrics["f_null"]
    f_3db = freq_metrics["f_3db"]
    side_peaks = freq_metrics["side_peaks"]
    k_null = freq_metrics["k_null"]
    unit_suffix = "" if fs is None else "Hz"

    ax.plot(freqs, magnitude_db)
    ax.set_xlabel("Frequency (Hz)" if fs is not None else "Normalized frequency (× Nyquist)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("Frequency Domain")
    ax.grid(True)

    ax.axvline(+f_null, linestyle="--", linewidth=0.5)
    ax.axvline(-f_null, linestyle="--", linewidth=0.5)
    ax.text(f_null, -6, rf"$\text{{MLW}}={freq_metrics['main_lobe_width']:.3f}${unit_suffix}", ha="left", va="top")

    if np.isfinite(f_3db):
        ax.axvline(f_3db, linestyle=":")
        ax.text(
            f_3db,
            -3,
            rf"$\text{{MLW}}_{{-3\text{{dB}}}}={freq_metrics['main_lobe_width_3db']:.3f}${unit_suffix}",
            ha="left",
            va="bottom",
        )

    if side_peaks.size:
        k_psl = side_peaks[np.argmax(magnitude_db[side_peaks])]
        ax.plot(freqs[k_psl], magnitude_db[k_psl], marker="o")
        ax.text(
            freqs[k_psl],
            magnitude_db[k_psl],
            rf"PSL$={magnitude_db[k_psl]:.1f}$dB",
            ha="left",
            va="bottom",
        )

    ax.fill_between(freqs[k_null + 1 :], magnitude_db[k_null + 1 :], -300, alpha=0.08)

    metrics_text = "\n".join([f"{k}={v:.3f}" for k, v in metrics_summary.items() if v is not None])
    ax.text(
        0.98,
        0.98,
        metrics_text,
        fontsize="x-small",
        ha="right",
        va="top",
        transform=ax.transAxes,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
    )
    ax.set_xlim(0, 1 if fs is None else fs / 2)
    ax.set_ylim(-150, 5)


def _plot_window_analysis(
    window: str | tuple[typing.Any, ...] | _Array1D,
    window_samples: _Array1D,
    freq_metrics: dict[str, typing.Any],
    metrics_summary: dict[str, typing.Any],
    fs: float | None,
) -> None:
    """Render the time-domain and frequency-domain window analysis plots."""
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    _, axs = plt.subplots(1, 2, figsize=(12, 6))
    _plot_window_time_domain(axs[0], window, window_samples)
    _plot_window_frequency_domain(axs[1], freq_metrics, metrics_summary, fs)
    plt.tight_layout()
    plt.show(block=False)


def _window_label(window: str | tuple[typing.Any, ...] | _Array1D, n_samples: int) -> str:
    """Build a human-readable label for a window spec.

    Args:
        window: SciPy window spec (string or tuple) or a 1D sample array.
        n_samples: Window length.

    Returns:
        Short label suitable for plot titles.
    """
    if isinstance(window, str):
        return f"{window}({n_samples=})"
    if isinstance(window, tuple) and len(window) >= 1:
        name = str(window[0])
        params = ", ".join(f"{p:g}" if isinstance(p, (int, float)) else str(p) for p in window[1:])
        return f"{name}({n_samples=}, {params})" if params else name
    return f"{n_samples=}"


def analyze_window(
    window: str | tuple[typing.Any, ...] | _Array1D,
    n_samples: int = 64,
    fs: float | None = None,
    num_freq_points: int = 2**17,
    *,
    fftbins: bool = True,
    plot: bool = False,
) -> dict[str, typing.Any]:
    """Compute key metrics (and optionally plot) for a window function.

    Args:
        window: SciPy window spec for ``get_window`` (e.g. ``"hann"``,
            ``("kaiser", 8.6)``) or a 1D array of samples.
        n_samples: Window length to synthesize when ``window`` is a spec.
            Ignored when ``window`` is array-like.
        fs: Sampling rate in Hz. When given, frequency-related values are in Hz.
        num_freq_points: FFT length for high-resolution spectral measurements
            (use large, e.g. ``2**17``).
        fftbins: If True, create a "periodic" window ready to use with
            ``ifftshift`` and to be multiplied by the result of an FFT. If False,
            create a "symmetric" window for use in filter design.
        plot: If True, produce a magnitude plot with annotations.

    Returns:
        Dict with short, MATLAB-ish keys:
            - ``CG``: Coherent gain (mean window amplitude).
            - ``ENBW_bins``: Equivalent noise bandwidth in DFT bins.
            - ``ENBW_Hz``: ENBW in Hz (only if ``fs`` is provided).
            - ``MLW``: Main-lobe width between first nulls (normalized, Nyquist=1).
            - ``MLW_3dB``: -3 dB bandwidth (normalized, Nyquist=1).
            - ``PSL_dB``: Peak sidelobe level in dB.
            - ``FSL_dB``: First sidelobe level in dB.
            - ``ISL_dB``: Integrated sidelobe level in dB.
            - ``RO_dBdec`` / ``RO_dBoct``: Sidelobe roll-off slope (dB/decade and dB/octave).
            - ``f_null``, ``f_3dB``: Normalized frequencies of key points.

    Notes:
        - ``ENBW_bins = mean(w^2) / mean(w)^2``.
        - ``ISL`` is the single-sided ``10·log10(Σ|H|² sidelobes / Σ|H|² main-lobe)``;
          the ratio is invariant to bin width.
        - Roll-off is estimated by linear regression of sidelobe *peak* levels
          vs ``log10(f)``.
    """
    if isinstance(window, (str, tuple)):
        window_samples = typing.cast(
            _Array1D,
            scipy.signal.get_window(typing.cast(typing.Any, window), n_samples, fftbins=fftbins),
        )
    else:
        window_samples = typing.cast(_Array1D, np.asarray(window, dtype=float))
        window_samples /= window_samples.max()  # normalize to unity gain

    time_metrics = _window_time_domain_metrics(window_samples, fs)
    freq_metrics = _window_frequency_domain_metrics(window_samples, num_freq_points, fs)

    out: dict[str, typing.Any] = {
        "CG": time_metrics["coherent_gain"],
        "ENBW_bins": time_metrics["enbw_bins"],
        "ENBW_Hz": time_metrics["enbw_hz"],
        "MLW": freq_metrics["main_lobe_width"],
        "MLW_3dB": freq_metrics["main_lobe_width_3db"],
        "PSL_dB": freq_metrics["psl_db"],
        "FSL_dB": freq_metrics["fsl_db"],
        "ISL_dB": freq_metrics["isl_db"],
        "RO_dBdec": freq_metrics["rolloff_db_per_decade"],
        "RO_dBoct": freq_metrics["rolloff_db_per_octave"],
        "f_null": freq_metrics["f_null"],
        "f_3dB": freq_metrics["f_3db"],
    }

    if plot:
        _plot_window_analysis(window, window_samples, freq_metrics, out, fs)

    return out


def _filter_freq_response(
    b_arr: _Array1D,
    a_arr: _Array1D,
    sos_arr: _Array2D | None,
    num_freq_points: int,
    fs: float | None,
) -> tuple[_Array1D, _Array1D]:
    """Compute the frequency response (frequencies and complex response).

    Args:
        b_arr: 1D numerator coefficients.
        a_arr: 1D denominator coefficients.
        sos_arr: 2D second-order sections ``(n_sections, 6)``, or ``None`` for transfer-function form.
        num_freq_points: Number of frequency points to evaluate.
        fs: Sampling rate; ``None`` returns a Nyquist-normalized axis.

    Returns:
        ``(freqs, complex_response)`` as 1D arrays.
    """
    if fs is None:  # Normalized frequency [0..1], Nyquist=1
        if sos_arr is not None:
            w, freq_response = typing.cast(
                tuple[_Array1D, _Array1D],
                scipy.signal.sosfreqz(sos_arr, worN=num_freq_points, whole=False),
            )
        else:
            w, freq_response = typing.cast(
                tuple[_Array1D, _Array1D],
                scipy.signal.freqz(b_arr, a_arr, worN=num_freq_points, whole=False),
            )
        freqs = typing.cast(_Array1D, w / np.pi)
    elif sos_arr is not None:
        freqs, freq_response = typing.cast(
            tuple[_Array1D, _Array1D],
            scipy.signal.sosfreqz(sos_arr, worN=num_freq_points, fs=fs),
        )
    else:
        freqs, freq_response = typing.cast(
            tuple[_Array1D, _Array1D],
            scipy.signal.freqz(b_arr, a_arr, worN=num_freq_points, fs=fs),
        )

    return freqs, freq_response


def _filter_passband_metrics(
    freqs: _Array1D,
    magnitude_db: _Array1D,
    passband_tol: float,
) -> dict[str, typing.Any]:
    """Find the passband edge, -3 dB bandwidth, and passband ripple.

    Args:
        freqs: 1D frequency axis.
        magnitude_db: 1D magnitude response in dB.
        passband_tol: Passband edge tolerance in dB (e.g. 3 dB).

    Returns:
        Dict with ``f_pass``, ``bw_3dB`` (alias of ``f_pass``), and ``ripple_pass``.
    """
    idx_pass = np.argmax(magnitude_db <= -passband_tol)
    f_pass = freqs[idx_pass] if idx_pass > 0 else np.nan
    ripple_pass = (magnitude_db[:idx_pass].max() - magnitude_db[:idx_pass].min()) if idx_pass > 0 else np.nan
    return {"f_pass": f_pass, "bw_3dB": f_pass, "ripple_pass": ripple_pass}


def _filter_stopband_metrics(
    freqs: _Array1D,
    magnitude_db: _Array1D,
    stopband_tol: float,
) -> dict[str, typing.Any]:
    """Find the stopband edge and attenuation.

    Args:
        freqs: 1D frequency axis.
        magnitude_db: 1D magnitude response in dB.
        stopband_tol: Stopband tolerance in dB (e.g. 60 dB).

    Returns:
        Dict with ``f_stop`` and ``atten_stop``.
    """
    idx_stop = np.argmax(magnitude_db <= -stopband_tol)
    f_stop = freqs[idx_stop] if idx_stop > 0 else np.nan
    atten_stop = -magnitude_db[idx_stop:].max() if idx_stop > 0 else np.nan
    return {"f_stop": f_stop, "atten_stop": atten_stop}


def _filter_stability_and_roots(
    b_arr: _Array1D,
    a_arr: _Array1D,
    sos_arr: _Array2D | None,
) -> tuple[bool | np.bool_, _Array1D, _Array1D]:
    """Compute filter stability and its zeros / poles.

    Args:
        b_arr: 1D numerator coefficients.
        a_arr: 1D denominator coefficients.
        sos_arr: 2D second-order sections ``(n_sections, 6)``, or ``None`` for transfer-function form.

    Returns:
        ``(stable, zeros, poles)`` — ``stable`` is True iff every pole lies inside
        the unit circle; ``zeros``/``poles`` are 1D arrays.
    """
    if sos_arr is None:
        zeros = typing.cast(_Array1D, np.roots(b_arr))
        poles = typing.cast(_Array1D, np.roots(a_arr))
        return np.all(np.abs(poles) < 1), zeros, poles

    all_zeros: list[typing.Any] = []
    all_poles: list[typing.Any] = []
    stability: bool | np.bool_ = True
    for section in sos_arr:
        b_sec = section[:3]
        a_sec = section[3:]
        if a_sec[0] != 0:
            a_sec = a_sec / a_sec[0]
            b_sec = b_sec / section[3]
        zeros_sec = np.roots(b_sec)
        poles_sec = np.roots(a_sec)
        all_zeros.extend(zeros_sec)
        all_poles.extend(poles_sec)
        if np.any(np.abs(poles_sec) >= 1):
            stability = False
    return stability, typing.cast(_Array1D, np.array(all_zeros)), typing.cast(_Array1D, np.array(all_poles))


def _filter_impulse_response(
    b_arr: _Array1D,
    a_arr: _Array1D,
    sos_arr: _Array2D | None,
    ftype: str,
    order: int,
) -> _Array1D:
    """Compute the impulse response of a filter.

    Args:
        b_arr: 1D numerator coefficients.
        a_arr: 1D denominator coefficients.
        sos_arr: 2D second-order sections, or ``None`` for transfer-function form.
        ftype: ``"FIR"`` or ``"IIR"``. FIR shortcut returns ``b_arr`` directly.
        order: Filter order (used to size the impulse for IIR filters).

    Returns:
        1D impulse response.
    """
    if ftype == "FIR":
        return b_arr
    n_imp = max(200, order * 4)
    impulse = np.zeros(n_imp)
    impulse[0] = 1
    if sos_arr is not None:
        return typing.cast(_Array1D, scipy.signal.sosfilt(typing.cast(typing.Any, sos_arr), impulse))
    return typing.cast(_Array1D, scipy.signal.lfilter(b_arr, a_arr, impulse))


def _plot_impulse_subplot(ax: typing.Any, h: _Array1D, is_sos: bool, n_sections: int | None) -> None:
    """Render the impulse-response subplot."""
    ax.stem(np.arange(len(h)), h)
    title = "Impulse Response"
    if is_sos:
        title += f" (SOS: {n_sections} sections)"
    ax.set_title(title)
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    ax.grid(True)


def _plot_pole_zero_subplot(ax: typing.Any, zeros: _Array1D, poles: _Array1D) -> None:
    """Render the pole-zero subplot."""
    import matplotlib.patches  # pylint: disable=import-outside-toplevel

    ax.scatter(np.real(zeros), np.imag(zeros), marker="o", label=f"{len(zeros)} Zeros")
    ax.scatter(np.real(poles), np.imag(poles), marker="x", label=f"{len(poles)} Poles")
    ax.add_patch(matplotlib.patches.Circle((0, 0), 1, color="black", fill=False, ls="--"))
    ax.set_title("Pole-Zero Plot")
    ax.set_xlabel(r"$\Re{\{z\}}$")
    ax.set_ylabel(r"$\Im{\{z\}}$")
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True)


def _plot_magnitude_subplot(
    ax: typing.Any,
    freqs: _Array1D,
    magnitude_db: _Array1D,
    *,
    fs: float | None,
    xscale: str,
    f_pass: float,
    f_stop: float,
) -> None:
    """Render the magnitude-response subplot."""
    ax.plot(freqs, magnitude_db)
    ax.set_title("Magnitude Response")
    ax.set_xscale(xscale)
    ax.set_xlabel("Normalized Frequency (× Nyquist)" if fs is None else "Frequency [Hz]")
    ax.set_ylabel("Magnitude [dB]")
    ax.grid(True)
    if np.isfinite(f_pass):
        ax.axvline(f_pass, ls="--", c="r", label=f"Passband edge {f_pass:.3g}")
    if np.isfinite(f_stop):
        ax.axvline(f_stop, ls="--", c="g", label=f"Stopband edge {f_stop:.3g}")
    ax.legend()


def _plot_phase_subplot(
    ax: typing.Any,
    freqs: _Array1D,
    freq_response: _Array1D,
    *,
    fs: float | None,
    xscale: str,
    share_x_with: typing.Any,
) -> None:
    """Render the phase-response subplot."""
    ax.plot(freqs, np.rad2deg(np.unwrap(np.angle(freq_response))))
    ax.set_title("Phase Response")
    ax.set_ylabel("Phase [deg]")
    ax.set_xscale(xscale)
    ax.sharex(share_x_with)
    ax.set_xlabel("Normalized Frequency (× Nyquist)" if fs is None else "Frequency [Hz]")
    ax.grid(True)


def _plot_filter_analysis(
    filter_repr: dict[str, typing.Any],
    response: dict[str, typing.Any],
    edges: dict[str, float],
    fs: float | None,
    xscale: str,
) -> None:
    """Render the impulse, pole-zero, magnitude, and phase plots for a filter.

    Args:
        filter_repr: Output of :func:`_parse_filter_representation`.
        response: Output of :func:`_compute_filter_response`, augmented with
            ``stability``, ``zeros`` and ``poles``.
        edges: Output of :func:`_compute_filter_edges`.
        fs: Sampling rate; ``None`` uses Nyquist-normalized axes.
        xscale: Matplotlib x-axis scale (e.g. ``"log"`` or ``"linear"``).
    """
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    _, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()

    h = _filter_impulse_response(
        filter_repr["b_arr"],
        filter_repr["a_arr"],
        filter_repr["sos_arr"],
        filter_repr["ftype"],
        filter_repr["order"],
    )
    _plot_impulse_subplot(axs[0], h, filter_repr["sos_arr"] is not None, filter_repr["n_sections"])
    _plot_pole_zero_subplot(axs[1], response["zeros"], response["poles"])
    _plot_magnitude_subplot(
        axs[2],
        response["freqs"],
        response["magnitude_db"],
        fs=fs,
        xscale=xscale,
        f_pass=edges["f_pass"],
        f_stop=edges["f_stop"],
    )
    _plot_phase_subplot(
        axs[3],
        response["freqs"],
        response["freq_response"],
        fs=fs,
        xscale=xscale,
        share_x_with=axs[2],
    )

    plt.tight_layout()
    plt.show(block=False)


def _parse_filter_representation(
    b: _Array1D | float | None,
    a: _Array1D | float,
    sos: _Array2D | None,
) -> dict[str, typing.Any]:
    """Validate filter inputs and normalize them to ``b``/``a``/``sos`` arrays with order and type.

    Args:
        b: 1D numerator coefficients or a scalar. Mutually exclusive with ``sos``.
        a: 1D denominator coefficients or a scalar.
        sos: 2D second-order sections ``(n_sections, 6)``. Mutually exclusive with ``b``.

    Returns:
        Dict with ``b_arr``, ``a_arr``, ``sos_arr``, ``n_sections``, ``order``, ``ftype``.

    Raises:
        ValueError: If neither or both of ``b``/``sos`` are given, or if ``sos`` has wrong shape.
    """
    if not ((b is None) ^ (sos is None)):  # pylint: disable=superfluous-parens
        raise ValueError("Either `b` or `sos` must be provided.")

    if b is None:
        sos_arr = typing.cast(_Array2D, np.asarray(sos))
        if sos_arr.ndim != 2 or sos_arr.shape[1] != 6:
            raise ValueError("`sos` must have shape (n_sections, 6)")
        n_sections = sos_arr.shape[0]
        b_arr, a_arr = typing.cast(tuple[_Array1D, _Array1D], scipy.signal.sos2tf(sos_arr))
        b_arr, a_arr = (
            typing.cast(_Array1D, np.atleast_1d(b_arr)),
            typing.cast(_Array1D, np.atleast_1d(a_arr)),
        )
        return {
            "b_arr": b_arr,
            "a_arr": a_arr,
            "sos_arr": sos_arr,
            "n_sections": n_sections,
            "order": n_sections * 2,
            "ftype": "IIR",
        }

    b_arr = typing.cast(_Array1D, np.atleast_1d(b))
    a_arr = typing.cast(_Array1D, np.atleast_1d(a))
    return {
        "b_arr": b_arr,
        "a_arr": a_arr,
        "sos_arr": None,
        "n_sections": None,
        "order": max(len(b_arr), len(a_arr)) - 1,
        "ftype": "FIR" if np.allclose(a_arr, [1]) else "IIR",
    }


def _filter_group_delay(
    filter_repr: dict[str, typing.Any],
    num_freq_points: int,
    fs: float | None,
) -> tuple[float, float]:
    """Compute mean and variance of the filter group delay over finite samples.

    Args:
        filter_repr: Output of :func:`_parse_filter_representation`.
        num_freq_points: Number of frequency samples used by ``scipy.signal.group_delay``.
        fs: Sampling rate; ``None`` evaluates the response on ``[0, 2π]``.

    Returns:
        ``(mean_group_delay, variance_group_delay)`` in samples.
    """
    sos_arr = filter_repr["sos_arr"]
    system: tuple[_Array, _Array] = (
        (sos_arr, typing.cast(_Array, np.array([])))
        if sos_arr is not None
        else (filter_repr["b_arr"], filter_repr["a_arr"])
    )
    _, gd = typing.cast(
        tuple[_Array1D, _Array1D],
        scipy.signal.group_delay(system, w=num_freq_points, fs=(fs if fs else 2 * np.pi)),
    )
    finite = gd[np.isfinite(gd)]
    return float(np.mean(finite)), float(np.var(finite))


def _compute_filter_response(
    filter_repr: dict[str, typing.Any],
    num_freq_points: int,
    fs: float | None,
    eps: float,
) -> dict[str, typing.Any]:
    """Compute the frequency response and magnitude in dB.

    Args:
        filter_repr: Output of :func:`_parse_filter_representation`.
        num_freq_points: Number of frequency points evaluated by ``scipy.signal.freqz`` / ``sosfreqz``.
        fs: Sampling rate; ``None`` returns a Nyquist-normalized axis.
        eps: Floor added before ``log10`` to avoid ``log(0)``.

    Returns:
        Dict with ``freqs``, ``freq_response``, ``magnitude_db``.
    """
    freqs, freq_response = _filter_freq_response(
        filter_repr["b_arr"],
        filter_repr["a_arr"],
        filter_repr["sos_arr"],
        num_freq_points,
        fs,
    )
    magnitude_db = typing.cast(_Array1D, 20 * np.log10(np.abs(freq_response) + eps))
    return {"freqs": freqs, "freq_response": freq_response, "magnitude_db": magnitude_db}


def _compute_filter_edges(
    freqs: _Array1D,
    magnitude_db: _Array1D,
    passband_tol: float,
    stopband_tol: float,
) -> dict[str, float]:
    """Compute passband / stopband edges plus the transition-width metric.

    Args:
        freqs: 1D frequency axis.
        magnitude_db: 1D magnitude response in dB.
        passband_tol: Passband edge tolerance in dB.
        stopband_tol: Stopband edge tolerance in dB.

    Returns:
        Dict with ``f_pass``, ``bw_3dB``, ``ripple_pass``, ``f_stop``, ``atten_stop``, ``tw``.
    """
    passband = _filter_passband_metrics(freqs, magnitude_db, passband_tol)
    stopband = _filter_stopband_metrics(freqs, magnitude_db, stopband_tol)
    tw = (
        stopband["f_stop"] - passband["f_pass"]
        if np.isfinite(passband["f_pass"]) and np.isfinite(stopband["f_stop"])
        else np.nan
    )
    return {
        "f_pass": passband["f_pass"],
        "bw_3dB": passband["bw_3dB"],
        "ripple_pass": passband["ripple_pass"],
        "f_stop": stopband["f_stop"],
        "atten_stop": stopband["atten_stop"],
        "tw": tw,
    }


def _build_filter_analysis_result(
    filter_repr: dict[str, typing.Any],
    response: dict[str, typing.Any],
    edges: dict[str, float],
    gd_mean: float,
    gd_var: float,
) -> dict[str, typing.Any]:
    """Combine the filter-analysis pieces into the user-facing result dict.

    Args:
        filter_repr: Output of :func:`_parse_filter_representation`.
        response: Output of :func:`_compute_filter_response` (with ``stability``).
        edges: Output of :func:`_compute_filter_edges`.
        gd_mean: Mean group delay.
        gd_var: Variance of the group delay.

    Returns:
        Dict matching :func:`analyze_filter`'s return contract.
    """
    out: dict[str, typing.Any] = {
        "order": filter_repr["order"],
        "type": filter_repr["ftype"],
        "bw_3dB": edges["bw_3dB"],
        "f_stop": edges["f_stop"],
        "tw": edges["tw"],
        "atten_stop": edges["atten_stop"],
        "ripple_pass": edges["ripple_pass"],
        "group_delay_mean": gd_mean,
        "group_delay_var": gd_var,
        "stability": response["stability"],
        "f": response["freqs"],
        "H": response["freq_response"],
    }
    if filter_repr["sos_arr"] is not None:
        out["n_sections"] = filter_repr["n_sections"]
    return out


def analyze_filter(  # pylint: disable=too-many-arguments
    b: _Array1D | float | None = None,
    a: _Array1D | float = 1,
    sos: _Array2D | None = None,
    fs: float | None = None,
    num_freq_points: int = 2**17,
    *,
    passband_tol: float = 3.0,
    stopband_tol: float = 60.0,
    eps: float = 1e-20,
    plot: bool = False,
    xscale: str = "log",
) -> dict[str, typing.Any]:
    """Analyze key characteristics of a digital filter.

    Args:
        b: 1D numerator coefficients (or a scalar). Mutually exclusive with ``sos``.
        a: 1D denominator coefficients (or a scalar). Set ``a == 1`` for FIR filters.
        sos: 2D second-order sections ``(n_sections, 6)`` where each row is
            ``[b0, b1, b2, a0, a1, a2]``. Mutually exclusive with ``b``.
        fs: Sampling rate in Hz. If ``None``, frequencies are normalized to Nyquist (0..1).
        num_freq_points: Number of frequency samples for freqz.
        passband_tol: Passband edge defined at ``-passband_tol`` dB.
        stopband_tol: Stopband edge defined at ``-stopband_tol`` dB.
        eps: Floor added before ``log10`` to avoid ``log(0)``.
        plot: If True, render impulse, pole-zero, magnitude and phase plots.
        xscale: Matplotlib x-axis scale used by the plot (``"log"`` by default).

    Returns:
        Dict with:
            - ``order``: Filter order.
            - ``type``: FIR/IIR guess.
            - ``bw_3dB``: -3 dB bandwidth.
            - ``f_stop``: Stopband edge frequency.
            - ``tw``: Transition width (``f_stop - f_pass``).
            - ``atten_stop``: Minimum stopband attenuation in dB.
            - ``ripple_pass``: Passband ripple in dB.
            - ``group_delay_mean``, ``group_delay_var``: Group-delay statistics in samples.
            - ``stability``: True iff all poles are inside the unit circle.
            - ``f``: Frequency axis.
            - ``H``: Complex frequency response.
            - ``n_sections``: Number of sections (only for SOS input).
    """
    filter_repr = _parse_filter_representation(b, a, sos)
    response = _compute_filter_response(filter_repr, num_freq_points, fs, eps)
    edges = _compute_filter_edges(response["freqs"], response["magnitude_db"], passband_tol, stopband_tol)
    response["stability"], response["zeros"], response["poles"] = _filter_stability_and_roots(
        filter_repr["b_arr"],
        filter_repr["a_arr"],
        filter_repr["sos_arr"],
    )
    out = _build_filter_analysis_result(
        filter_repr,
        response,
        edges,
        *_filter_group_delay(filter_repr, num_freq_points, fs),
    )

    if plot:
        _plot_filter_analysis(filter_repr, response, edges, fs, xscale)

    return out
