import typing

import numpy as np
import scipy.signal

_Array1D = np.ndarray[tuple[int], np.dtype[typing.Any]]
_Array2D = np.ndarray[tuple[int, int], np.dtype[typing.Any]]

_SI_PREFIXES = {0: "", 3: "k", 6: "M", 9: "G", 12: "T"}


def spectrum_display_data(
    spectrum: _Array1D,
    *,
    which: str = "power",
    db: bool = False,
    eps: float = 1e-20,
) -> tuple[_Array1D, str]:
    """Convert a complex spectrum to displayable amplitude/power/phase data.

    Args:
        spectrum: 1D complex spectrum.
        which: One of ``"amp"``, ``"power"``, ``"phase"``.
        db: If True (and ``which`` is amp/power), convert to dB.
        eps: Floor added before ``log10`` to avoid log(0).

    Returns:
        ``(ydata, ylabel)`` — the display values and the matching y-axis label.

    Raises:
        ValueError: If ``which`` is not one of ``"amp"``, ``"power"``, ``"phase"``.
    """
    which = which.lower()
    ydata: _Array1D
    if which == "amp":
        ydata = typing.cast(_Array1D, np.abs(spectrum))
        ylabel = "Amplitude"
    elif which == "power":
        ydata = typing.cast(_Array1D, np.abs(spectrum) ** 2)
        ylabel = "Power"
    elif which == "phase":
        ydata = typing.cast(_Array1D, np.degrees(np.unwrap(np.angle(spectrum))))
        ylabel = "Phase [deg]"
    else:
        raise ValueError(f"which must be one of 'amp', 'power', or 'phase'. Got: {which}")

    if db and which in ("amp", "power"):
        ydata = typing.cast(_Array1D, 10 * np.log10(ydata + eps))
        ylabel += " [dB]"

    return ydata, ylabel


def fft_data(
    x: _Array1D,
    fs: float = 1.0,
    n: int | None = None,
    *,
    one_sided: bool = True,
    normalize: bool = False,
    input_time: bool = True,
) -> tuple[_Array1D, _Array1D]:
    """Compute an FFT spectrum and its frequency axis, one- or two-sided.

    Args:
        x: 1D input signal. Complex inputs always yield two-sided output.
        fs: Sampling frequency in Hz.
        n: FFT length. If None, uses ``len(x)``.
        one_sided: If True, return only the positive frequencies (auto-disabled for complex x).
        normalize: If True, scale the spectrum so its peak is 1.
        input_time: If True, treat ``x`` as time-domain and apply FFT.
            If False, treat ``x`` as already in the frequency domain.

    Returns:
        ``(spectrum, freqs)`` as 1D arrays (two-sided output is fft-shifted).
    """
    x_arr = typing.cast(_Array1D, np.asarray(x))
    if n is None:
        n = x_arr.shape[0]

    spectrum: _Array1D
    if input_time:
        if np.iscomplexobj(x_arr):
            one_sided = False
        spectrum = typing.cast(_Array1D, np.fft.fft(x_arr, n=n, axis=0))
    else:
        spectrum = x_arr.copy()

    if normalize:
        spectrum /= spectrum.max(axis=0)

    freqs = typing.cast(_Array1D, np.fft.fftfreq(n=n, d=1 / fs))

    if one_sided:
        spectrum = typing.cast(_Array1D, spectrum[: n // 2])
        freqs = typing.cast(_Array1D, freqs[: n // 2])
    else:
        spectrum = typing.cast(_Array1D, np.fft.fftshift(spectrum, axes=0))
        freqs = typing.cast(_Array1D, np.fft.fftshift(freqs))

    return spectrum, freqs


def periodogram_data(
    x: _Array1D,
    fs: float = 1.0,
    n: int | None = None,
    *,
    window: str = "boxcar",
    one_sided: bool = True,
    normalize: bool = False,
) -> tuple[_Array1D, _Array1D]:
    """Compute a PSD estimate via ``scipy.signal.periodogram``.

    Args:
        x: 1D input signal. Complex inputs always yield two-sided output.
        fs: Sampling frequency in Hz.
        n: FFT length; ``None`` uses ``len(x)``.
        window: Window function name accepted by ``scipy.signal.periodogram``.
        one_sided: If True, return only the positive frequencies (auto-disabled for complex x).
        normalize: If True, scale the PSD so its peak is 1.

    Returns:
        ``(psd, freqs)`` as 1D arrays (two-sided output is fft-shifted).
    """
    x_arr = typing.cast(_Array1D, np.asarray(x))
    if n is None:
        n = x_arr.shape[0]
    if np.iscomplexobj(x_arr):
        one_sided = False

    freqs, psd = scipy.signal.periodogram(
        x_arr,
        fs=fs,
        window=typing.cast(typing.Any, window),
        nfft=n,
        detrend=False,
        return_onesided=one_sided,
        scaling="density",
        axis=0,
    )

    if normalize:
        psd = psd / psd.max(axis=0)

    if not one_sided:
        psd = typing.cast(_Array1D, np.fft.fftshift(psd, axes=0))
        freqs = typing.cast(_Array1D, np.fft.fftshift(freqs))

    return typing.cast(_Array1D, psd), typing.cast(_Array1D, freqs)


def frequency_response_data(
    b: _Array1D | float,
    a: _Array1D | float = 1,
    fs: float | None = 1.0,
    num_freq_points: int = 512,
    *,
    one_sided: bool = True,
) -> tuple[_Array1D, _Array1D]:
    """Compute the frequency response of an LTI system given its ``(b, a)`` coefficients.

    For ``fs is None`` the system is treated as continuous-time (``scipy.signal.freqs``);
    otherwise ``scipy.signal.freqz`` is used.

    Args:
        b: 1D numerator coefficients (or a scalar).
        a: 1D denominator coefficients (set ``a == 1`` for FIR systems).
        fs: Sampling frequency in Hz. ``None`` selects continuous-time.
        num_freq_points: Number of frequency points to evaluate.
        one_sided: If True, evaluate ``[0, fs/2)``; otherwise ``[-fs/2, fs/2)``, centered.

    Returns:
        ``(h, freqs)`` — complex response and matching frequency axis (Hz, or rad/s/2π
        for continuous-time).
    """
    b_arr = typing.cast(_Array1D, np.atleast_1d(b))
    a_arr = typing.cast(_Array1D, np.atleast_1d(a))

    if fs is None:  # Continuous-time system
        w, h = scipy.signal.freqs(
            typing.cast(typing.Any, b_arr),
            typing.cast(typing.Any, a_arr),
            worN=num_freq_points,
        )
        return typing.cast(_Array1D, h), typing.cast(_Array1D, np.asarray(w) / (2 * np.pi))

    freqs, h = scipy.signal.freqz(b=b_arr, a=a_arr, fs=fs, worN=num_freq_points, whole=not one_sided)
    freqs = typing.cast(_Array1D, np.asarray(freqs))
    h = typing.cast(_Array1D, np.asarray(h))
    if not one_sided:
        # freqz(whole=True) evaluates [0, fs); recenter to [-fs/2, fs/2).
        upper = freqs >= fs / 2
        freqs = typing.cast(_Array1D, np.concatenate([freqs[upper] - fs, freqs[~upper]]))
        h = typing.cast(_Array1D, np.concatenate([h[upper], h[~upper]]))
    return h, freqs


def impulse_response_data(
    b: _Array1D | float,
    a: _Array1D | float = 1,
    dt: float | None = 1,
    t: _Array1D | None = None,
    *,
    n: int | None = None,
) -> tuple[_Array1D, _Array1D, bool]:
    """Compute the impulse response of an LTI system given its ``(b, a)`` coefficients.

    Args:
        b: 1D numerator coefficients (or a scalar).
        a: 1D denominator coefficients (set ``a == 1`` for FIR systems).
        dt: Sample period. If None, treats the system as continuous-time and requires ``t``.
        t: 1D time grid for the response. If None for discrete-time, ``n`` is used to build one.
        n: Sample count for the discrete-time response; required when ``t`` is None.

    Returns:
        ``(h, t_out, is_discrete)``.

    Raises:
        ValueError: For continuous-time when ``t`` is not given; for discrete-time when
            neither ``t`` nor ``n`` is given.
    """
    b_arr = typing.cast(_Array1D, np.atleast_1d(b))
    a_arr = typing.cast(_Array1D, np.atleast_1d(a))
    if len(b_arr) > len(a_arr):
        a_arr = typing.cast(_Array1D, np.pad(a_arr, (0, len(b_arr) - len(a_arr)), "constant", constant_values=0))
    elif len(a_arr) > len(b_arr):
        b_arr = typing.cast(_Array1D, np.pad(b_arr, (0, len(a_arr) - len(b_arr)), "constant", constant_values=0))

    if dt is None:  # Continuous-time system
        system: typing.Any = scipy.signal.lti(typing.cast(typing.Any, b_arr), typing.cast(typing.Any, a_arr))
        if t is None:
            raise ValueError("t should be given for continuous-time system.")
        t_out, h = scipy.signal.impulse(system, T=t)
        return typing.cast(_Array1D, h), typing.cast(_Array1D, t_out), False

    system = scipy.signal.dlti(typing.cast(typing.Any, b_arr), typing.cast(typing.Any, a_arr), dt=dt)
    if t is None:
        if n is None:
            raise ValueError("Either t or n should be given.")
        t = typing.cast(_Array1D, np.arange(0, n * dt, dt))
    t_out, h_seq = scipy.signal.dimpulse(system, n=len(t))
    h = typing.cast(_Array1D, np.squeeze(h_seq))
    return h, typing.cast(_Array1D, np.asarray(t_out)), True


def spectrogram_data(
    y: _Array1D,
    fs: float = 1.0,
    *,
    nfft: int = 4096,
    window: str = "blackmanharris",
    overlap_fraction: float = 0.85,
    db: bool = True,
    eps: float = 1e-20,
) -> tuple[_Array2D, _Array1D, _Array1D]:
    """Compute a spectrogram via ``scipy.signal.spectrogram``.

    Args:
        y: 1D time-domain signal.
        fs: Sample rate in Hz.
        nfft: Segment/FFT length.
        window: Window function name accepted by ``scipy.signal.get_window``.
        overlap_fraction: Segment overlap as a fraction of ``nfft``.
        db: If True, convert power to dB.
        eps: Floor added before ``log10`` to avoid log(0).

    Returns:
        ``(spec, freqs, times)`` — spectrogram matrix of shape (freqs, times) and its axes.
    """
    y_arr = typing.cast(_Array1D, np.asarray(y))
    freqs, times, spec = scipy.signal.spectrogram(
        y_arr,
        fs=fs,
        window=typing.cast(typing.Any, scipy.signal.get_window(window, nfft)),
        nperseg=nfft,
        noverlap=int(overlap_fraction * nfft),
        detrend=False,
        scaling="density",
        mode="psd",
    )
    spec = typing.cast(_Array2D, spec)
    if db:
        spec = typing.cast(_Array2D, 10 * np.log10(spec + eps))
    return spec, typing.cast(_Array1D, freqs), typing.cast(_Array1D, times)


def si_prefix_scale(max_value: float) -> tuple[int, str]:
    """Pick an SI prefix (engineering scale) for a positive axis maximum.

    Args:
        max_value: Largest value on the axis.

    Returns:
        ``(scale_exponent, prefix_letter)`` — e.g. ``(3, "k")``; divide values by
        ``10**scale_exponent`` and prepend the letter to the unit.
    """
    if max_value <= 1:
        return 0, ""
    scale = 3 * (int(np.log10(max_value)) // 3)
    scale = min(max(scale, 0), 12)
    return scale, _SI_PREFIXES[scale]
