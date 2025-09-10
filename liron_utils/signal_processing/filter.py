import numpy as np
import scipy.fft
import scipy.signal
import scipy.ndimage.filters
import scipy.optimize
import scipy.linalg

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
    a :     array_like

    Returns
    -------
    p :     array_like

    """

    if np.isscalar(a):
        if a == 0:
            p = 0
        else:
            p = int(np.ceil(np.log2(a)))
    else:
        a = np.asarray(a)
        p = np.zeros(a.shape, dtype=int)
        idx = (a != 0)
        p[idx] = np.ceil(np.log2(a[idx]))

    return p


# Convolution
conv = scipy.signal.convolve
conv2 = scipy.signal.convolve2d
convn = scipy.signal.convolve
deconv = scipy.signal.deconvolve
convolution_matrix = scipy.linalg.convolution_matrix  # in 'scipy.linalg._special_matrices'

# Digital filtering
filter = scipy.signal.lfilter
filtfilt = scipy.signal.filtfilt  # Zero-phase digital filtering
movmedian = scipy.ndimage.filters.median_filter
movmax = scipy.ndimage.filters.maximum_filter
movmin = scipy.ndimage.filters.minimum_filter


def filter2(h, x, shape="full"):
    """
    2-D digital FIR filter

    Parameters
    ----------
    h :     array_like
        The filter, given as a 2D matrix
    x :     array_like
        The data, given as a 2D matrix
    shape : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:

        ``full``
           Return the full 2-D filtered data.
        ``valid``
           Return only parts of the filtered data that are computed without zero-padded edges.
        ``same``
           Return the central part of the filtered data, which is the same size as x.

    Returns
    -------
    out :   array_like
        The filtered data
    """

    out = scipy.signal.convolve2d(x, np.rot90(h, 2), mode=shape)
    return out


def movsum(x: np.ndarray, N: int, mode: str = 'same', axis: int = -1) -> np.ndarray:
    """
    Moving sum filter

    Args:
        x:          Input array
        N:          Filter size
        mode:       Mode
        axis:       Axis to filter along

    Returns:
        Filtered array
    """
    kernel = np.ones(N)

    return np.apply_along_axis(lambda m: np.convolve(x, kernel, mode=mode), axis=axis, arr=x)


def movmean(x: np.ndarray, N: int, mode: str = 'same', axis: int = -1) -> np.ndarray:
    """Moving average filter"""

    return movsum(x, N, mode=mode, axis=axis) / N


def movrms(x: np.ndarray, N: int, mode: str = 'same', axis: int = -1) -> np.ndarray:
    """Moving RMS filter"""

    return np.sqrt(movmean(x ** 2, N, mode=mode, axis=axis))


def movvar(x: np.ndarray, N: int, ddof: int = 1, mode: str = 'same', axis: int = -1) -> np.ndarray:
    """Moving variance filter"""

    out = movmean(x ** 2, N, mode=mode, axis=axis) - movmean(x, N, mode=mode, axis=axis) ** 2
    out *= N / (N - ddof)
    return out


def movstd(x: np.ndarray, N: int, mode: str = 'same', axis: int = -1) -> np.ndarray:
    """Moving std filter"""
    out = np.sqrt(movvar(x, N, mode=mode, axis=axis))
    return out


def analyze_window(window, N=64, fs=None, worN=2 ** 17, fftbins=True, plot=False):
    """
    Computes key metrics and optionally plots time- and frequency-domain characteristics of a window function.

    Parameters
    ----------
    window : str | tuple | array_like
        SciPy window spec for `get_window` (e.g., "hann", ("kaiser", 8.6)) or a 1-D array of samples.
    N : int, optional, default=64
        Window length to synthesize if `window` is a spec. Ignored if `window` is array-like.
    fs : float, optional
        Sampling rate [Hz]. If given, values are provided in Hz where applicable.
    worN : int, optional, default=131072
        FFT length for high-resolution spectral measurements (use large, e.g., 2^17).
    fftbins : bool, optional, default=True
        If True, create a "periodic" window, ready to use with `ifftshift` and be multiplied by the result of an FFT
        (see also :func:`~scipy.fft.fftfreq`). If False, create a "symmetric" window, for use in filter design.
    plot : bool, optional, default=False
        If True, produce a magnitude plot with annotations.

    Returns
    -------
    out : dict
        Keys (short, MATLAB-ish naming):
            - 'CG'        : Coherent gain (= mean window amplitude)
            - 'ENBW_bins' : Equivalent Noise Bandwidth in DFT bins (dimensionless)
            - 'ENBW_Hz'   : ENBW in Hz (only if fs is provided)
            - 'MLW'       : Main-lobe width between first nulls (normalized, Nyquist=1)
            - 'MLW_3dB'   : -3 dB bandwidth (normalized, Nyquist=1)
            - 'PSL_dB'    : Peak sidelobe level [dB] (relative to main-lobe peak)
            - 'FSL_dB'    : First sidelobe level [dB] (relative to main-lobe peak)
            - 'ISL_dB'    : Integrated sidelobe level [dB] (power outside main-lobe / inside)
            - 'RO_dBdec'  : Sidelobe roll-off slope [dB/decade] (fit to sidelobe peaks)
            - 'RO_dBoct'  : Same, in dB/octave
            - 'f_axes'    : Dict with normalized frequencies of key points
                           {'f_null': ..., 'f_3dB': ...}  (useful for plotting/verification)
    Notes
    -----
    - ENBW (bins) = mean(w^2) / (mean(w))^2
    - ISL is computed as 10*log10( sum |H|^2 over sidelobes / sum |H|^2 over main lobe ),
      using the single-sided (0..π) spectrum; the ratio is invariant to the bin width.
    - Roll-off is estimated by linear regression of sidelobe *peak* levels vs log10(f).
    """

    # --- get window samples
    if isinstance(window, (str, tuple)):
        w = scipy.signal.get_window(window, N, fftbins=fftbins)
    else:
        w = np.asarray(window, dtype=float)
        w /= w.max()  # normalize to unity gain
        N = w.size

    # --- Coherent Gain (DC gain / mean amplitude)
    CG = w.mean()

    # --- ENBW
    ENBW_bins = np.mean(w ** 2) / (np.mean(w) ** 2)  # In DFT bins
    ENBW_Hz = None
    if fs is not None:
        ENBW_Hz = ENBW_bins * fs / N  # In Hz (if fs is given)

    # --- high-res frequency response (0..pi)
    w_freq, H = scipy.signal.freqz(w, worN=worN, whole=False)  # rad/sample, 0..pi
    freqs = w_freq / np.pi  # normalized (Nyquist = 1)
    if fs is not None:
        freqs *= fs / 2  # in Hz if fs is given
    H_mag = np.abs(H)
    H_mag /= H_mag.max()  # normalize to 1 at peak
    H_dB = 20 * np.log10(np.maximum(H_mag, 1e-300))  # dB-safe

    # --- find first *minimum* after DC (first null)
    # use minima of magnitude: peaks of -H_mag
    minima_idx, _ = scipy.signal.find_peaks(-H_mag)
    if minima_idx.size == 0:
        # fallback: first index where it stops decreasing
        deriv = np.diff(H_mag)
        k_null = np.argmax(deriv > 0) or 1
    else:
        k_null = minima_idx[0]
    f_null = freqs[k_null]
    MLW = 2 * f_null  # width between ±f_null in normalized frequency

    # --- -3 dB bandwidth (interpolated)
    # find first crossing to -3 dB before first null
    mask_main = np.arange(k_null + 1)
    k_hp = np.where(H_dB[mask_main] <= -3.0)[0]
    if k_hp.size:
        k2 = k_hp[0]
        k1 = max(k2 - 1, 0)
        # linear interpolation on dB scale
        y1, y2 = H_dB[k1], H_dB[k2]
        x1, x2 = freqs[k1], freqs[k2]
        if y2 == y1:
            f_3dB = x2
        else:
            f_3dB = x1 + (x2 - x1) * ((-3.0 - y1) / (y2 - y1))
        MLW_3dB = 2 * f_3dB
    else:
        f_3dB, MLW_3dB = np.nan, np.nan

    # --- sidelobe peaks and levels
    peaks_idx, _ = scipy.signal.find_peaks(H_mag)
    side_peaks = peaks_idx[freqs[peaks_idx] > f_null]
    if side_peaks.size:
        PSL_dB = H_dB[side_peaks].max()  # max sidelobe
        FSL_dB = H_dB[side_peaks[0]]  # first sidelobe
    else:
        PSL_dB, FSL_dB = np.nan, np.nan

    # --- ISL (power ratio sidelobes/main-lobe)
    P = H_mag ** 2
    P_main = P[:k_null + 1].sum()
    P_side = P[k_null + 1:].sum()
    ISL_dB = 10 * np.log10(P_side / P_main) if P_main > 0 else np.nan

    # --- roll-off: slope of sidelobe *peaks* in dB vs log10(freq)
    if side_peaks.size >= 2:
        fpk = freqs[side_peaks]
        ypk = H_dB[side_peaks]
        # ignore any numerically tiny/late peaks
        good = (fpk > f_null) & np.isfinite(ypk) & (ypk < -1)  # below -1 dB to avoid main-lobe skirt
        fpk, ypk = fpk[good], ypk[good]
        if fpk.size >= 2:
            x = np.log10(fpk)
            a, b = np.polyfit(x, ypk, 1)  # y ≈ a*log10(f) + b
            RO_dBdec = float(a)
            RO_dBoct = float(a * np.log10(2.0))  # per octave
        else:
            RO_dBdec = RO_dBoct = np.nan
    else:
        RO_dBdec = RO_dBoct = np.nan

    out = dict(
            CG=CG,
            ENBW_bins=ENBW_bins,
            ENBW_Hz=ENBW_Hz,
            MLW=MLW,
            MLW_3dB=MLW_3dB,
            PSL_dB=PSL_dB,
            FSL_dB=FSL_dB,
            ISL_dB=ISL_dB,
            RO_dBdec=RO_dBdec,
            RO_dBoct=RO_dBoct,
            f_null=f_null,
            f_3dB=f_3dB,
    )

    if plot:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs = iter(axs)

        # ---- Time-domain
        ax = next(axs)
        ax.plot(np.arange(N), w)

        ax.set_xlabel("Samples")
        ax.set_ylabel("Amplitude")

        # Build window label for title (if name/spec provided)
        win_label = f"{N=}"
        if isinstance(window, str):
            win_label = f"{window}({N=})"
        elif isinstance(window, tuple) and len(window) >= 1:
            name = str(window[0])
            params = ", ".join(f"{p:g}" if isinstance(p, (int, float)) else str(p) for p in window[1:])
            win_label = f"{name}({N=}, {params})" if params else name
        ax.set_title(f"Time Domain — {win_label if win_label else ''}")

        ax.set_ylim(0, w.max() * 1.05)

        # ---- Frequency domain
        ax = next(axs)
        ax.plot(freqs, H_dB)

        xlabel = "Normalized frequency (× Nyquist)"
        if fs is not None:
            xlabel = "Frequency (Hz)"
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Magnitude (dB)")

        ax.set_title(f"Frequency Domain")
        ax.grid(True)

        # annotate main-lobe width
        ax.axvline(+f_null, linestyle="--", linewidth=0.5)
        ax.axvline(-f_null, linestyle="--", linewidth=0.5)
        ax.text(f_null, -6, rf"$\text{{MLW}}={MLW:.3f}${'' if fs is None else 'Hz'}", ha="left", va="top")

        # annotate 3dB bandwidth
        if np.isfinite(f_3dB):
            ax.axvline(f_3dB, linestyle=":")
            ax.text(f_3dB, -3, rf"$\text{{MLW}}_{{-3\text{{dB}}}}={MLW_3dB:.3f}${'' if fs is None else 'Hz'}",
                    ha="left", va="bottom")

        # mark PSL peak
        if side_peaks.size:
            k_psl = side_peaks[np.argmax(H_dB[side_peaks])]
            ax.plot(freqs[k_psl], H_dB[k_psl], marker="o")
            ax.text(freqs[k_psl], H_dB[k_psl], rf"PSL$={H_dB[k_psl]:.1f}$dB", ha="left", va="bottom")

        # shade sidelobe region for ISL visualization
        ax.fill_between(freqs[k_null + 1:], H_dB[k_null + 1:], -300, alpha=0.08)

        # Metrics textbox
        unit_freq = "" if fs is None else " Hz"
        metrics_text = "\n".join([f"{k}={v:.3f}" for k, v in out.items() if v is not None])
        ax.text(0.98, 0.98,
                metrics_text,
                fontsize="x-small",
                ha="right", va="top",
                transform=ax.transAxes,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="0.8"),
        )

        ax.set_xlim(0, 1 if fs is None else fs / 2)
        ax.set_ylim(-150, 5)

        plt.tight_layout()
        plt.show(block=False)

    return out


def analyze_filter(b=None, a=1, sos=None, fs=None, worN=2 ** 14,
        passband_tol=3.0, stopband_tol=60.0, eps=1e-20, plot=False):
    """
    Analyze key characteristics of a digital filter.

    Parameters
    ----------
    b, a : array_like, optional
        Numerator (b) and denominator (a) coefficients of the filter.
        Ignored if sos is provided.
    sos : array_like, optional
        Second-order sections representation of the IIR filter.
        Should be an array of shape (n_sections, 6) where each row
        represents one second-order section in the format [b0, b1, b2, a0, a1, a2].
        If provided, b and a parameters are ignored.
    fs : float, optional
        Sampling rate [Hz]. If None, frequencies are normalized to Nyquist (0..1).
    worN : int, optional
        Number of frequency samples for freqz. Default 2^14.
    passband_tol : float
        Passband edge defined at -passband_tol [dB] (default 3 dB).
    stopband_tol : float
        Stopband defined at -stopband_tol [dB] (default 60 dB).
    plot : bool, optional
        If True, plot responses.

    Returns
    -------
    out : dict
        Keys:
            - 'order'        : Filter order
            - 'type'         : FIR/IIR guess
            - 'bw_3dB'       : -3 dB bandwidth
            - 'f_stop'       : Stopband edge frequency
            - 'tw'           : Transition width = f_stop - f_pass
            - 'atten_stop'   : Min stopband attenuation [dB]
            - 'ripple_pass'  : Passband ripple [dB]
            - 'group_delay_mean': Mean group delay [samples]
            - 'group_delay_var' : Variance of group delay
            - 'stability'    : Stable (all poles inside unit circle)
            - 'f'            : Frequency axis
            - 'H'            : Frequency response (complex)
            - 'n_sections'   : Number of sections (only for SOS)
    """

    # Determine filter representation and convert if necessary
    if not ((b is None) ^ (sos is None)):
        raise ValueError("Either `b` or `sos` must be provided.")

    is_sos = b is None

    if is_sos:  # sos
        sos = np.asarray(sos)
        if sos.ndim != 2 or sos.shape[1] != 6:
            raise ValueError("`sos` must have shape (n_sections, 6)")

        n_sections = sos.shape[0]

        # Convert SOS to transfer function for some calculations
        b, a = scipy.signal.sos2tf(sos)
        b, a = np.atleast_1d(b), np.atleast_1d(a)

        # Order calculation for SOS
        order = n_sections * 2  # Each section is 2nd order
        ftype = "IIR"  # SOS is always IIR

    else:  # b, a
        b, a = np.atleast_1d(b), np.atleast_1d(a)
        n_sections = None

        # Order and type
        order = max(len(b), len(a)) - 1
        ftype = "FIR" if np.allclose(a, [1]) else "IIR"

    # Frequency response - use appropriate function for SOS vs b,a
    if fs is None:  # Normalized frequency [0..1], with Nyquist=1
        if is_sos:
            w, H = scipy.signal.sosfreqz(sos, worN=worN, whole=False)
        else:
            w, H = scipy.signal.freqz(b, a, worN=worN, whole=False)
        f = w / np.pi
    else:
        if is_sos:
            f, H = scipy.signal.sosfreqz(sos, worN=worN, fs=fs)
        else:
            f, H = scipy.signal.freqz(b, a, worN=worN, fs=fs)

    H_mag = np.abs(H)
    H_dB = 20 * np.log10(H_mag + eps)

    # --- Passband edge (first -3 dB crossing)
    idx_pass = np.argmax(H_dB <= -passband_tol)
    f_pass = f[idx_pass] if idx_pass > 0 else np.nan
    bw_3dB = f_pass

    # Passband ripple (up to f_pass)
    ripple_pass = (H_dB[:idx_pass].max() - H_dB[:idx_pass].min()) if idx_pass > 0 else np.nan

    # --- Stopband edge (first frequency below stopband_tol)
    idx_stop = np.argmax(H_dB <= -stopband_tol)
    f_stop = f[idx_stop] if idx_stop > 0 else np.nan

    # Stopband attenuation (after f_stop)
    atten_stop = -H_dB[idx_stop:].max() if idx_stop > 0 else np.nan

    # Transition width
    tw = (f_stop - f_pass) if np.isfinite(f_pass) and np.isfinite(f_stop) else np.nan

    # --- Group delay
    try:
        system = sos if is_sos else (b, a)
        w_gd, gd = scipy.signal.group_delay(system, worN=worN, fs=(fs if fs else 2 * np.pi))
        group_delay_mean = np.mean(gd[np.isfinite(gd)])
        group_delay_var = np.var(gd[np.isfinite(gd)])
    except Exception:
        group_delay_mean = group_delay_var = np.nan

    # --- Stability
    if is_sos:
        # Check stability for each section in SOS
        stability = True
        all_zeros, all_poles = [], []
        for section in sos:
            b_sec = section[:3]  # [b0, b1, b2]
            a_sec = section[3:]  # [a0, a1, a2]
            # Normalize by a0
            if a_sec[0] != 0:
                a_sec = a_sec / a_sec[0]
                b_sec = b_sec / section[3]

            zeros_sec = np.roots(b_sec)
            poles_sec = np.roots(a_sec)

            all_zeros.extend(zeros_sec)
            all_poles.extend(poles_sec)

            # Check if any pole is outside unit circle
            if np.any(np.abs(poles_sec) >= 1):
                stability = False

        zeros = np.array(all_zeros)
        poles = np.array(all_poles)
    else:
        zeros, poles = np.roots(b), np.roots(a)
        stability = np.all(np.abs(poles) < 1)

    out = dict(
            order=order,
            type=ftype,
            bw_3dB=bw_3dB,
            f_stop=f_stop,
            tw=tw,
            atten_stop=atten_stop,
            ripple_pass=ripple_pass,
            group_delay_mean=group_delay_mean,
            group_delay_var=group_delay_var,
            stability=stability,
            f=f,
            H=H
    )
    if is_sos:
        out['n_sections'] = n_sections

    if plot:
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        axs = iter(axs.flatten())

        # Impulse response
        if ftype == "FIR":
            h = b
        else:  # IIR
            n_imp = max(200, order * 4)
            x = np.zeros(n_imp)
            x[0] = 1
            if is_sos:
                h = scipy.signal.sosfilt(sos, x)
            else:
                h = scipy.signal.lfilter(b, a, x)
        ax = next(axs)
        ax.stem(np.arange(len(h)), h)
        title = f"Impulse Response"
        if is_sos:
            title += f" (SOS: {n_sections} sections)"
        ax.set_title(title)
        ax.set_xlabel("Samples")
        ax.set_ylabel("Amplitude")
        ax.grid(True)

        # Pole-zero plot
        ax = next(axs)
        ax.scatter(np.real(zeros), np.imag(zeros), marker='o', label='Zeros')
        ax.scatter(np.real(poles), np.imag(poles), marker='x', label='Poles')
        circ = plt.Circle((0, 0), 1, color='black', fill=False, ls='--')
        ax.add_patch(circ)
        ax.set_title("Pole-Zero Plot")
        ax.set_xlabel(r"$\Re{\{z\}}$")
        ax.set_ylabel(r"$\Im{\{z\}}$")
        ax.set_aspect('equal')
        ax.legend()
        ax.grid(True)

        # Magnitude response
        ax = next(axs)
        ax.plot(f, H_dB)
        ax.set_title("Magnitude Response")
        if fs is None:
            ax.set_xlabel("Normalized Frequency (× Nyquist)")
            ax.set_xlim(0, 1)
        else:
            ax.set_xlabel("Frequency [Hz]")
            ax.set_xlim(0, fs / 2)
        ax.set_ylabel("Magnitude [dB]")
        ax.grid(True)
        if np.isfinite(f_pass):
            ax.axvline(f_pass, ls="--", c="r", label=f"Passband edge {f_pass:.3g}")
        if np.isfinite(f_stop):
            ax.axvline(f_stop, ls="--", c="g", label=f"Stopband edge {f_stop:.3g}")
        ax.legend()

        # Phase response
        ax = next(axs)
        ax.plot(f, np.rad2deg(np.unwrap(np.angle(H))))
        ax.set_title("Phase Response")
        ax.set_ylabel("Phase [deg]")
        if fs is None:
            ax.set_xlabel("Normalized Frequency (× Nyquist)")
            ax.set_xlim(0, 1)
        else:
            ax.set_xlabel("Frequency [Hz]")
            ax.set_xlim(0, fs / 2)
        ax.grid(True)

        plt.tight_layout()
        plt.show(block=False)

    return out
