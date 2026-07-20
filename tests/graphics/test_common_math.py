import typing

import numpy as np
import pytest

from liron_utils.graphics.common.fitting import (
    curve_fit_confidence_band,
    curve_fit_prep_data,
)
from liron_utils.graphics.common.spectra import (
    fft_data,
    frequency_response_data,
    impulse_response_data,
    periodogram_data,
    si_prefix_scale,
    spectrogram_data,
    spectrum_display_data,
)

_Array1D = np.ndarray[tuple[int], np.dtype[typing.Any]]


def test_spectrum_display_data_power_db() -> None:
    spectrum = typing.cast(_Array1D, np.array([1.0 + 0j, 10.0 + 0j]))
    ydata, ylabel = spectrum_display_data(spectrum, which="power", db=True)
    assert ylabel == "Power [dB]"
    np.testing.assert_allclose(ydata, [0.0, 20.0], atol=1e-6)


def test_spectrum_display_data_amp_phase() -> None:
    spectrum = typing.cast(_Array1D, np.array([3.0 + 4.0j]))
    ydata, ylabel = spectrum_display_data(spectrum, which="amp")
    assert ylabel == "Amplitude"
    np.testing.assert_allclose(ydata, [5.0])
    _, ylabel = spectrum_display_data(spectrum, which="phase")
    assert ylabel == "Phase [deg]"


def test_spectrum_display_data_invalid_which() -> None:
    with pytest.raises(ValueError):
        spectrum_display_data(typing.cast(_Array1D, np.ones(4)), which="bogus")


def test_fft_data_real_sine_peak() -> None:
    fs = 100.0
    t = np.arange(0, 1, 1 / fs)
    x = typing.cast(_Array1D, np.sin(2 * np.pi * 10 * t))
    spectrum, freqs = fft_data(x, fs=fs)
    assert spectrum.shape == freqs.shape == (len(t) // 2,)
    assert freqs[np.argmax(np.abs(spectrum))] == pytest.approx(10.0)


def test_fft_data_complex_is_two_sided() -> None:
    x = typing.cast(_Array1D, np.exp(2j * np.pi * 0.1 * np.arange(64)))
    spectrum, freqs = fft_data(x, fs=1.0, one_sided=True)  # auto-disabled for complex input
    assert spectrum.shape == freqs.shape == (64,)
    assert freqs[0] == pytest.approx(-0.5)


def test_periodogram_data_peak() -> None:
    fs = 100.0
    t = np.arange(0, 2, 1 / fs)
    x = typing.cast(_Array1D, np.sin(2 * np.pi * 10 * t))
    psd, freqs = periodogram_data(x, fs=fs)
    assert freqs[np.argmax(psd)] == pytest.approx(10.0)


def test_frequency_response_data_fir_dc_gain() -> None:
    b = typing.cast(_Array1D, np.ones(4) / 4)
    h, freqs = frequency_response_data(b, fs=1.0, num_freq_points=128)
    assert h.shape == freqs.shape == (128,)
    assert np.abs(h[0]) == pytest.approx(1.0)
    assert freqs[0] == pytest.approx(0.0)


def test_frequency_response_data_two_sided_centered() -> None:
    b = typing.cast(_Array1D, np.ones(4) / 4)
    h, freqs = frequency_response_data(b, fs=1.0, num_freq_points=128, one_sided=False)
    assert h.shape == freqs.shape == (128,)
    assert freqs[0] == pytest.approx(-0.5)
    assert np.all(np.diff(freqs) > 0)


def test_impulse_response_data_fir_discrete() -> None:
    b = typing.cast(_Array1D, np.array([1.0, 2.0, 3.0]))
    h, t_out, is_discrete = impulse_response_data(b, n=3)
    assert is_discrete
    np.testing.assert_allclose(h, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(t_out, [0.0, 1.0, 2.0])


def test_impulse_response_data_requires_t_or_n() -> None:
    with pytest.raises(ValueError):
        impulse_response_data(typing.cast(_Array1D, np.array([1.0])))


def test_spectrogram_data_shapes() -> None:
    fs = 8000.0
    y = typing.cast(_Array1D, np.random.default_rng(0).standard_normal(2**14))
    spec, freqs, times = spectrogram_data(y, fs=fs, nfft=256, overlap_fraction=0.5)
    assert spec.shape == (freqs.shape[0], times.shape[0])
    assert freqs[-1] == pytest.approx(fs / 2)


def test_si_prefix_scale() -> None:
    assert si_prefix_scale(500.0) == (0, "")
    assert si_prefix_scale(24_000.0) == (3, "k")
    assert si_prefix_scale(2.4e6) == (6, "M")
    assert si_prefix_scale(1.0e10) == (9, "G")


def test_curve_fit_prep_data_sorts_by_x() -> None:
    x = typing.cast(_Array1D, np.array([3.0, 1.0, 2.0]))
    y = typing.cast(_Array1D, np.array([30.0, 10.0, 20.0]))
    yerr = typing.cast(_Array1D, np.array([0.3, 0.1, 0.2]))
    x_out, y_out, xerr_out, yerr_out, p_opt = curve_fit_prep_data(x, y, None, yerr, typing.cast(_Array1D | None, None))
    np.testing.assert_allclose(x_out, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(y_out, [10.0, 20.0, 30.0])
    assert yerr_out is not None
    np.testing.assert_allclose(yerr_out, [0.1, 0.2, 0.3])
    assert xerr_out is None
    assert p_opt is None


def test_curve_fit_confidence_band_linear() -> None:
    def fit_fcn(x: _Array1D, a: float, b: float) -> _Array1D:
        return typing.cast(_Array1D, a * x + b)

    x = typing.cast(_Array1D, np.linspace(0, 1, 11))
    p_opt = typing.cast(_Array1D, np.array([2.0, 1.0]))
    p_cov = np.diag([0.01, 0.04])
    fit_low, fit_high = curve_fit_confidence_band(fit_fcn, x, p_opt, p_cov, n_std=2)
    mid = fit_fcn(x, 2.0, 1.0)
    assert np.all(fit_low <= mid)
    assert np.all(mid <= fit_high)
    assert fit_high[0] - fit_low[0] == pytest.approx(2 * 2 * 0.2)  # at x=0 only the intercept perturbs
