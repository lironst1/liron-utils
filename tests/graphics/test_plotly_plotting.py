import typing

import numpy as np
import pytest
import scipy.stats

from liron_utils.graphics.plotly import Figure

_Array1D = np.ndarray[tuple[int], np.dtype[typing.Any]]


def _arr(values: list[float]) -> _Array1D:
    return typing.cast(_Array1D, np.asarray(values, dtype=np.float64))


def test_plot_line() -> None:
    fig = Figure()
    fig.plot(_arr([0, 1, 2]), _arr([0, 1, 4]), name="curve")
    assert len(fig.data) == 1
    assert fig.data[0].type == "scatter"
    assert fig.data[0].mode == "lines"


def test_plot_y_only_uses_index_x() -> None:
    fig = Figure()
    fig.plot(_arr([5, 6, 7]))
    np.testing.assert_allclose(fig.data[0].x, [0, 1, 2])
    np.testing.assert_allclose(fig.data[0].y, [5, 6, 7])


def test_plot_3d() -> None:
    fig = Figure(subplot_type="scene")
    fig.plot(_arr([0, 1]), _arr([0, 1]), _arr([0, 1]))
    assert fig.data[0].type == "scatter3d"


def test_plot_vlines_label_on_last_only() -> None:
    fig = Figure()
    fig.plot_vlines(_arr([1.0, 2.0, 3.0]), label="cuts")
    shapes = fig.layout.shapes
    assert len(shapes) == 3
    assert [bool(s.showlegend) for s in shapes] == [False, False, True]
    assert shapes[-1].name == "cuts"


def test_plot_hlines_scalar() -> None:
    fig = Figure()
    fig.plot_hlines(2.5)
    assert len(fig.layout.shapes) == 1


def test_draw_xy_lines() -> None:
    fig = Figure()
    fig.draw_xy_lines()
    assert len(fig.layout.shapes) == 2


def test_plot_errorbar() -> None:
    fig = Figure()
    fig.plot_errorbar(_arr([1, 2, 3]), _arr([2, 4, 6]), yerr=_arr([0.1, 0.2, 0.3]))
    trace = fig.data[0]
    assert trace.mode == "markers"
    np.testing.assert_allclose(trace.error_y.array, [0.1, 0.2, 0.3])
    assert trace.error_x.array is None or len(trace.error_x.array) == 0


def test_plot_errorbar_y_only_rejects_errors() -> None:
    fig = Figure()
    with pytest.raises(AssertionError):
        fig.plot_errorbar(_arr([1, 2]), yerr=_arr([0.1, 0.2]))


def test_plot_filled_error_band() -> None:
    fig = Figure()
    fig.plot_filled_error(_arr([0, 1, 2]), _arr([1, 2, 3]), yerr=_arr([0.5, 0.5, 0.5]), n_std=2)
    assert len(fig.data) == 2
    np.testing.assert_allclose(fig.data[0].y, [0, 1, 2])  # low = y - 2*0.5
    np.testing.assert_allclose(fig.data[1].y, [2, 3, 4])  # high = y + 2*0.5
    assert fig.data[1].fill == "tonexty"


def test_plot_filled_error_explicit_bounds() -> None:
    fig = Figure()
    fig.plot_filled_error(_arr([0, 1]), y_low=_arr([0, 0]), y_high=_arr([1, 1]))
    assert len(fig.data) == 2


def test_plot_filled_error_rejects_mixed_args() -> None:
    fig = Figure()
    with pytest.raises(AssertionError):
        fig.plot_filled_error(_arr([0, 1]), _arr([1, 1]), y_low=_arr([0, 0]), y_high=_arr([2, 2]))


def test_plot_data_and_curve_fit() -> None:
    def fit_fcn(x: _Array1D, a: float, b: float) -> _Array1D:
        return typing.cast(_Array1D, a * x + b)

    x = _arr([0, 1, 2, 3])
    y = _arr([1, 3, 5, 7])
    p_opt = _arr([2.0, 1.0])
    p_cov = np.diag([0.01, 0.01])
    fig = Figure()
    fig.plot_data_and_curve_fit(x, y, fit_fcn, yerr=_arr([0.1] * 4), p_opt=p_opt, p_cov=p_cov)
    # errorbar + fit line + 2 band traces
    assert len(fig.data) == 4
    assert fig.data[0].name == "Data"
    assert fig.data[1].name == "Curve fit"


def test_plot_data_and_curve_fit_without_cov() -> None:
    def fit_fcn(x: _Array1D, a: float) -> _Array1D:
        return typing.cast(_Array1D, a * x)

    fig = Figure()
    fig.plot_data_and_curve_fit(_arr([0, 1, 2]), _arr([0, 2, 4]), fit_fcn, p_opt=_arr([2.0]))
    assert len(fig.data) == 2  # no band without p_cov


def test_plot_data_and_lin_reg() -> None:
    rng = np.random.default_rng(0)
    x = np.arange(50, dtype=np.float64)
    y = typing.cast(_Array1D, 2 * x + rng.standard_normal(50))
    reg = scipy.stats.linregress(x, y)
    fig = Figure()
    fig.plot_data_and_lin_reg(x, y, reg)
    assert len(fig.data) == 2
    assert "slope" in fig.data[1].name


def test_plot_fft() -> None:
    fs = 100.0
    t = np.arange(0, 1, 1 / fs)
    x = typing.cast(_Array1D, np.sin(2 * np.pi * 10 * t))
    fig = Figure()
    spectrum, freqs = fig.plot_fft(x, fs=fs, db=True)
    assert len(fig.data) == 1
    assert freqs[np.argmax(np.abs(spectrum))] == pytest.approx(10.0)
    assert fig.layout.xaxis.title.text == "Frequency [Hz]"
    assert fig.layout.yaxis.title.text == "Power [dB]"


def test_plot_fft_normalized_axis_label() -> None:
    fig = Figure()
    fig.plot_fft(_arr([0, 1, 0, -1] * 8), fs=1.0)
    assert fig.layout.xaxis.title.text == "Frequency [normalized]"


def test_plot_periodogram() -> None:
    fs = 100.0
    t = np.arange(0, 2, 1 / fs)
    x = typing.cast(_Array1D, np.sin(2 * np.pi * 10 * t))
    fig = Figure()
    psd, freqs = fig.plot_periodogram(x, fs=fs)
    assert len(fig.data) == 1
    assert freqs[np.argmax(psd)] == pytest.approx(10.0)


def test_plot_frequency_response() -> None:
    b = typing.cast(_Array1D, np.ones(8) / 8)
    fig = Figure()
    h, freqs = fig.plot_frequency_response(b, fs=1.0, db=True)
    assert len(fig.data) == 1
    assert h.shape == freqs.shape
    assert np.abs(h[0]) == pytest.approx(1.0)


def test_plot_impulse_response_discrete_stem() -> None:
    fig = Figure()
    h, t_out = fig.plot_impulse_response(_arr([1, 2, 3]), n=3)
    assert len(fig.data) == 2  # stem segments + markers
    assert fig.data[0].mode == "lines"
    assert fig.data[1].mode == "markers"
    np.testing.assert_allclose(h, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(t_out, [0.0, 1.0, 2.0])


def test_plot_specgram() -> None:
    fs = 48_000.0
    t = np.arange(0, 0.5, 1 / fs)
    y = typing.cast(_Array1D, np.sin(2 * np.pi * 1000 * t))
    fig = Figure()
    spec, freqs, times = fig.plot_specgram(y, fs=fs, nfft=1024)
    assert fig.data[0].type == "heatmap"
    assert spec.shape == (freqs.shape[0], times.shape[0])
    assert fig.layout.yaxis.title.text == "Frequency [kHz]"
    assert fig.layout.xaxis.title.text == "Time [sec]"
    np.testing.assert_allclose(fig.data[0].y, freqs / 1e3)


def test_plot_surf_1d_inputs_meshgridded() -> None:
    x = _arr([0, 1, 2])
    y = _arr([0, 1])
    z = typing.cast(np.ndarray[tuple[int, int], np.dtype[typing.Any]], np.ones((2, 3)))
    fig = Figure(subplot_type="scene")
    fig.plot_surf(x, y, z)
    assert fig.data[0].type == "surface"
    assert np.asarray(fig.data[0].z).shape == (2, 3)


def test_plot_surf_callable_z() -> None:
    fig = Figure(subplot_type="scene")
    fig.plot_surf(_arr([0, 1, 2]), _arr([0, 1, 2]), lambda x_grid, y_grid: x_grid + y_grid)
    assert np.asarray(fig.data[0].z).shape == (3, 3)


def test_plot_surf_square_grid_not_transposed() -> None:
    x = _arr([0, 1, 2])
    y = _arr([10, 11, 12])
    z = typing.cast(
        np.ndarray[tuple[int, int], np.dtype[typing.Any]],
        np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]], dtype=np.float64),
    )
    fig = Figure(subplot_type="scene")
    fig.plot_surf(x, y, z)
    np.testing.assert_allclose(np.asarray(fig.data[0].z), z)


def test_plot_contour_int_levels() -> None:
    fig = Figure()
    fig.plot_contour(
        _arr([0, 1, 2]),
        _arr([0, 1]),
        typing.cast(np.ndarray[tuple[int, int], np.dtype[typing.Any]], np.arange(6, dtype=np.float64).reshape(2, 3)),
        contours=5,
    )
    assert fig.data[0].type == "contour"
    assert fig.data[0].ncontours == 5
    assert fig.data[0].contours.showlabels is True


def test_plot_contour_explicit_levels() -> None:
    fig = Figure()
    fig.plot_contour(
        _arr([0, 1, 2]),
        _arr([0, 1]),
        typing.cast(np.ndarray[tuple[int, int], np.dtype[typing.Any]], np.arange(6, dtype=np.float64).reshape(2, 3)),
        contours=(0.0, 5.0, 1.0),
    )
    assert fig.data[0].contours.start == 0.0
    assert fig.data[0].contours.size == 1.0


def test_plot_animation_images() -> None:
    data = typing.cast(
        np.ndarray[tuple[int, int, int], np.dtype[typing.Any]],
        np.random.default_rng(0).random((5, 8, 8)),
    )
    fig = Figure()
    fig.plot_animation(data)
    assert fig.data[0].type == "heatmap"
    assert len(fig.frames) == 5
    assert fig.layout.updatemenus[0].buttons[0].label == "Play"
    assert len(fig.layout.sliders[0].steps) == 5


def test_plot_animation_lines_with_titles() -> None:
    n_pts = 20
    data = typing.cast(
        np.ndarray[tuple[int, int, int], np.dtype[typing.Any]],
        np.stack([np.stack([np.arange(n_pts), np.sin(np.arange(n_pts) + i)]) for i in range(3)]),
    )
    fig = Figure()
    fig.plot_animation(data, trace_type="lines", titles=lambda i: f"frame {i}")
    assert fig.data[0].type == "scatter"
    assert len(fig.frames) == 3
    assert fig.frames[1].layout.title.text == "frame 1"


def test_plot_animation_ambiguous_shape_raises() -> None:
    data = typing.cast(
        np.ndarray[tuple[int, int, int], np.dtype[typing.Any]],
        np.zeros((4, 2, 10)),
    )
    fig = Figure()
    with pytest.raises(ValueError):
        fig.plot_animation(data)  # h == 2 is ambiguous; must pass trace_type
