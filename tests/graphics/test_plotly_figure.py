import pathlib

import plotly.graph_objects as go
import pytest

from liron_utils.graphics.plotly import Figure


def _num_xaxes(fig: go.Figure) -> int:
    return sum(1 for key in fig.to_dict()["layout"] if key.startswith("xaxis"))


def test_default_single_subplot() -> None:
    fig = Figure()
    assert isinstance(fig, go.Figure)
    fig.add_scatter(x=[1, 2], y=[3, 4], row=1, col=1)
    assert len(fig.data) == 1


def test_grid_shape() -> None:
    fig = Figure(shape=(2, 3))
    assert _num_xaxes(fig) == 6
    fig.add_scatter(x=[1], y=[1], row=2, col=3)
    assert len(fig.data) == 1


def test_span_layout() -> None:
    # 2x3 grid: one 2x2 span at top-left, singles at (0,2) and (1,2) -> 3 subplots.
    fig = Figure(shape=(2, 3), span_layout=[[(0, 2), (0, 2)], [0, (2, 3)]])
    assert _num_xaxes(fig) == 3


def test_span_layout_int_promotes_to_range() -> None:
    fig = Figure(shape=(2, 2), span_layout=[[0, (0, 2)]])
    assert _num_xaxes(fig) == 3  # top span + two bottom cells


def test_shared_xaxes() -> None:
    fig = Figure(shape=(2, 1), shared_xaxes=True)
    assert fig.layout.xaxis2.matches == "x" or fig.layout.xaxis.matches == "x2"


def test_scene_subplot_accepts_surface() -> None:
    fig = Figure(subplot_type="scene")
    fig.add_trace(go.Surface(z=[[1, 2], [3, 4]]), row=1, col=1)
    assert len(fig.data) == 1


def test_surface_on_xy_subplot_raises() -> None:
    fig = Figure()
    with pytest.raises(ValueError):
        fig.add_trace(go.Surface(z=[[1, 2], [3, 4]]), row=1, col=1)


def test_save_html(tmp_path: pathlib.Path) -> None:
    fig = Figure()
    fig.add_scatter(x=[1, 2], y=[3, 4])
    out = fig.save(str(tmp_path / "fig.html"))
    assert out == str(tmp_path / "fig.html")
    assert (tmp_path / "fig.html").stat().st_size > 0


def test_save_defaults_to_html(tmp_path: pathlib.Path) -> None:
    fig = Figure()
    out = fig.save(str(tmp_path / "figure"))
    assert out.endswith(".html")
    assert pathlib.Path(out).exists()


def test_save_png(tmp_path: pathlib.Path) -> None:
    pytest.importorskip("kaleido")
    fig = Figure()
    fig.add_scatter(x=[1, 2], y=[3, 4])
    try:
        out = fig.save(str(tmp_path / "fig.png"))
    except Exception as error:  # noqa: BLE001  # kaleido needs a Chrome binary; skip when absent
        pytest.skip(f"static image export unavailable: {error}")
    assert pathlib.Path(out).stat().st_size > 0
