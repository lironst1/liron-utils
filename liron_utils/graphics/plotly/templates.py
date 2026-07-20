import typing

import plotly.graph_objects as go
import plotly.io as pio

from ..common import COLORS

COLORWAY: list[str] = list(COLORS.DEFAULT_COLOR_CYCLE)

_TRANSPARENT = "rgba(0, 0, 0, 0)"


def _axis_defaults(**overrides: typing.Any) -> dict[str, typing.Any]:
    return {
        "showgrid": True,
        "gridcolor": COLORS.LIGHT_GREY,
        "showline": True,
        "linecolor": COLORS.BLACK,
        "mirror": False,
        "zeroline": False,
        "ticks": "outside",
        "exponentformat": "power",
        "title": {"font": {"size": 15}},
    } | overrides


def _template_default() -> go.layout.Template:
    return go.layout.Template(
        layout={
            "colorway": COLORWAY,
            "font": {"size": 13},
            "title": {"font": {"size": 19}},
            "width": 800,
            "height": 800,
            "paper_bgcolor": _TRANSPARENT,
            "plot_bgcolor": _TRANSPARENT,
            "xaxis": _axis_defaults(),
            "yaxis": _axis_defaults(),
        },
        data={
            "surface": [go.Surface(colorscale="Viridis")],
            "heatmap": [go.Heatmap(colorscale="Inferno")],
        },
    )


def _template_article() -> go.layout.Template:
    axis = {
        "showgrid": False,
        "mirror": True,
        "ticks": "inside",
        "tickfont": {"size": 13},
        "title": {"font": {"size": 14}},
    }
    return go.layout.Template(
        layout={
            "font": {"size": 13},
            "title": {"font": {"size": 16}},
            "width": 1040,
            "height": 715,
            "xaxis": axis,
            "yaxis": axis,
        },
    )


def _template_notebook() -> go.layout.Template:
    axis = {"tickfont": {"size": 10}, "title": {"font": {"size": 11}}}
    return go.layout.Template(
        layout={
            "font": {"size": 11},
            "title": {"font": {"size": 13}},
            "width": 640,
            "height": 480,
            "xaxis": axis,
            "yaxis": axis,
        },
    )


def text_color_template(color: str) -> go.layout.Template:
    """Build an overlay template that recolors all text (titles, labels, ticks, legend).

    Args:
        color: Any CSS color string.

    Returns:
        Template to combine with a base, e.g.
        ``pio.templates.default = "liron-utils-default+my-color"`` after registering it,
        or by passing ``template=...`` to ``update_layout``.
    """
    return go.layout.Template(layout={"font": {"color": color}})


def register_templates() -> None:
    """Register the liron-utils templates in ``plotly.io.templates``."""
    pio.templates["liron-utils-default"] = _template_default()
    pio.templates["liron-utils-article"] = _template_article()
    pio.templates["liron-utils-notebook"] = _template_notebook()
