import plotly.io as pio

from liron_utils.graphics.common import COLORS

# Importing anything from the plotly subpackage runs its __init__, which registers the templates.
from liron_utils.graphics.plotly.templates import COLORWAY, text_color_template


def test_templates_registered() -> None:
    for name in ("liron-utils-default", "liron-utils-article", "liron-utils-notebook"):
        assert name in pio.templates


def test_default_template_active() -> None:
    assert str(pio.templates.default).startswith("liron-utils-default")


def test_default_template_colorway() -> None:
    template = pio.templates["liron-utils-default"]
    assert list(template.layout.colorway) == COLORWAY
    assert COLORWAY[0] == COLORS.DARK_BLUE


def test_default_template_trace_defaults() -> None:
    template = pio.templates["liron-utils-default"]
    assert len(template.data.surface) == 1
    assert len(template.data.heatmap) == 1


def test_text_color_template() -> None:
    template = text_color_template("#FF0000")
    assert template.layout.font.color == "#FF0000"
