# flake8: noqa: F401

import plotly.io as pio

from ...pure_python import is_notebook
from .figure import Figure
from .templates import COLORWAY, register_templates, text_color_template

__all__ = ["COLORWAY", "Figure", "register_templates", "text_color_template"]

register_templates()

pio.templates.default = "liron-utils-default"
if is_notebook():
    pio.templates.default = "liron-utils-default+liron-utils-notebook"
