import pathlib

import liron_utils.graphics


def test_graphics_root_is_matplotlib_free() -> None:
    """graphics/__init__ and graphics/common must not reference matplotlib.

    A runtime sys.modules check is impossible here: importing liron_utils.graphics
    executes liron_utils/__init__, which imports signal_processing -> fitting ->
    graphics.mpl -> matplotlib. So assert at the source level instead.
    """
    root = pathlib.Path(liron_utils.graphics.__file__).parent
    sources = [root / "__init__.py", *(root / "common").glob("*.py")]
    assert len(sources) >= 6
    for source_file in sources:
        assert "matplotlib" not in source_file.read_text(), source_file


def test_graphics_root_exports() -> None:
    from liron_utils import (
        graphics,  # pylint: disable=import-outside-toplevel,reimported
    )

    assert isinstance(graphics.COLORS.DARK_BLUE, str)
    assert graphics.hex2rgb("#FF0000") == (1.0, 0.0, 0.0)
    assert callable(graphics.get_savefig_file_name)
    assert callable(graphics.get_pixel_color)


def test_graphics_mpl_exports() -> None:
    from liron_utils.graphics import (  # pylint: disable=import-outside-toplevel
        mpl as gr,
    )

    assert hasattr(gr, "Axes")
    assert hasattr(gr, "set_props")
    assert hasattr(gr, "update_rc_params")
    assert hasattr(gr, "COLORS")
    axes = gr.Axes(shape=(2, 1))
    assert axes.axs.shape == (2, 1)
