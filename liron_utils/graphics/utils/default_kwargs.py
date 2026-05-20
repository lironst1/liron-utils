import typing

from scipy.signal import windows

from ...pure_python import MetaDict
from . import COLORS


class _FuncDefaultKwargs(dict[str, typing.Any]):
    def __init__(self, **kwargs: typing.Any) -> None:
        super().__init__(**kwargs)


class DefaultKwargs(metaclass=MetaDict):
    # Figure
    FIG_KW = _FuncDefaultKwargs()

    # Axes
    SET_PROPS_KW = _FuncDefaultKwargs(
        sup_title=None,  # str
        ax_title=None,  # list(str)
        axis=None,  # list(bool)
        spines=None,  # list(bool)
        ticks=None,  # list(bool/list)
        tick_labels=None,  # list(bool/list)
        labels=None,  # list(list(str))
        limits=None,  # list(list(float))
        view=None,  # list(list(float))
        grid=None,  # list(bool)
        legend=True,  # list(bool/list(str))
        legend_loc=None,  # list(str)
        colorbar=False,  # list(bool/list)
        xy_lines=False,  # list(bool)
        face_color=None,  # list(color)
        show_fig=True,  # bool
        open_dir=False,  # bool
        close_fig=False,  # bool
    )

    XY_LINES_KW = _FuncDefaultKwargs(color=COLORS.DARK_GREY, linewidth=2)

    # 2D Plotting
    PLOT_KW = _FuncDefaultKwargs()

    ERRORBAR_KW = _FuncDefaultKwargs(linestyle="none", marker=".", markersize=10, ecolor=COLORS.RED_E, elinewidth=1.4)

    FILL_BETWEEN_KW = _FuncDefaultKwargs(linestyle="-", color=COLORS.LIGHT_GRAY, alpha=0.4)

    SPECGRAM_KW = _FuncDefaultKwargs(
        NFFT=4096,
        window=windows.blackmanharris(4096),
        noverlap=int(0.85 * 4096),
        pad_to=4096 + int(1 * 4096),
        cmap="inferno",
    )

    # 3D Plotting
    PLOT_SURFACE_KW = _FuncDefaultKwargs(
        cmap="viridis",
    )


def update_kwargs(key: str | None = None, **kwargs: typing.Any) -> type[DefaultKwargs]:
    """Update the defaults stored on ``DefaultKwargs`` in place.

    Args:
        key: When given, update ``DefaultKwargs[key.upper()]`` with ``kwargs``. When None,
            iterate ``kwargs`` and update ``DefaultKwargs[k.upper()]`` with each value.
        **kwargs: New defaults to merge in.

    Returns:
        The ``DefaultKwargs`` class itself (mutated in place).
    """
    if key:
        typing.cast(_FuncDefaultKwargs, DefaultKwargs[key.upper()]).update(**kwargs)
    else:
        for k in kwargs:  # pylint: disable=consider-using-dict-items
            typing.cast(_FuncDefaultKwargs, DefaultKwargs[k.upper()]).update(**kwargs[k])
    return DefaultKwargs


def merge_kwargs(**kwargs: typing.Any) -> dict[str, typing.Any]:
    """Merge each provided kwargs dict with its ``DefaultKwargs`` counterpart.

    For each ``key=value`` in ``kwargs``, returns ``DefaultKwargs[key.upper()] | value``,
    so values supplied by the caller win over defaults. ``None`` values are treated as
    empty dicts.

    Args:
        **kwargs: Mapping of ``<name>=<partial-kwargs-dict-or-None>`` to merge.

    Returns:
        Dict with the same keys as ``kwargs``, each mapped to the merged dict.
    """
    result: dict[str, typing.Any] = {}
    for key in kwargs:  # pylint: disable=consider-using-dict-items
        value = kwargs[key]
        if value is None:
            value = {}
        result[key] = DefaultKwargs[key.upper()] | value
    return result
