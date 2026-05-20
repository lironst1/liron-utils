import typing


@typing.overload
def hex2rgb(h: str) -> tuple[float, float, float]: ...


@typing.overload
def hex2rgb(h: list[str]) -> list[tuple[float, float, float]]: ...


def hex2rgb(h: str | list[str]) -> tuple[float, float, float] | list[tuple[float, float, float]]:
    """Convert a hex color string (or list thereof) to a 0-1 normalized RGB tuple.

    Args:
        h: Hex color string (e.g. ``"#FF5733"``), or a list of such strings.

    Returns:
        A single ``(r, g, b)`` tuple if ``h`` is a string, or a list of such
        tuples if ``h`` is a list. Each component is in ``[0, 1]``.

    Example:
        >>> hex2rgb("#FF5733")  # (1.0, 0.3411..., 0.2)
    """

    def hex2rgb_inner(h: str) -> tuple[float, float, float]:
        return typing.cast(
            tuple[float, float, float],
            tuple(int(h.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)),
        )

    if isinstance(h, str):
        return hex2rgb_inner(h)

    return [hex2rgb_inner(hh) for hh in h]


@typing.overload
def rgb2hex(rgb: tuple[float, float, float]) -> str: ...


@typing.overload
def rgb2hex(rgb: list[tuple[float, float, float]]) -> list[str]: ...


def rgb2hex(rgb: tuple[float, float, float] | list[tuple[float, float, float]]) -> str | list[str]:
    """Convert an RGB tuple (or list thereof) to a hex color string.

    Args:
        rgb: A 3-tuple of floats in ``[0, 1]``, or a list of such tuples.

    Returns:
        A hex color string (e.g. ``"#FF5733"``), or a list of such strings when
        ``rgb`` is a list.

    Example:
        >>> rgb2hex((1.0, 0.3411, 0.2))  # "#FF5733"
    """

    def rgb2hex_inner(rgb: tuple[float, float, float]) -> str:
        c = [int(v * 255) for v in rgb]
        return f"#{c[0]:02X}{c[1]:02X}{c[2]:02X}"

    if len(rgb) == 3 and isinstance(rgb[0], (int, float)):
        return rgb2hex_inner(rgb)

    return [rgb2hex_inner(r) for r in rgb]
