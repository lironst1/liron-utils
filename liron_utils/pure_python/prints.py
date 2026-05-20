from collections.abc import Sequence


def to_uint8(*args: float) -> tuple[int, ...]:
    """Convert float values to unsigned 8-bit integers.

    Each input is interpreted as already-uint8 (an integer in ``[0, 255]``) when
    it equals its int cast, or as a unit-interval value otherwise (scaled by 255).

    Args:
        *args: Float values in ``[0, 1]`` or integer values in ``[0, 255]``.

    Returns:
        Tuple of converted integers, one per input.

    Raises:
        ValueError: If any input is outside ``[0, 1]`` and ``[0, 255]``.
    """
    out = [0] * len(args)
    for i, arg in enumerate(args):
        if int(arg) == arg and 0 <= arg <= 255:
            out[i] = int(arg)
        elif 0 <= arg <= 1:
            out[i] = int(arg * 255)
        else:
            raise ValueError(f"Value {arg} is out of range [0, 1] or [0, 255] for conversion to uint8.")
    return tuple(out)


def print_in_color(
    text: str,
    foreground_rgb: Sequence[float] | None = None,
    background_rgb: Sequence[float] | None = None,
) -> str:
    """Wrap ``text`` in ANSI 24-bit color escapes.

    Args:
        text: Text to colorize.
        foreground_rgb: RGB triple for the foreground color (in ``[0, 1]`` or ``[0, 255]``).
        background_rgb: RGB triple for the background color (in ``[0, 1]`` or ``[0, 255]``).

    Returns:
        ANSI-decorated string, always terminated by a reset sequence.
    """
    out = text
    if foreground_rgb is not None:
        fg = to_uint8(*foreground_rgb)
        fg_strs = [str(c) for c in fg]
        out = f"\x1b[38;2;{';'.join(fg_strs)}m{text}"
    if background_rgb is not None:
        bg = to_uint8(*background_rgb)
        bg_strs = [str(c) for c in bg]
        out = f"\x1b[48;2;{';'.join(bg_strs)}m{text}"

    return f"{out}\x1b[0m"  # Reset all styles
