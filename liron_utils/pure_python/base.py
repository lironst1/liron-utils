import inspect
import sys
import typing

is_debugger = sys.gettrace() is not None


def class_vars(cls: type) -> dict[str, typing.Any]:
    """Return the (non-magic, non-callable) class attributes of ``cls``.

    Args:
        cls: The class to introspect.

    Returns:
        Mapping of attribute name to its value, skipping magic methods and any
        attribute that is a method, function, or raises on access.
    """
    all_vars: dict[str, typing.Any] = {}
    for name in dir(cls):
        if name.startswith("__"):
            continue
        try:
            attr = getattr(cls, name)
            if not inspect.ismethod(attr) and not inspect.isfunction(attr):
                all_vars[name] = attr
        except Exception:  # pylint: disable=broad-exception-caught
            pass
    return all_vars


def is_notebook() -> bool:
    """Return True when running inside an IPython/Jupyter notebook kernel."""
    try:
        from IPython import (  # type: ignore[attr-defined]  # pylint: disable=import-outside-toplevel
            get_ipython,
        )

        shell = get_ipython()  # type: ignore[no-untyped-call]
        if shell is not None:
            return typing.cast(str, shell.__class__.__name__) == "ZMQInteractiveShell"
        return False
    except ImportError:
        return False
