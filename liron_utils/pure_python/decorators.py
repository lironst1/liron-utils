import typing


def decorate_repr(cls: type) -> type:
    """Override ``cls.__repr__`` to print every instance attribute.

    Args:
        cls: Class to decorate (modified in place).

    Returns:
        The same class, with a new ``__repr__`` that lists all ``self.__dict__`` items.
    """

    def _repr(self: typing.Any) -> str:
        cls_name = self.__class__.__name__
        attrs = ",\n\t".join(f"{k} = {v!r}" for k, v in self.__dict__.items())
        return f"{cls_name}(\n\t{attrs}\n)"

    cls.__repr__ = _repr  # type: ignore[method-assign,assignment]
    return cls
