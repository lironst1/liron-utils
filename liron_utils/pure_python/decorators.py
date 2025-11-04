def decorate_repr(cls):
    """
    Change __repr__ method to print all attributes of the class.

    Parameters
    ----------
    cls

    Returns
    -------

    """

    def __repr__(self):
        cls_name = self.__class__.__name__
        attrs = ",\n\t".join(f"{k} = {v!r}" for k, v in self.__dict__.items())
        return f"{cls_name}(\n\t{attrs}\n)"

    cls.__repr__ = __repr__
    return cls
