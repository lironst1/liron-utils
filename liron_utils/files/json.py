import typing
from pathlib import Path

import pandas as pd


def load_json(file: str | Path) -> typing.Any:
    """Load a JSON file and return its decoded contents.

    Args:
        file: Path to the JSON file.

    Returns:
        Decoded JSON (any value supported by ``json.load``).
    """
    # TODO replace with pd.read_json
    import json  # pylint: disable=import-outside-toplevel

    with open(file, "rb") as f:
        d = json.load(f)
    return d


def write_json(d: typing.Any, file: str | Path, *args: typing.Any, **kwargs: typing.Any) -> None:
    """Write a JSON-serializable object to disk via ``pandas.Series.to_json``.

    Args:
        d: JSON-serializable value to write.
        file: Destination file path.
        *args: Forwarded to ``pandas.Series.to_json``.
        **kwargs: Forwarded to ``pandas.Series.to_json`` (``indent=4`` by default).
    """
    kwargs = {"indent": 4} | kwargs
    pd.Series(d).to_json(file, *args, **kwargs)
