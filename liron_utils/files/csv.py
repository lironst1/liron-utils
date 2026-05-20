import typing
from pathlib import Path

import numpy as np
import pandas as pd
from uncertainties import unumpy

_Array = np.ndarray[typing.Any, np.dtype[typing.Any]]

load_csv = pd.read_csv


def load_csv_to_dict(
    file: str | Path,
    *args: typing.Any,
    **kwargs: typing.Any,
) -> dict[str, _Array]:
    """Load a CSV file as a ``{column: ndarray}`` mapping.

    Args:
        file: CSV file path.
        *args: Forwarded to ``pandas.read_csv``.
        **kwargs: Forwarded to ``pandas.read_csv``.

    Returns:
        Dictionary mapping each column name to its values as an ndarray.
    """
    table = pd.read_csv(file, *args, **kwargs)
    return {column: table[column].to_numpy() for column in table.columns}


def load_csv_to_dict_with_uncertainties(
    file: str | Path,
    *args: typing.Any,
    dev_str_identifier: str = " dev",
    **kwargs: typing.Any,
) -> dict[str, typing.Any]:
    """Load a CSV file into a ``{column: unumpy.uarray}`` mapping.

    Each pair of columns ``<name>`` and ``<name><dev_str_identifier>`` is combined
    into a single ``uncertainties`` array keyed by ``<name>``.

    Args:
        file: CSV file path.
        *args: Forwarded to ``pandas.read_csv`` (via ``load_csv_to_dict``).
        dev_str_identifier: Suffix that marks a column of standard deviations.
        **kwargs: Forwarded to ``pandas.read_csv`` (via ``load_csv_to_dict``).

    Returns:
        Dictionary mapping each base column name to its uncertainties uarray.
    """
    d = load_csv_to_dict(file, *args, **kwargs)
    d_uncertainties: dict[str, typing.Any] = {}

    for key, value in d.items():
        if key.endswith(dev_str_identifier):
            key_wo_dev = key.split(dev_str_identifier)[0]
            d_uncertainties[key_wo_dev] = unumpy.uarray(d[key_wo_dev], value)

    return d_uncertainties
