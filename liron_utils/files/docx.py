# TODO make/edit Word documents

import typing
from pathlib import Path


def load_docx(file: str | Path) -> typing.Any:
    """Load a ``.docx`` file as a ``python-docx`` Document.

    Args:
        file: Path to the ``.docx`` file.

    Returns:
        The opened ``docx.Document`` object.
    """
    import docx  # type: ignore[import-not-found]  # pylint: disable=import-outside-toplevel,import-error

    return docx.Document(str(file))
