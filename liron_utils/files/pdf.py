from collections.abc import Iterable
from pathlib import Path


def merge_pdf(out: str | Path, files: Iterable[str | Path]) -> None:
    """Merge multiple PDF files into a single output file.

    Args:
        out: Output PDF path.
        files: PDF input files to concatenate in order.
    """
    from PyPDF2 import (  # type: ignore[import-not-found]  # pylint: disable=import-outside-toplevel,import-error
        PdfFileMerger,
    )

    h = PdfFileMerger()
    for file in files:
        h.append(file)
    h.write(out)
    h.close()
