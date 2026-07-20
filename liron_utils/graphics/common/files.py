import os

from ...files import MAIN_FILE_DIR, mkdirs
from ...time import TIME_STR, get_time_str


def get_savefig_file_name(
    file_name: str | None = None,
    time_dir: bool = False,
    mkdir: bool = False,
) -> str:
    """Build the full path for saving a figure.

    Args:
        file_name: If it has no directory component, the path resolves under
            ``<MAIN_FILE_DIR>/figs``. If None, an auto-generated ``"fig <ts>"`` name
            is used.
        time_dir: If True, append a timestamped subdirectory.
        mkdir: Create the directory if it doesn't exist.

    Returns:
        Resolved absolute (or near-absolute) file path.
    """
    if file_name is None or os.path.dirname(file_name) == "":
        dir_name = os.path.join(MAIN_FILE_DIR, "figs")
    else:
        dir_name = os.path.dirname(file_name)

    if time_dir:
        dir_name = os.path.join(dir_name, TIME_STR)

    if mkdir and not os.path.exists(dir_name):
        mkdirs(dir_name)

    if file_name is None:
        file_name = os.path.join(dir_name, f"fig {get_time_str()}")
    elif os.path.dirname(file_name) == "":
        file_name = os.path.join(dir_name, file_name)

    return file_name
