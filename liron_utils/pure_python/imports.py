import os
import sys
import types


def import_module(file_name: str) -> types.ModuleType:
    """Import a Python file as a module by absolute or relative path.

    Args:
        file_name: Path to a Python source file. The containing directory is
            temporarily added to ``sys.path``.

    Returns:
        The imported module object.
    """
    dirname, file_name = os.path.split(file_name)
    sys.path.append(dirname)
    h = __import__(file_name)
    sys.path.pop(-1)
    return h
