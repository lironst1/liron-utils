# flake8: noqa: F401, F403

import os
import sys

from .csv import *
from .docx import *
from .files import *
from .json import *
from .pdf import *

__all__ = [s for s in dir() if not s.startswith("_")]

try:
    _main_file = sys.modules["__main__"].__file__
    MAIN_FILE_DIR = os.path.split(_main_file)[0] if _main_file is not None else ""
except AttributeError:
    MAIN_FILE_DIR = ""
