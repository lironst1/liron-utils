from .files import *
from .json import *
from .csv import *

import os
import IPython

__all__ = [s for s in dir() if not s.startswith('_')]

try:
    MAIN_FILE_DIR = os.path.split(IPython.sys.modules['__main__'].__file__)[0]
except AttributeError:
    MAIN_FILE_DIR = ''
