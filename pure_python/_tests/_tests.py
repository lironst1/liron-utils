import logging
from liron_utils.pure_python import logs

file_name = "/Users/lironst/Desktop/tmp.log"

with logs.Logger(file_name) as logger:
    logger.log("Hello")
    logger.log("World", logging.ERROR)
pass
