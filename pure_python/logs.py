import logging
import time


class Logger:
    """
    Logger file object.

    Example:
        >> file_name = "logger.log"
        >> with logs.Logger(file_name) as logger:
        >>      logger.log("Hello")
        >>      logger.log("World", logging.ERROR)

    """

    LEVELS = logging._nameToLevel

    def __init__(self,
                 file_name: str,
                 print_to_console: bool = True,
                 default_level: int = logging.INFO,
                 log_message_format: str = '%(asctime)s | %(levelname)5s >> %(message)s'
                 ):
        """
        Start logger.

        Args:
            file_name ():               Path logger file (should end with .log)
            print_to_console ():        Whether to print to console or not
            default_level ():           Default logging level, if no level is provided. Default is INFO
            log_message_format ():      Default string format of a message
        """

        with open(file_name, 'a'):
            pass
        self.logger = logging.getLogger(file_name)
        self.logger.setLevel(default_level)

        self.file_handler = logging.FileHandler(file_name)
        formatter = logging.Formatter(log_message_format)
        self.file_handler.setFormatter(formatter)

        self.logger.addHandler(self.file_handler)

        self.print_to_console = print_to_console

    def __enter__(self):
        self.log("STARTING LOGGER...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.log("CLOSING LOGGER...")
        self.logger.removeHandler(self.file_handler)
        self.file_handler.close()

    def log(self,
            msg: str,
            level: int = logging.INFO,
            exc_info: bool = False,
            time_log: float = None,
            f: str = ':.3f', *args, **kwargs):
        """
        Send log message to log file

        Args:
            msg:            Message string
            level:          Logging level (one of self.LEVELS)
            exc_info:       Error info
            time_log:
            f:
            *args:          Passed to Logger.log
            **kwargs:       Passed to Logger.log

        Returns:

        """

        tm = time.time()
        if time_log is not None:
            msg = msg.replace('{}', '{' + f + '}')
            msg = msg.format(tm - time_log)

        if self.print_to_console:
            print(msg)

        self.logger.log(level, msg, exc_info=exc_info, *args, **kwargs)
        return tm
