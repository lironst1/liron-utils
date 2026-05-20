import atexit
import inspect
import logging
import sys
import time
import typing
from logging.handlers import RotatingFileHandler
from types import TracebackType

from colorama import Back, Fore, Style

from .dicts import dict_

# todo: change to use loguru


class Logger:
    """Context-managed logger with a colored console handler and a rotating file handler.

    Examples:
        >>> with Logger(min_level_console=Logger.NAME2LEVEL["WARNING"]) as logger:
        ...     logger.log("Hello")  # Will not be printed
        ...     logger.log("World", level=logging.ERROR)  # Will be printed

        >>> logger = Logger()  # __enter__ is called automatically
        >>> logger.log("Hello")  # __exit__ called automatically on completion

        >>> logger = Logger()
        >>> raise ValueError("This is an error.")  # Logged via sys.excepthook
    """

    NAME2LEVEL = dict_(logging._nameToLevel)  # pylint: disable=protected-access
    LEVEL2NAME_OLD = dict_(logging._levelToName)  # pylint: disable=protected-access
    LEVEL2NAME = dict_(
        {
            logging.CRITICAL: "Fatal",
            logging.ERROR: "Error",
            logging.WARNING: "Warn",
            logging.INFO: "Info",
            logging.DEBUG: "Debug",
            logging.NOTSET: "Notset",
        },
    )
    ENTER_MSG = "STARTING LOGGER\t------------------------------------"
    EXIT_MSG = "CLOSING LOGGER\t------------------------------------"

    ENTER_CALLED = False

    def __init__(
        self,
        file_name: str = "./logs.log",
        *,
        min_level_file: int = logging.INFO,
        min_level_console: int | None = None,
        default_level: int = logging.INFO,
        log_message_format: str = "%(asctime)s"
        " | "
        "%(levelname)-14s"
        " | "
        "%(filename)-15s:%(lineno)4d"
        " | "
        "%(class_func)-30s"
        " >> "
        "%(message)s",
        max_file_size: int = 10 * 1024 * 1024,
        backup_count: int = 5,
    ) -> None:
        """Initialize the logger and install file and console handlers.

        Args:
            file_name: Logger file path; ``.log`` is appended if missing.
            min_level_file: Lowest level written to file.
            min_level_console: Lowest level written to console; ``None`` disables console output.
            default_level: Level used by ``log`` when no explicit level is provided.
            log_message_format: ``logging.Formatter`` format string applied to records.
            max_file_size: Rotating-file size limit in bytes.
            backup_count: Number of rotated backups to keep.

        Examples:
            >>> with Logger(min_level_console=Logger.NAME2LEVEL["WARNING"]) as logger:
            >>>      logger.log("Hello")  # Will not be printed
            >>>      logger.log("World", logging.ERROR)  # Will be printed

            >>> logger = Logger()  # __enter__ is called automatically
            >>> logger.log("Hello")  # __exit__ will be called automatically upon code completion

            >>> logger = Logger()
            >>> raise ValueError("This is an error.")  # Uncaught exception will be logged and the logger will be closed
        """
        if not file_name.endswith(".log"):
            file_name += ".log"

        with open(file_name, "a", encoding="utf-8"):  # Create the file if it doesn't exist
            pass
        self.logger = logging.getLogger(file_name)
        self.default_level = default_level
        self.logger.setLevel(logging.DEBUG)

        # Create a file handler to write log messages to a file
        class _Formatter(logging.Formatter):
            # Formatter that changes the level name to a shorthand version (e.g., WARNING->Warn).
            def format(self, record: logging.LogRecord) -> str:
                record.levelname = record.levelname.replace(
                    Logger.LEVEL2NAME_OLD[record.levelno],
                    Logger.LEVEL2NAME[record.levelno],
                )

                record.class_func = self._get_class_func(record)

                return super().format(record)

            @staticmethod
            def _get_class_func(record: logging.LogRecord) -> str:
                frame = inspect.currentframe()
                while frame:
                    if frame.f_code.co_name == record.funcName:
                        # Look for 'self' or 'cls' in the frame's local variables
                        local_self = frame.f_locals.get("self", None) or frame.f_locals.get("cls", None)
                        if local_self:
                            return f"{local_self.__class__.__name__}.{record.funcName}"
                    frame = frame.f_back
                return record.funcName  # Fallback

        # Use RotatingFileHandler instead of FileHandler for file size rotation
        self.file_handler = RotatingFileHandler(file_name, maxBytes=max_file_size, backupCount=backup_count)
        if min_level_file is not None:
            self.file_handler.setLevel(min_level_file)
            formatter = _Formatter(log_message_format)
            self.file_handler.setFormatter(formatter)
            self.logger.addHandler(self.file_handler)

        # Create a console handler to write log messages to the console
        class _FormatterColored(_Formatter):
            # Formatter that adds colors to log messages based on their level.
            COLORS = {
                logging.CRITICAL: Fore.LIGHTWHITE_EX + Back.LIGHTRED_EX,
                logging.ERROR: Fore.RED,
                logging.WARNING: Fore.YELLOW,
                logging.INFO: Fore.BLUE,
                logging.DEBUG: Fore.MAGENTA,
                logging.NOTSET: Fore.BLACK,
            }

            def format(self, record: logging.LogRecord) -> str:
                color = self.COLORS.get(record.levelno, "")
                record.levelname = f"{color}{record.levelname}{Style.RESET_ALL}"
                return super().format(record)

        self.console_handler = logging.StreamHandler(sys.stdout)
        if min_level_console is not None:
            self.console_handler.setLevel(min_level_console)
            formatter = _FormatterColored(log_message_format)
            self.console_handler.setFormatter(formatter)
            self.logger.addHandler(self.console_handler)

        atexit.register(self.__exit__, None, None, None)  # Register exit function

        excepthook_old = sys.excepthook  # Save the old exception hook

        def excepthook_new(
            exc_type: type[BaseException],
            exc_value: BaseException,
            exc_traceback: TracebackType | None,
        ) -> None:
            self.__exit__(exc_type, exc_value, exc_traceback)
            excepthook_old(exc_type, exc_value, exc_traceback)

        sys.excepthook = excepthook_new  # Set the new exception hook

        self.__enter__()

    def __enter__(self) -> "Logger":
        if self.ENTER_CALLED:
            return self

        self.info(self.ENTER_MSG)
        self.ENTER_CALLED = True  # pylint: disable=invalid-name
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: TracebackType | None,
    ) -> None:
        if exc_type is not None:
            if not issubclass(exc_type, KeyboardInterrupt):  # Ignore keyboard interrupts
                self.error("Uncaught exception:", exc_info=(exc_type, exc_value, exc_traceback))

        self.info(self.EXIT_MSG)
        self.logger.removeHandler(self.file_handler)
        self.file_handler.close()
        self.logger.removeHandler(self.console_handler)
        self.console_handler.close()

    def log(
        self,
        msg: str,
        *args: typing.Any,
        level: int | None = None,
        exc_info: bool = False,
        time_log: float | None = None,
        f: str = ":.3f",
        stacklevel: int = 2,
        **kwargs: typing.Any,
    ) -> float:
        """Emit a log message and return the timestamp at which it was emitted.

        Args:
            msg: Message string. If ``time_log`` is given, any ``{}`` placeholder
                is replaced with the elapsed time using format spec ``f``.
            *args: Forwarded to ``logging.Logger.log``.
            level: Logging level; defaults to ``self.default_level``.
            exc_info: If True, attach current exception info.
            time_log: Reference timestamp; if given, message includes elapsed time.
            f: Format spec used inside the elapsed-time placeholder.
            stacklevel: Forwarded to ``logging.Logger.log``.
            **kwargs: Forwarded to ``logging.Logger.log``.

        Returns:
            ``time.time()`` taken at the moment of logging.
        """
        if level is None:
            level = self.default_level

        kwargs = {"stacklevel": stacklevel} | kwargs

        tm = time.time()
        if time_log is not None:
            msg = msg.replace("{}", "{" + f + "}")
            msg = msg.format(tm - time_log)

        self.logger.log(level, msg, *args, exc_info=exc_info, **kwargs)

        return tm

    def debug(self, msg: str, *args: typing.Any, **kwargs: typing.Any) -> float:
        """Log ``msg`` at ``DEBUG`` level."""
        kwargs = {"stacklevel": 3} | kwargs
        return self.log(msg, *args, level=logging.DEBUG, **kwargs)

    def info(self, msg: str, *args: typing.Any, **kwargs: typing.Any) -> float:
        """Log ``msg`` at ``INFO`` level."""
        kwargs = {"stacklevel": 3} | kwargs
        return self.log(msg, *args, level=logging.INFO, **kwargs)

    def warning(self, msg: str, *args: typing.Any, **kwargs: typing.Any) -> float:
        """Log ``msg`` at ``WARNING`` level."""
        kwargs = {"stacklevel": 3} | kwargs
        return self.log(msg, *args, level=logging.WARNING, **kwargs)

    warn = warning

    def error(self, msg: str, *args: typing.Any, **kwargs: typing.Any) -> float:
        """Log ``msg`` at ``ERROR`` level."""
        kwargs = {"stacklevel": 3} | kwargs
        return self.log(msg, *args, level=logging.ERROR, **kwargs)

    def critical(self, msg: str, *args: typing.Any, **kwargs: typing.Any) -> float:
        """Log ``msg`` at ``CRITICAL`` level."""
        kwargs = {"stacklevel": 3} | kwargs
        return self.log(msg, *args, level=logging.CRITICAL, **kwargs)

    fatal = critical
