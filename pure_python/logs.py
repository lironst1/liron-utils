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
			print_to_console_level: (None, int) = None,
			default_level: int = logging.INFO,
			log_message_format: str = '%(asctime)s | %(levelname)5s >> %(message)s'
	):
		"""
		Initialize logger.

		Parameters
		----------
		file_name :                 str
			Path logger file (should end with .log)
		print_to_console_level :    int
			Level of messages to print to console. Default is INFO
		default_level :             int
			Default logging level, if no level is provided. Default is INFO
		log_message_format :        str
			Default string format of a message
		"""
		with open(file_name, 'a'):
			pass
		self.logger = logging.getLogger(file_name)
		self.default_level = default_level
		self.logger.setLevel(default_level)

		self.file_handler = logging.FileHandler(file_name)
		formatter = logging.Formatter(log_message_format)
		self.file_handler.setFormatter(formatter)

		self.logger.addHandler(self.file_handler)

		self.print_to_console_level = print_to_console_level

	def __enter__(self):
		self.log("STARTING LOGGER...")
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		self.log("CLOSING LOGGER...")
		self.logger.removeHandler(self.file_handler)
		self.file_handler.close()

	def log(self,
			msg: str,
			level: int = None,
			exc_info: bool = False,
			time_log: float = None,
			f: str = ':.3f', *args, **kwargs):
		"""
		Send a log message to the log file.

		Parameters
		----------
		msg :               str
			Message string
		level :             int
			Logging level (one of self.LEVELS)
		exc_info :          bool
			Error info
		time_log :          float
			Time log
		f :                 str
			Format string for time log
		*args, **kwargs :   Passed to Logger.log

		**kwargs :      dict
			Passed to Logger.log

		Returns
		-------

		"""

		if level is None:
			level = self.default_level

		tm = time.time()
		if time_log is not None:
			msg = msg.replace('{}', '{' + f + '}')
			msg = msg.format(tm - time_log)

		self.logger.log(level, msg, exc_info=exc_info, *args, **kwargs)

		# Print to console if level is above the threshold
		if self.print_to_console_level is not None and level >= self.print_to_console_level:
			record = self.logger.makeRecord(
					name=self.logger.name,
					level=level,
					fn='', lno=0, msg=msg,
					exc_info=exc_info,
					args=args,
					extra=kwargs
			)
			final_msg = self.logger.handlers[0].formatter.format(record)
			print(final_msg)

		return tm
