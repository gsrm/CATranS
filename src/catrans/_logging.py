import logging
import sys

logger = logging.getLogger(__name__)
logger.propagate = False

class CustomFormatter(logging.Formatter):
    """
    Custom log formatter providing clean, consistent formatting for CATranS operations.

    Formats INFO and WARNING messages with timestamp and log level, and enriches DEBUG
    messages with the calling function name (`STEP: <funcName>`).
    """

    info_format = "[%(asctime)s] | %(levelname)-8s | %(message)s"
    debug_format = "[%(asctime)s] | %(levelname)-8s | STEP: %(funcName)s | %(message)s"

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record with centered log-level tags and dynamic debug detail.

        Parameters:
            record: The logging.LogRecord instance to format.

        Returns:
            str: The formatted log line string.
        """
        orig_levelname = record.levelname
        record.levelname = orig_levelname.center(8)
       
        if record.levelno == logging.DEBUG:
            self._style._fmt = self.debug_format
        else:
            self._style._fmt = self.info_format
        return super().format(record)


def _get_logger() -> logging.Logger:
    """
    Retrieve the internal package logger for catrans.

    Returns:
        logging.Logger: The configured catrans logger instance.
    """
    return logging.getLogger(name=__name__)


def _setup_internal_logger(verbose: bool) -> None:
    """
    Configure the internal stream logger handler and dynamic log level.

    Initializes a standard output stream handler with `CustomFormatter` if no handlers
    are currently registered. Sets the log level to INFO when `verbose=True`, or WARNING
    when `verbose=False` (preserving DEBUG level if set externally).

    Parameters:
        verbose: If True, sets logger level to INFO; otherwise, sets to WARNING.
    """
    # If no handlers exist, attach a stream handler with CustomFormatter
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(CustomFormatter(datefmt='%y-%m-%d %H:%M:%S'))
        logger.addHandler(handler)
    
    # If verbose is enabled, set level to INFO
    # If the user explicitly configured DEBUG from the outside, do not overwrite it
    if verbose:
        logger.setLevel(logging.INFO)
    else:
        if logger.level == logging.NOTSET or logger.level == logging.INFO:
            logger.setLevel(logging.WARNING)