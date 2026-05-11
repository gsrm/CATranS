import logging
import sys

logger = logging.getLogger(__name__)
logger.propagate = False

class CustomFormatter(logging.Formatter):
    info_format = "[%(asctime)s] | %(levelname)-8s | %(message)s"
    debug_format = "[%(asctime)s] | %(levelname)-8s | STEP: %(funcName)s | %(message)s"

    def format(self, record):
        orig_levelname = record.levelname
        record.levelname = orig_levelname.center(8)
       
        if record.levelno == logging.DEBUG:
            self._style._fmt = self.debug_format
        else:
            self._style._fmt = self.info_format
        return super().format(record)

def _get_logger() -> logging.Logger:
    return logging.getLogger(name=__name__)

def _setup_internal_logger(verbose):
    # Si no hay handlers, creamos uno con el CustomFormatter
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(CustomFormatter(datefmt='%y-%m-%d %H:%M:%S'))
        logger.addHandler(handler)
    
    # Si el usuario activa verbose, bajamos a INFO
    # Si tú desde fuera pusiste DEBUG, esto no lo sobrescribe (porque DEBUG < INFO)
    if verbose:
        # if logger.level > logging.INFO or logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)
    else:
        if logger.level == logging.NOTSET or logger.level == logging.INFO:
            logger.setLevel(logging.WARNING)