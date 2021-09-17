import logging
from datetime import datetime

LOG_FORMAT = "[%(asctime)-15s] {%(name)s:%(lineno)d} %(levelname)s - %(message)s"


def configure_parent_logger(file_name: str = None):
    """
    Configure handler and formatter for the top-level (parent) logger.

    Create child loggers for modules within this package as needed:
    logger = logging.getLogger(__name__)
    """
    logger = logging.getLogger(__package__)
    # set the effective level of the logger to lowest level (DEBUG) and configure handlers however we want
    logger.setLevel(logging.DEBUG)

    # set formatter
    formatter = logging.Formatter(LOG_FORMAT)

    # set stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # set file handler
    if file_name:
        stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        file_handler = logging.FileHandler(f"../logs/{stamp}_{file_name}.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
