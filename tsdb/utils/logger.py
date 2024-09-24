import logging
from logging.handlers import RotatingFileHandler
from logging import Logger

from pathlib import Path

def setup_logger(log_path: str, logger_name: str) -> tuple[Logger, RotatingFileHandler]:
    """
    Creates and returns a Logger object

    Args:
        log_path: Path to store the log file
    Returns:
        The Logger object and the RotatingFileHandler object
    """
    # TODO: Log file may become too large, may need to be partitioned by date/week/month

    # Create the log directory if it doesn't exist
    log_dir = str(Path(log_path).parent)

    # Set up logging
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    try:
        # Create a rotating file handler
        handler = RotatingFileHandler(log_path, maxBytes=1000000, backupCount=1)
        handler.setLevel(logging.INFO)

        # Create a logging format
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)

        # Add the handler to the logger and then you can use the logger
        logger.addHandler(handler)
    except Exception as e:
        print(f"Error setting up logging: {e}")
        raise e

    return logger, handler