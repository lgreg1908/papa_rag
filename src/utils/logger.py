from logging import Logger
import logging
import os
from dotenv import load_dotenv

load_dotenv()

log_file: str = os.environ.get('LOG_PATH', 'logs/app.log')

def get_logger(log_file: str, level: int = logging.INFO) -> Logger:
    """
    Configure and return a logger.

    Args:
        log_file:  Path to the log file.
        level:     Logging level (e.g., logging.INFO).

    Returns:
        Configured Logger instance.
    """
    logger: Logger = logging.getLogger(__name__)
    logger.setLevel(level)

    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger

logger = get_logger(log_file=log_file)
