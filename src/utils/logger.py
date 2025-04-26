import os
import logging
from logging import Logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Determine log file path from environment or default
log_file: str = os.environ.get('LOG_PATH', 'logs/app.log')

# Ensure the log directory exists
log_dir = os.path.dirname(log_file)
if log_dir and not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)


def get_logger(log_file: str = log_file, level: int = logging.INFO) -> Logger:
    """
    Configure and return a logger.

    Args:
        log_file:  Path to the log file.
        level:     Logging level (e.g., logging.INFO).

    Returns:
        Configured Logger instance.
    """
    # Ensure the log directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logger: Logger = logging.getLogger(__name__)
    logger.setLevel(level)

    # Avoid adding multiple handlers to the logger
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_file)
               for h in logger.handlers):
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

# Module-level logger
logger = get_logger()
