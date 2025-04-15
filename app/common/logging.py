"""
Logging configuration for Astroshield.
"""

import logging
import os
import sys
from datetime import datetime

# Create logger
logger = logging.getLogger("astroshield")

def setup_logging(level=None, log_file=None):
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (default: from environment or INFO)
        log_file: Path to log file (default: from environment or None)
    """
    # Set log level from environment or use provided level or default to INFO
    log_level_env = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = level or getattr(logging, log_level_env, logging.INFO)
    
    # Get log file from environment if not provided
    log_file = log_file or os.environ.get("LOG_FILE")
    
    # Configure logger
    logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log file provided
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    logger.info(f"Logging initialized at level {logging.getLevelName(logger.level)}")
    
    return logger 