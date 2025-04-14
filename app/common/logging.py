"""
Centralized logging configuration
"""
import logging

def setup_logging():
    """Configure application-wide logging"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("astroshield")
    return logger

# Create a logger that can be imported by other modules
logger = setup_logging() 