"""
Logging Utilities Module

This module provides enhanced logging capabilities for AstroShield with traceability support.
It ensures consistent logging format and includes trace IDs in log messages for easy correlation.
"""

import logging
import threading
import uuid
from typing import Optional, Any, Dict, Callable
from contextlib import contextmanager


# Thread-local storage for trace IDs
_thread_local = threading.local()


def get_current_trace_id() -> str:
    """
    Get the current trace ID from thread-local storage.
    
    Returns:
        str: Current trace ID or "NO_TRACE" if none is set
    """
    return getattr(_thread_local, 'trace_id', 'NO_TRACE')


def set_current_trace_id(trace_id: str) -> None:
    """
    Set the current trace ID in thread-local storage.
    
    Args:
        trace_id: The trace ID to set
    """
    _thread_local.trace_id = trace_id


@contextmanager
def trace_context(trace_id: Optional[str] = None) -> None:
    """
    Context manager for setting the current trace ID within a context block.
    
    Args:
        trace_id: Trace ID to use (generates a new one if None)
        
    Example:
        with trace_context('abc-123'):
            # All logs within this block will include the trace ID
            logger.info("Processing message")
    """
    if trace_id is None:
        trace_id = str(uuid.uuid4())
    
    old_trace_id = getattr(_thread_local, 'trace_id', None)
    _thread_local.trace_id = trace_id
    
    try:
        yield
    finally:
        if old_trace_id is not None:
            _thread_local.trace_id = old_trace_id
        else:
            delattr(_thread_local, 'trace_id')


class TraceIDFilter(logging.Filter):
    """Filter that adds trace ID to log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Add trace ID to log record.
        
        Args:
            record: Log record to modify
            
        Returns:
            bool: Always True (to keep the record)
        """
        record.trace_id = get_current_trace_id()
        return True


class TracedLogger(logging.LoggerAdapter):
    """
    Logger adapter that automatically includes trace IDs in log messages.
    
    This adapter ensures that all logs include the current trace ID,
    even if it changes during execution.
    """
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """
        Process the log message and add trace ID.
        
        Args:
            msg: Log message
            kwargs: Additional logging arguments
            
        Returns:
            tuple: (modified_message, modified_kwargs)
        """
        kwargs["extra"] = kwargs.get("extra", {})
        kwargs["extra"]["trace_id"] = get_current_trace_id()
        return msg, kwargs


def configure_logging(
    level: int = logging.INFO,
    log_format: str = '%(asctime)s - %(name)s - [%(trace_id)s] - %(levelname)s - %(message)s'
) -> None:
    """
    Configure the logging system with trace ID support.
    
    Args:
        level: Logging level (default: INFO)
        log_format: Log format string (default includes trace ID)
    """
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create and configure stream handler
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)
    
    # Add trace ID filter
    handler.addFilter(TraceIDFilter())
    root_logger.addHandler(handler)


def get_logger(name: str) -> TracedLogger:
    """
    Get a logger with trace ID support.
    
    Args:
        name: Logger name
        
    Returns:
        TracedLogger: Logger with trace ID support
    """
    logger = logging.getLogger(name)
    return TracedLogger(logger, {})


def trace_method(func: Callable) -> Callable:
    """
    Decorator to add trace context to methods.
    
    This decorator ensures that methods either use an existing trace ID
    or create a new one if none exists.
    
    Args:
        func: Function to decorate
        
    Returns:
        Callable: Decorated function with trace context
    """
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Wrapper function that adds trace context."""
        # Use existing trace ID or generate a new one
        trace_id = get_current_trace_id()
        if trace_id == 'NO_TRACE':
            trace_id = str(uuid.uuid4())
        
        with trace_context(trace_id):
            return func(*args, **kwargs)
    
    return wrapper


# Example usage:
# configure_logging()
# logger = get_logger(__name__)
# 
# with trace_context('abc-123'):
#     logger.info("Processing message with trace context")
# 
# @trace_method
# def process_data():
#     logger.info("Processing data with trace decoration")
#     
# process_data() 