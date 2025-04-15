"""
Error handling and retry utilities for AstroShield.

This module provides classes and functions for robust error handling
and automatic retry of operations with configurable backoff strategies.
"""

import logging
import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type, Union

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError
)

logger = logging.getLogger(__name__)

class RetryHandler:
    """Handler for retrying operations with configurable backoff."""
    
    @staticmethod
    def with_exponential_backoff(
        max_attempts: int = 3,
        max_wait: int = 10,
        exception_types: List[Type[Exception]] = None
    ) -> Callable:
        """
        Decorator for retrying a function with exponential backoff.
        
        Args:
            max_attempts: Maximum number of retry attempts
            max_wait: Maximum wait time between retries in seconds
            exception_types: List of exception types to retry on (defaults to Exception)
            
        Returns:
            Decorated function with retry logic
        """
        if exception_types is None:
            exception_types = [Exception]
            
        retry_on_exceptions = retry_if_exception_type(tuple(exception_types))
        
        def decorator(func):
            @wraps(func)
            @retry(
                stop=stop_after_attempt(max_attempts),
                wait=wait_exponential(multiplier=1, max=max_wait),
                retry=retry_on_exceptions,
                reraise=True
            )
            async def async_wrapper(*args, **kwargs):
                return await func(*args, **kwargs)
                
            @wraps(func)
            @retry(
                stop=stop_after_attempt(max_attempts),
                wait=wait_exponential(multiplier=1, max=max_wait),
                retry=retry_on_exceptions,
                reraise=True
            )
            def sync_wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return sync_wrapper
            
        return decorator


class ErrorHandler:
    """Handles errors with configurable strategies and reporting."""
    
    def __init__(
        self,
        log_errors: bool = True,
        raise_errors: bool = True,
        error_callback: Optional[Callable[[Exception], Any]] = None
    ):
        """
        Initialize the error handler.
        
        Args:
            log_errors: Whether to log errors
            raise_errors: Whether to re-raise errors after handling
            error_callback: Optional callback function to call when an error occurs
        """
        self.log_errors = log_errors
        self.raise_errors = raise_errors
        self.error_callback = error_callback
        
    def handle(self, error: Exception, context: Dict[str, Any] = None) -> None:
        """
        Handle an error according to the configured strategy.
        
        Args:
            error: The exception to handle
            context: Optional dictionary of context information
        
        Raises:
            Exception: The original exception if raise_errors is True
        """
        if context is None:
            context = {}
            
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context
        }
        
        if self.log_errors:
            logger.error(
                f"Error occurred: {error_info['error_type']}: {error_info['error_message']}",
                extra={"context": context}
            )
            
        if self.error_callback:
            try:
                self.error_callback(error)
            except Exception as callback_error:
                logger.error(f"Error in error callback: {callback_error}")
                
        if self.raise_errors:
            raise error
            
    def __call__(self, func):
        """
        Decorator to handle errors from a function.
        
        Args:
            func: The function to wrap with error handling
            
        Returns:
            Wrapped function with error handling
        """
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                self.handle(e, {"args": args, "kwargs": kwargs})
                return None
                
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self.handle(e, {"args": args, "kwargs": kwargs})
                return None
                
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper


import asyncio 