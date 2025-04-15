"""
Retry and backoff utilities for handling transient failures.

This module provides decorators and functions for retrying operations that may
fail due to transient errors, with configurable backoff strategies.
"""

import functools
import logging
import random
import time
from typing import Any, Callable, List, Optional, Type, TypeVar, Union, cast

from .exceptions import RetryExhaustedError

logger = logging.getLogger(__name__)

# Type variable for the function return type
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

def retry(
    max_attempts: int = 3,
    retry_exceptions: Union[Type[Exception], List[Type[Exception]]] = Exception,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    on_retry: Optional[Callable[[str, int, Exception, float], None]] = None,
) -> Callable[[F], F]:
    """
    Decorator to retry functions on exception with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts (default: 3)
        retry_exceptions: Exception or list of exceptions to retry on (default: Exception)
        base_delay: Initial delay between retries in seconds (default: 1.0)
        max_delay: Maximum delay between retries in seconds (default: 60.0)
        backoff_factor: Multiplier for the delay on each retry (default: 2.0)
        jitter: Whether to add randomness to the delay (default: True)
        on_retry: Optional callback function called before each retry with parameters:
                  (operation_name, attempt_number, exception, next_delay)
    
    Returns:
        A decorated function that will retry on failure
    
    Example:
        @retry(max_attempts=5, retry_exceptions=[ConnectionError, TimeoutError])
        def fetch_data(url):
            # code that might raise transient errors
            return requests.get(url)
    """
    if not isinstance(retry_exceptions, list):
        retry_exceptions = [retry_exceptions]
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            operation_name = func.__qualname__
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except tuple(retry_exceptions) as exc:
                    if attempt == max_attempts:
                        logger.error(
                            "All %d retry attempts exhausted for operation '%s'",
                            max_attempts, operation_name
                        )
                        error_message = f"All retry attempts exhausted for operation '{operation_name}' after {max_attempts} attempts"
                        raise RetryExhaustedError(
                            message=error_message,
                            operation=operation_name,
                            attempts=max_attempts,
                            details={"last_error": str(exc)}
                        ) from exc
                    
                    # Calculate backoff delay with exponential backoff
                    delay = min(base_delay * (backoff_factor ** (attempt - 1)), max_delay)
                    
                    # Add jitter to prevent thundering herd problem
                    if jitter:
                        delay = delay * (0.5 + random.random())
                    
                    logger.warning(
                        "Retry attempt %d/%d for operation '%s' after %.2f seconds. Error: %s",
                        attempt, max_attempts, operation_name, delay, str(exc)
                    )
                    
                    # Call the on_retry callback if provided
                    if on_retry:
                        on_retry(operation_name, attempt, exc, delay)
                    
                    time.sleep(delay)
            
            # This code should never be reached
            raise RuntimeError("Unexpected execution flow in retry decorator")
        
        return cast(F, wrapper)
    
    return decorator


def with_retry(
    func: Callable[..., T],
    *args: Any,
    max_attempts: int = 3,
    retry_exceptions: Union[Type[Exception], List[Type[Exception]]] = Exception,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    on_retry: Optional[Callable[[str, int, Exception, float], None]] = None,
    **kwargs: Any,
) -> T:
    """
    Function to retry a callable on exception with exponential backoff.
    
    Args:
        func: The function to retry
        *args: Positional arguments to pass to the function
        max_attempts: Maximum number of retry attempts (default: 3)
        retry_exceptions: Exception or list of exceptions to retry on (default: Exception)
        base_delay: Initial delay between retries in seconds (default: 1.0)
        max_delay: Maximum delay between retries in seconds (default: 60.0)
        backoff_factor: Multiplier for the delay on each retry (default: 2.0)
        jitter: Whether to add randomness to the delay (default: True)
        on_retry: Optional callback function called before each retry with parameters:
                  (operation_name, attempt_number, exception, next_delay)
        **kwargs: Keyword arguments to pass to the function
    
    Returns:
        The return value of the function
    
    Raises:
        RetryExhaustedError: When all retry attempts have been exhausted
    
    Example:
        def send_request(url, data):
            # code that might raise transient errors
            return requests.post(url, json=data)
        
        # Use with_retry for one-off calls
        result = with_retry(
            send_request,
            "https://api.example.com",
            data={"key": "value"},
            max_attempts=5,
            retry_exceptions=[ConnectionError, TimeoutError]
        )
    """
    if not isinstance(retry_exceptions, list):
        retry_exceptions = [retry_exceptions]
    
    operation_name = getattr(func, "__qualname__", str(func))
    
    for attempt in range(1, max_attempts + 1):
        try:
            return func(*args, **kwargs)
        except tuple(retry_exceptions) as exc:
            if attempt == max_attempts:
                logger.error(
                    "All %d retry attempts exhausted for operation '%s'",
                    max_attempts, operation_name
                )
                error_message = f"All retry attempts exhausted for operation '{operation_name}' after {max_attempts} attempts"
                raise RetryExhaustedError(
                    message=error_message,
                    operation=operation_name,
                    attempts=max_attempts,
                    details={"last_error": str(exc)}
                ) from exc
            
            # Calculate backoff delay with exponential backoff
            delay = min(base_delay * (backoff_factor ** (attempt - 1)), max_delay)
            
            # Add jitter to prevent thundering herd problem
            if jitter:
                delay = delay * (0.5 + random.random())
            
            logger.warning(
                "Retry attempt %d/%d for operation '%s' after %.2f seconds. Error: %s",
                attempt, max_attempts, operation_name, delay, str(exc)
            )
            
            # Call the on_retry callback if provided
            if on_retry:
                on_retry(operation_name, attempt, exc, delay)
            
            time.sleep(delay)
    
    # This code should never be reached
    raise RuntimeError("Unexpected execution flow in with_retry function") 