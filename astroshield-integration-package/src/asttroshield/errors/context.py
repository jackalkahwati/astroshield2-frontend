"""
Context-based error handling and tracing utilities for AstroShield.

This module provides context managers and decorators for standardized
error handling across different execution contexts.
"""

import contextlib
import logging
import sys
import traceback
from typing import Callable, Dict, List, Optional, Type, Union

from .exceptions import AstroShieldError


logger = logging.getLogger(__name__)


@contextlib.contextmanager
def error_context(
    context_name: str,
    error_map: Dict[Type[Exception], Type[AstroShieldError]] = None,
    reraise: bool = True,
    log_level: int = logging.ERROR,
    callback: Callable[[Exception, Dict], None] = None,
    additional_context: Dict = None
):
    """
    Context manager for standardized error handling.
    
    Args:
        context_name: Name of the operation context for error messages
        error_map: Mapping from source exceptions to AstroShield exceptions
        reraise: Whether to reraise exceptions after handling
        log_level: Logging level for errors
        callback: Optional callback function to execute when an error occurs
        additional_context: Additional context data to include in error details
    
    Yields:
        None
    
    Example:
        ```python
        with error_context(
            "database_operation", 
            error_map={SQLError: ConnectionError}
        ):
            db.execute_query()
        ```
    """
    error_map = error_map or {}
    additional_context = additional_context or {}
    
    try:
        yield
    except Exception as e:
        exc_type, exc_value, exc_tb = sys.exc_info()
        stack_trace = traceback.format_exception(exc_type, exc_value, exc_tb)
        
        details = {
            "context": context_name,
            "original_error": str(e),
            "original_error_type": e.__class__.__name__,
            "stack_trace": stack_trace,
            **additional_context
        }
        
        # Map the exception to an AstroShield-specific exception if defined
        if type(e) in error_map:
            mapped_error = error_map[type(e)](
                message=f"Error in {context_name}: {str(e)}",
                details=details
            )
            logger.log(log_level, f"Mapped exception in {context_name}", exc_info=True, extra=details)
            if callback:
                try:
                    callback(e, details)
                except Exception as callback_error:
                    logger.error(
                        f"Error in callback for {context_name}: {callback_error}",
                        exc_info=True
                    )
            if reraise:
                raise mapped_error from e
            return
        
        # If it's already an AstroShield error, update its details
        if isinstance(e, AstroShieldError):
            e.details.update(details)
            logger.log(log_level, f"Error in {context_name}: {e}", exc_info=True, extra=details)
            if callback:
                try:
                    callback(e, details)
                except Exception as callback_error:
                    logger.error(
                        f"Error in callback for {context_name}: {callback_error}",
                        exc_info=True
                    )
            if reraise:
                raise
            return
            
        # If it's not mapped and not an AstroShield error, log it
        logger.log(log_level, f"Unmapped exception in {context_name}: {e}", exc_info=True, extra=details)
        if callback:
            try:
                callback(e, details)
            except Exception as callback_error:
                logger.error(
                    f"Error in callback for {context_name}: {callback_error}",
                    exc_info=True
                )
        if reraise:
            raise


def with_error_context(
    context_name: str = None,
    error_map: Dict[Type[Exception], Type[AstroShieldError]] = None,
    reraise: bool = True,
    log_level: int = logging.ERROR,
    callback: Callable[[Exception, Dict], None] = None
):
    """
    Decorator for applying error context to functions.
    
    Args:
        context_name: Name of the operation context (defaults to function name)
        error_map: Mapping from source exceptions to AstroShield exceptions
        reraise: Whether to reraise exceptions after handling
        log_level: Logging level for errors
        callback: Optional callback function to execute when an error occurs
    
    Returns:
        Decorated function
    
    Example:
        ```python
        @with_error_context(error_map={SQLError: ConnectionError})
        def fetch_data(query):
            return db.execute_query(query)
        ```
    """
    def decorator(func):
        nonlocal context_name
        if context_name is None:
            context_name = func.__name__
            
        def wrapper(*args, **kwargs):
            additional_context = {
                "function": func.__name__,
                "args": str(args),
                "kwargs": str(kwargs)
            }
            
            with error_context(
                context_name=context_name,
                error_map=error_map,
                reraise=reraise,
                log_level=log_level,
                callback=callback,
                additional_context=additional_context
            ):
                return func(*args, **kwargs)
                
        return wrapper
    
    return decorator 