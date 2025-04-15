"""
Error handling utilities for AstroShield.

This package provides robust error handling, retry logic, and 
other utilities for graceful handling of failure conditions.
"""

from .handling import RetryHandler, ErrorHandler

__all__ = ['RetryHandler', 'ErrorHandler'] 