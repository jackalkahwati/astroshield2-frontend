"""
Custom exceptions for the AstroShield project.

This module defines application-specific exceptions that can be used
throughout the codebase for more granular error handling.
"""

from typing import Any, Dict, Optional


class AstroShieldError(Exception):
    """Base exception class for all AstroShield errors."""
    
    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the AstroShieldError.
        
        Args:
            message: Human-readable error message
            code: Optional error code for categorization
            details: Optional dictionary with additional error details
        """
        self.message = message or "An unknown error occurred"
        self.code = code
        self.details = details or {}
        super().__init__(self.message)
        
    def __str__(self):
        if self.details:
            return f"{self.message} - Details: {self.details}"
        return self.message


class ValidationError(AstroShieldError):
    """Raised when data validation fails."""
    
    def __init__(
        self,
        message: str,
        validation_errors: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize ValidationError.
        
        Args:
            message: Human-readable error message
            validation_errors: Dictionary containing validation error details
            **kwargs: Additional keyword arguments for AstroShieldError
        """
        details = kwargs.pop("details", {})
        if validation_errors:
            details["validation_errors"] = validation_errors
        super().__init__(message, code="VALIDATION_ERROR", details=details, **kwargs)


class SchemaError(ValidationError):
    """Raised when a schema validation error occurs."""
    
    def __init__(self, schema_name: str = None, details: dict = None):
        message = f"Schema validation failed"
        if schema_name:
            message = f"Schema validation failed for '{schema_name}'"
        super().__init__(message, details)


class ConnectionError(AstroShieldError):
    """Raised when a connection to an external service fails."""
    
    def __init__(self, service: str = None, details: dict = None):
        message = "Connection failed"
        if service:
            message = f"Connection to {service} failed"
        super().__init__(message, details)


class ConfigurationError(AstroShieldError):
    """Raised when there is an issue with configuration."""
    
    def __init__(self, config_name: str = None, details: dict = None):
        message = "Configuration error"
        if config_name:
            message = f"Configuration error with '{config_name}'"
        super().__init__(message, details)
        

class AuthenticationError(AstroShieldError):
    """Raised when authentication fails."""
    
    def __init__(self, service: str = None, details: dict = None):
        message = "Authentication failed"
        if service:
            message = f"Authentication failed for {service}"
        super().__init__(message, details)


class MessageProcessingError(AstroShieldError):
    """Raised when message processing fails."""
    
    def __init__(self, message_type: str = None, details: dict = None):
        message = "Message processing failed"
        if message_type:
            message = f"Processing failed for message type '{message_type}'"
        super().__init__(message, details)


class RetryExhaustedError(AstroShieldError):
    """Raised when all retry attempts have been exhausted."""
    
    def __init__(
        self,
        message: str,
        operation: str,
        attempts: int,
        **kwargs
    ):
        """
        Initialize RetryExhaustedError.
        
        Args:
            message: Human-readable error message
            operation: Name of the operation that failed
            attempts: Number of retry attempts that were made
            **kwargs: Additional keyword arguments for AstroShieldError
        """
        details = kwargs.pop("details", {})
        details.update({
            "operation": operation,
            "attempts": attempts
        })
        super().__init__(message, code="RETRY_EXHAUSTED", details=details, **kwargs) 