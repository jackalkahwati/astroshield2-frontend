from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ErrorCode:
    """Error code constants for standardized error responses"""
    # Client errors (4xx)
    INVALID_INPUT = "INVALID_INPUT"  # Invalid input parameters
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"  # Requested resource not found
    UNAUTHORIZED = "UNAUTHORIZED"  # User not authenticated
    FORBIDDEN = "FORBIDDEN"  # User not authorized
    RATE_LIMITED = "RATE_LIMITED"  # Too many requests
    VALIDATION_ERROR = "VALIDATION_ERROR"  # Request validation failed
    
    # Server errors (5xx)
    SERVER_ERROR = "SERVER_ERROR"  # Generic server error
    DATABASE_ERROR = "DATABASE_ERROR"  # Database operation failed
    EXTERNAL_SERVICE_ERROR = "EXTERNAL_SERVICE_ERROR"  # External API error
    TIMEOUT_ERROR = "TIMEOUT_ERROR"  # Operation timed out

class StandardError(Exception):
    """Base class for standardized API errors"""
    
    def __init__(
        self, 
        code: str, 
        message: str, 
        status_code: int = 500, 
        details: Optional[Dict[str, Any]] = None
    ):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to standardized dictionary format"""
        error_dict = {
            "error": self.code,
            "message": self.message,
            "status_code": self.status_code
        }
        
        # Add details if available
        if self.details:
            error_dict["details"] = self.details
            
        return error_dict

# Specific error classes
class InvalidInputError(StandardError):
    """Error raised when input validation fails"""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = value
            
        super().__init__(
            code=ErrorCode.INVALID_INPUT,
            message=message,
            status_code=status.HTTP_400_BAD_REQUEST,
            details=details
        )

class ResourceNotFoundError(StandardError):
    """Error raised when a requested resource is not found"""
    
    def __init__(self, resource_type: str, resource_id: str):
        super().__init__(
            code=ErrorCode.RESOURCE_NOT_FOUND,
            message=f"{resource_type} with ID '{resource_id}' not found",
            status_code=status.HTTP_404_NOT_FOUND,
            details={"resource_type": resource_type, "resource_id": resource_id}
        )

class DatabaseError(StandardError):
    """Error raised when a database operation fails"""
    
    def __init__(self, message: str, operation: Optional[str] = None):
        details = {}
        if operation:
            details["operation"] = operation
            
        super().__init__(
            code=ErrorCode.DATABASE_ERROR,
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details
        )

class ExternalServiceError(StandardError):
    """Error raised when an external service call fails"""
    
    def __init__(self, service_name: str, message: str):
        super().__init__(
            code=ErrorCode.EXTERNAL_SERVICE_ERROR,
            message=message,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details={"service": service_name}
        )

class RateLimitError(StandardError):
    """Error raised when rate limit is exceeded"""
    
    def __init__(self, message: str = "Rate limit exceeded", retry_after: Optional[int] = None):
        details = {}
        if retry_after:
            details["retry_after"] = retry_after
            
        super().__init__(
            code=ErrorCode.RATE_LIMITED,
            message=message,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            details=details
        )

def register_exception_handlers(app: FastAPI):
    """Register exception handlers for standardized error responses"""
    
    @app.exception_handler(StandardError)
    async def standard_error_handler(request: Request, exc: StandardError):
        """Handle StandardError exceptions"""
        logger.error(f"StandardError: {exc.code} - {exc.message}")
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict()
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        """Handle FastAPI RequestValidationError exceptions"""
        errors = exc.errors()
        error_details = []
        
        for error in errors:
            error_details.append({
                "loc": error.get("loc", []),
                "msg": error.get("msg", ""),
                "type": error.get("type", "")
            })
            
        logger.warning(f"Validation error: {error_details}")
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": ErrorCode.VALIDATION_ERROR,
                "message": "Request validation failed",
                "status_code": status.HTTP_422_UNPROCESSABLE_ENTITY,
                "details": error_details
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle all other exceptions with standardized format"""
        logger.exception(f"Unhandled exception: {str(exc)}")
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": ErrorCode.SERVER_ERROR,
                "message": "An unexpected error occurred",
                "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "details": {"type": str(type(exc).__name__)}
            }
        ) 