"""
Centralized error handling for the AstroShield API.
This module provides standardized error responses and exception handlers.
"""
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from typing import Optional, Any, Dict, List
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Error response model
class ErrorResponse(BaseModel):
    status_code: int
    message: str
    details: Optional[Any] = None
    error_code: Optional[str] = None
    
# Common errors with standard codes
class ErrorCode:
    VALIDATION_ERROR = "VALIDATION_ERROR"
    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"
    AUTHORIZATION_ERROR = "AUTHORIZATION_ERROR"
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    INTEGRATION_ERROR = "INTEGRATION_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"
    UDL_SERVICE_ERROR = "UDL_SERVICE_ERROR"

# FastAPI exception handlers
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors from request data"""
    error_details = []
    for error in exc.errors():
        error_details.append({
            "loc": error.get("loc", []),
            "msg": error.get("msg", ""),
            "type": error.get("type", "")
        })

    logger.warning(f"Validation error: {error_details}")
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            message="Validation error in request data",
            details=error_details,
            error_code=ErrorCode.VALIDATION_ERROR
        ).dict()
    )

async def udl_service_exception_handler(request: Request, exc: Exception):
    """Handle exceptions from UDL service integration"""
    error_msg = str(exc)
    logger.error(f"UDL service error: {error_msg}")
    
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=ErrorResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            message="Error communicating with UDL service",
            details=error_msg,
            error_code=ErrorCode.UDL_SERVICE_ERROR
        ).dict()
    )

async def not_found_exception_handler(request: Request, exc: Exception):
    """Handle resource not found errors"""
    error_msg = str(exc)
    logger.warning(f"Resource not found: {error_msg}")
    
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content=ErrorResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            message="Resource not found",
            details=error_msg,
            error_code=ErrorCode.RESOURCE_NOT_FOUND
        ).dict()
    )

async def generic_exception_handler(request: Request, exc: Exception):
    """Handle any uncaught exceptions"""
    error_msg = str(exc)
    logger.error(f"Unhandled exception: {error_msg}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message="Internal server error",
            details=error_msg if logger.level <= logging.DEBUG else None,
            error_code=ErrorCode.INTERNAL_SERVER_ERROR
        ).dict()
    )

# Custom exceptions
class UDLServiceException(Exception):
    """Exception raised when there's an error with the UDL service"""
    pass

class ResourceNotFoundException(Exception):
    """Exception raised when a requested resource cannot be found"""
    pass

# Register these handlers with FastAPI app
def register_exception_handlers(app):
    """Register all exception handlers with the FastAPI app"""
    from fastapi.exceptions import RequestValidationError
    
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(UDLServiceException, udl_service_exception_handler)
    app.add_exception_handler(ResourceNotFoundException, not_found_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)
    
    # Log that handlers are registered
    logger.info("Exception handlers registered successfully") 