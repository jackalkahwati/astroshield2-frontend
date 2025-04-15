"""Custom exceptions for the backend application."""

class ServiceError(Exception):
    """Base class for service layer errors."""
    def __init__(self, detail: str = "Service error occurred"):
        self.detail = detail
        super().__init__(self.detail)

class ObjectNotFoundError(ServiceError):
    """Raised when an object is not found."""
    def __init__(self, object_id: str):
        super().__init__(detail=f"Object '{object_id}' not found")
        self.object_id = object_id

class AnalysisError(ServiceError):
    """Raised when an analysis task fails."""
    def __init__(self, detail: str = "Analysis failed"):
        super().__init__(detail=detail)

class InvalidInputError(ServiceError):
    """Raised when input data for a service is invalid."""
    def __init__(self, detail: str = "Invalid input data"):
        super().__init__(detail=detail) 