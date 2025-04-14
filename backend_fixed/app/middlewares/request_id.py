import uuid
from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable
import logging
import contextvars

# Create contextvar to store request ID
request_id_var = contextvars.ContextVar("request_id", default=None)

logger = logging.getLogger(__name__)

class RequestIdMiddleware(BaseHTTPMiddleware):
    """Middleware to add request ID to each request for tracking"""
    
    def __init__(self, app: FastAPI, header_name: str = "X-Request-ID"):
        super().__init__(app)
        self.header_name = header_name
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add request ID to request and response headers"""
        # Get request ID from header or generate a new one
        request_id = request.headers.get(self.header_name)
        if not request_id:
            request_id = str(uuid.uuid4())
            
        # Store in context var for logging
        token = request_id_var.set(request_id)
        
        # Add to request state for use in endpoint functions
        request.state.request_id = request_id
        
        # Process the request
        response = await call_next(request)
        
        # Add request ID to response
        response.headers[self.header_name] = request_id
        
        # Reset context var
        request_id_var.reset(token)
        
        return response

def add_request_id_middleware(app: FastAPI):
    """Add request ID middleware to FastAPI application"""
    app.add_middleware(RequestIdMiddleware)
    logger.info("Request ID middleware added to application")
    
def get_request_id() -> str:
    """Get the current request ID from context"""
    request_id = request_id_var.get()
    if request_id is None:
        # Not in a request context
        request_id = "no-request-id"
    return request_id 