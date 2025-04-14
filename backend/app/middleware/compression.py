from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.datastructures import MutableHeaders
import gzip
import logging
import os

logger = logging.getLogger(__name__)

class CompressionMiddleware(BaseHTTPMiddleware):
    """
    Middleware for compressing API responses.
    
    This middleware automatically compresses responses if they exceed
    a certain size threshold and the client supports compression.
    
    Benefits:
    - Reduces bandwidth usage
    - Improves response time for clients
    - Especially useful for large datasets like historical analysis
    """
    
    def __init__(
        self,
        app: FastAPI,
        minimum_size: int = 1000,  # Minimum size in bytes to compress
        compression_level: int = 6,  # Gzip compression level (1-9)
        exclude_paths: list = None,  # Paths to exclude from compression
    ):
        super().__init__(app)
        self.minimum_size = minimum_size
        self.compression_level = compression_level
        self.exclude_paths = exclude_paths or []
        
        # Get environment
        self.environment = os.getenv("DEPLOYMENT_ENV", "development")
        
        logger.info(f"CompressionMiddleware initialized (min size: {minimum_size} bytes, level: {compression_level})")
    
    async def dispatch(self, request: Request, call_next):
        """
        Process the request and compress the response if needed.
        
        Args:
            request: The incoming request
            call_next: The next middleware in the chain
            
        Returns:
            The response, potentially compressed
        """
        # Check if client accepts gzip encoding
        accept_encoding = request.headers.get("Accept-Encoding", "")
        accepts_gzip = "gzip" in accept_encoding.lower()
        
        # Check if path should be excluded
        path = request.url.path
        should_exclude = any(path.startswith(excluded) for excluded in self.exclude_paths)
        
        if not accepts_gzip or should_exclude:
            # Client doesn't accept gzip or path is excluded
            return await call_next(request)
        
        # Get response from the next middleware/endpoint
        response = await call_next(request)
        
        # Get response body
        body = b""
        async for chunk in response.body_iterator:
            body += chunk
            
        # Check if response should be compressed
        if len(body) < self.minimum_size:
            # Response is small, don't compress
            return Response(
                content=body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type
            )
        
        # Compress the response body
        compressed_body = gzip.compress(body, compresslevel=self.compression_level)
        
        # Create a new response with the compressed body
        new_response = Response(
            content=compressed_body,
            status_code=response.status_code,
            media_type=response.media_type
        )
        
        # Copy headers from original response
        new_headers = MutableHeaders(new_response.headers)
        for name, value in response.headers.items():
            if name.lower() not in ('content-length', 'content-encoding'):
                new_headers[name] = value
        
        # Add compression headers
        new_headers["Content-Encoding"] = "gzip"
        new_headers["Content-Length"] = str(len(compressed_body))
        new_headers["Vary"] = "Accept-Encoding"
        
        # Add compression info to headers in development environment
        if self.environment != "production":
            original_size = len(body)
            compressed_size = len(compressed_body)
            reduction = (original_size - compressed_size) / original_size * 100
            new_headers["X-Compression-Ratio"] = f"{reduction:.1f}%"
            new_headers["X-Original-Size"] = str(original_size)
            new_headers["X-Compressed-Size"] = str(compressed_size)
        
        return new_response

def add_compression_middleware(
    app: FastAPI,
    minimum_size: int = 5000,
    compression_level: int = 6,
    exclude_paths: list = None
):
    """
    Add compression middleware to the app.
    
    Args:
        app: The FastAPI application
        minimum_size: Minimum size in bytes to compress (default: 5KB)
        compression_level: Gzip compression level (1-9, default: 6)
        exclude_paths: Paths to exclude from compression
    """
    exclude_paths = exclude_paths or [
        "/health",      # Health checks
        "/metrics",     # Metrics endpoints
        "/static/",     # Static files (already compressed)
    ]
    
    app.add_middleware(
        CompressionMiddleware,
        minimum_size=minimum_size,
        compression_level=compression_level,
        exclude_paths=exclude_paths
    )
    
    logger.info("Compression middleware added to application") 