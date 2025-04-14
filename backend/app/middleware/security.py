from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
import os
import logging

logger = logging.getLogger(__name__)

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add security headers to responses.
    
    These headers help improve the security posture of the application
    by protecting against common web vulnerabilities.
    """
    
    def __init__(
        self,
        app: FastAPI,
        content_security_policy: str = None,
        enable_hsts: bool = True,
        enable_xframe_options: bool = True,
        enable_content_type_options: bool = True,
        enable_xss_protection: bool = True,
        enable_referrer_policy: bool = True
    ):
        super().__init__(app)
        
        # Get environment
        self.environment = os.getenv("DEPLOYMENT_ENV", "development")
        self.is_production = self.environment == "production"
        
        # Default CSP
        if content_security_policy is None:
            content_security_policy = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data:; "
                "font-src 'self'; "
                "connect-src 'self'; "
                "frame-ancestors 'none';"
            )
        
        self.content_security_policy = content_security_policy
        self.enable_hsts = enable_hsts
        self.enable_xframe_options = enable_xframe_options
        self.enable_content_type_options = enable_content_type_options
        self.enable_xss_protection = enable_xss_protection
        self.enable_referrer_policy = enable_referrer_policy
        
        logger.info(f"SecurityHeadersMiddleware initialized (env: {self.environment})")
    
    async def dispatch(self, request: Request, call_next):
        """
        Add security headers to the response.
        
        Args:
            request: The incoming request
            call_next: The next middleware in the chain
        
        Returns:
            The response with added security headers
        """
        # Process the request and get the response
        response = await call_next(request)
        
        # Skip adding headers for non-HTML, non-JSON responses like images, etc.
        content_type = response.headers.get("content-type", "")
        if not any(ct in content_type.lower() for ct in ["html", "json", "text", "xml"]):
            return response
        
        # Add security headers
        if self.enable_content_type_options:
            # Prevents MIME type sniffing
            response.headers["X-Content-Type-Options"] = "nosniff"
        
        if self.enable_xframe_options:
            # Prevents clickjacking
            response.headers["X-Frame-Options"] = "DENY"
        
        if self.enable_xss_protection:
            # Enables XSS filtering
            response.headers["X-XSS-Protection"] = "1; mode=block"
        
        if self.enable_referrer_policy:
            # Controls what information is included with navigation events
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Add Content-Security-Policy header
        response.headers["Content-Security-Policy"] = self.content_security_policy
        
        # Only add HSTS in production
        if self.enable_hsts and self.is_production:
            # HTTP Strict Transport Security (only in production)
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
        
        # Add Permissions-Policy to limit features
        response.headers["Permissions-Policy"] = (
            "accelerometer=(), camera=(), geolocation=(), gyroscope=(), "
            "magnetometer=(), microphone=(), payment=(), usb=()"
        )
        
        return response

def add_security_middleware(app: FastAPI):
    """
    Add security middleware to the app.
    
    Args:
        app: The FastAPI application
    """
    # Add security headers middleware
    app.add_middleware(SecurityHeadersMiddleware)
    
    logger.info("Security middleware added to application") 