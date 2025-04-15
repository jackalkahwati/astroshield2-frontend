from typing import Dict, Optional, Callable
import time
import logging
from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import redis
import os
from starlette.status import HTTP_429_TOO_MANY_REQUESTS

logger = logging.getLogger(__name__)

# Load from environment or use defaults
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_ENABLED = os.getenv("REDIS_ENABLED", "true").lower() == "true"
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"

# Default rate limits for different endpoints (requests per minute)
DEFAULT_RATE_LIMITS = {
    "default": 100,
    "/api/historical": 30,  # Resource-intensive endpoint
    "/api/get_historical_analysis": 30,
    "/api/analyze_conjunction": 60,
    "/api/get_assessment": 120,
    "/api/batch_analyze": 10
}

class RateLimiter:
    """Rate limiter implementation using Redis"""
    
    def __init__(self, redis_host: str = REDIS_HOST, redis_port: int = REDIS_PORT):
        """Initialize rate limiter with Redis connection if enabled"""
        self.enabled = RATE_LIMIT_ENABLED
        
        if not self.enabled:
            logger.info("Rate limiting is disabled")
            return
            
        try:
            self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
            self.redis.ping()  # Test connection
            logger.info(f"Rate limiter connected to Redis at {redis_host}:{redis_port}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            self.enabled = False
            
        self.rate_limits = DEFAULT_RATE_LIMITS.copy()
        
    def get_rate_limit(self, endpoint: str) -> int:
        """Get rate limit for a specific endpoint"""
        # Find the most specific endpoint match
        for path, limit in self.rate_limits.items():
            if path in endpoint:
                return limit
                
        return self.rate_limits["default"]
    
    def is_rate_limited(self, key: str, limit: int, window: int = 60) -> tuple[bool, int, int]:
        """
        Check if request should be rate limited
        
        Args:
            key: Rate limit key (typically user_id:endpoint)
            limit: Maximum number of requests allowed
            window: Time window in seconds
            
        Returns:
            Tuple of (is_limited, current_count, reset_time)
        """
        if not self.enabled:
            return False, 0, 0
            
        try:
            pipe = self.redis.pipeline()
            now = time.time()
            reset_time = int(now) + window
            
            # Key for storing the request counts
            window_key = f"{key}:{int(now / window)}"
            
            # Get current count
            current_count = self.redis.get(window_key)
            current_count = int(current_count) if current_count else 0
            
            # Check if rate limit exceeded
            if current_count >= limit:
                return True, current_count, reset_time
                
            # Increment counter and set expiry
            pipe.incr(window_key)
            pipe.expire(window_key, window)
            pipe.execute()
            
            return False, current_count + 1, reset_time
        except Exception as e:
            logger.error(f"Rate limiter error: {str(e)}")
            return False, 0, 0  # Fail open if Redis is down
    
    def update_rate_limit(self, endpoint: str, new_limit: int) -> None:
        """Update rate limit for an endpoint"""
        self.rate_limits[endpoint] = new_limit
        logger.info(f"Updated rate limit for {endpoint} to {new_limit} requests per minute")

class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting"""
    
    def __init__(self, app: FastAPI, rate_limiter: Optional[RateLimiter] = None):
        super().__init__(app)
        self.rate_limiter = rate_limiter or RateLimiter()
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Rate limit based on client IP or user ID"""
        # Skip rate limiting for certain paths
        if self._should_skip(request.url.path):
            return await call_next(request)
        
        # Get client IP
        client_ip = self._get_client_ip(request)
        
        # Get rate limit key based on user ID (if authenticated) or IP
        key = None
        
        # Try to get user safely without raising an assertion error
        user = None
        try:
            if "user" in request.scope and request.scope["user"] is not None:
                user = request.scope["user"]
        except (AssertionError, AttributeError):
            # No user in request scope, use IP only
            pass
        
        # Use user ID if available, otherwise fall back to IP
        if user is not None and hasattr(user, "id"):
            # Use both user ID and IP to prevent shared account abuse
            key = f"rate_limit:{user.id}:{client_ip}"
            
            # Allow higher limits for authenticated users
            max_requests = self.auth_rate_limit
        else:
            # Use IP only for unauthenticated requests
            key = f"rate_limit:{client_ip}"
            max_requests = self.anon_rate_limit
        
        # Get rate limit for endpoint
        limit = self.rate_limiter.get_rate_limit(request.url.path)
        
        # Check if rate limited
        is_limited, current, reset = self.rate_limiter.is_rate_limited(key, limit)
        
        # Add rate limit headers
        headers = {
            "X-RateLimit-Limit": str(limit),
            "X-RateLimit-Remaining": str(max(0, limit - current)),
            "X-RateLimit-Reset": str(reset)
        }
        
        if is_limited:
            logger.warning(f"Rate limit exceeded for {client_ip}")
            
            # Return 429 Too Many Requests
            content = {
                "detail": "Rate limit exceeded. Please try again later.",
                "status_code": HTTP_429_TOO_MANY_REQUESTS
            }
            
            response = Response(
                content=str(content),
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                headers=headers
            )
            return response
            
        # Process the request
        response = await call_next(request)
        
        # Add rate limit headers to response
        for key, value in headers.items():
            response.headers[key] = value
            
        return response

def add_rate_limit_middleware(app: FastAPI) -> None:
    """Add rate limiting middleware to FastAPI application"""
    rate_limiter = RateLimiter()
    app.add_middleware(RateLimitMiddleware, rate_limiter=rate_limiter)
    logger.info("Rate limiting middleware added to application") 