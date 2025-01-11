from functools import wraps
from typing import Dict, Optional
import time
import logging
from redis import Redis
from flask import request, jsonify

logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiter implementation using Redis"""
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        self.redis = Redis(host=redis_host, port=redis_port)
        self.default_limit = 100  # requests per minute
        self.default_window = 60  # seconds
        
        # Custom limits for different endpoints
        self.endpoint_limits: Dict[str, Dict[str, int]] = {
            '/api/bulk_analysis': {'limit': 20, 'window': 60},
            '/api/generate_ccdm_report': {'limit': 30, 'window': 60},
            '/api/correlation_analysis': {'limit': 50, 'window': 60}
        }

    def get_rate_limit(self, endpoint: str) -> tuple[int, int]:
        """Get rate limit and window for an endpoint"""
        if endpoint in self.endpoint_limits:
            return (
                self.endpoint_limits[endpoint]['limit'],
                self.endpoint_limits[endpoint]['window']
            )
        return self.default_limit, self.default_window

    def is_rate_limited(self, key: str, limit: int, window: int) -> bool:
        """Check if request should be rate limited"""
        try:
            pipe = self.redis.pipeline()
            now = time.time()
            
            # Remove old requests
            pipe.zremrangebyscore(key, 0, now - window)
            
            # Add current request
            pipe.zadd(key, {str(now): now})
            
            # Get request count
            pipe.zcard(key)
            
            # Set key expiration
            pipe.expire(key, window)
            
            # Execute pipeline
            _, _, request_count, _ = pipe.execute()
            
            return request_count > limit
        except Exception as e:
            logger.error(f"Rate limiter error: {str(e)}")
            return False  # Fail open on errors

    def rate_limit(self, 
                  limit: Optional[int] = None,
                  window: Optional[int] = None,
                  key_func=None):
        """
        Rate limiting decorator
        
        Args:
            limit: Maximum number of requests per window
            window: Time window in seconds
            key_func: Function to generate rate limit key
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    endpoint = request.path
                    endpoint_limit, endpoint_window = self.get_rate_limit(endpoint)
                    
                    # Use provided values or endpoint defaults
                    final_limit = limit or endpoint_limit
                    final_window = window or endpoint_window
                    
                    # Generate rate limit key
                    if key_func:
                        rate_limit_key = f"rate_limit:{key_func()}"
                    else:
                        # Default to IP-based rate limiting
                        client_ip = request.remote_addr
                        rate_limit_key = f"rate_limit:{endpoint}:{client_ip}"
                    
                    if self.is_rate_limited(rate_limit_key, final_limit, final_window):
                        logger.warning(f"Rate limit exceeded for {rate_limit_key}")
                        return jsonify({
                            'error': 'Rate limit exceeded',
                            'message': f'Maximum {final_limit} requests per {final_window} seconds'
                        }), 429
                    
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Rate limiter error: {str(e)}")
                    return func(*args, **kwargs)  # Fail open
                    
            return wrapper
        return decorator

# Example usage:
# rate_limiter = RateLimiter(redis_host='localhost', redis_port=6379)
#
# @app.route('/api/endpoint')
# @rate_limiter.rate_limit(limit=100, window=60)
# def my_endpoint():
#     return jsonify({'message': 'Success'})
