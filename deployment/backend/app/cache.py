from functools import wraps
from datetime import datetime, timedelta
import redis
from flask import request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Initialize Redis for caching
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Initialize rate limiter
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

def cache_response(expiration=300):  # Cache for 5 minutes by default
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Create a cache key from the function name and arguments
            cache_key = f"{f.__name__}:{request.path}:{str(request.get_json())}"
            
            # Try to get the cached response
            cached_response = redis_client.get(cache_key)
            if cached_response:
                return jsonify(eval(cached_response))
            
            # If no cache, execute the function
            response = f(*args, **kwargs)
            
            # Cache the response
            redis_client.setex(
                cache_key,
                expiration,
                str(response.get_json())
            )
            
            return response
        return decorated_function
    return decorator

def configure_cache_and_limits(app):
    """Configure caching and rate limiting for the application"""
    limiter.init_app(app)
    
    # Configure rate limits for specific endpoints
    limiter.limit("100/hour")(app.view_functions['advanced.analyze_stimulation'])
    limiter.limit("100/hour")(app.view_functions['advanced.analyze_launch_tracking'])
    limiter.limit("100/hour")(app.view_functions['advanced.analyze_eclipse_tracking'])
    limiter.limit("100/hour")(app.view_functions['advanced.analyze_orbit_occupancy'])
    limiter.limit("100/hour")(app.view_functions['advanced.verify_un_registry'])
    limiter.limit("300/hour")(app.view_functions['advanced.batch_analyze'])
    
    return app
