from functools import wraps
from typing import Optional, Any, Callable
import json
import hashlib
import logging
from redis import Redis
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class Cache:
    """Caching implementation using Redis"""
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        self.redis = Redis(host=redis_host, port=redis_port)
        
        # Default cache times for different types of data
        self.cache_times = {
            'historical_analysis': timedelta(hours=24),
            'correlation_analysis': timedelta(hours=12),
            'ccdm_report': timedelta(hours=6),
            'object_analysis': timedelta(hours=1),
            'default': timedelta(minutes=30)
        }

    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate a cache key from function arguments"""
        # Convert args and kwargs to a string representation
        key_parts = [str(arg) for arg in args]
        key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
        
        # Create a hash of the arguments
        key_string = "|".join(key_parts)
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        
        return f"{prefix}:{key_hash}"

    def _serialize(self, data: Any) -> str:
        """Serialize data for caching"""
        try:
            if isinstance(data, (dict, list)):
                return json.dumps(data)
            return str(data)
        except Exception as e:
            logger.error(f"Cache serialization error: {str(e)}")
            raise

    def _deserialize(self, data: str) -> Any:
        """Deserialize cached data"""
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return data
        except Exception as e:
            logger.error(f"Cache deserialization error: {str(e)}")
            raise

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            data = self.redis.get(key)
            if data:
                return self._deserialize(data)
            return None
        except Exception as e:
            logger.error(f"Cache get error: {str(e)}")
            return None

    def set(self, key: str, value: Any, expire: Optional[timedelta] = None):
        """Set value in cache"""
        try:
            serialized_value = self._serialize(value)
            if expire:
                self.redis.setex(key, expire, serialized_value)
            else:
                self.redis.set(key, serialized_value)
        except Exception as e:
            logger.error(f"Cache set error: {str(e)}")

    def delete(self, key: str):
        """Delete value from cache"""
        try:
            self.redis.delete(key)
        except Exception as e:
            logger.error(f"Cache delete error: {str(e)}")

    def cached(self, 
              prefix: str,
              expire: Optional[timedelta] = None,
              key_func: Optional[Callable] = None,
              unless: Optional[Callable] = None):
        """
        Caching decorator
        
        Args:
            prefix: Prefix for cache key
            expire: Cache expiration time
            key_func: Custom function to generate cache key
            unless: Function that returns True if result should not be cached
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    # Generate cache key
                    if key_func:
                        cache_key = key_func(*args, **kwargs)
                    else:
                        cache_key = self._generate_key(prefix, *args, **kwargs)
                    
                    # Try to get from cache
                    cached_value = self.get(cache_key)
                    if cached_value is not None:
                        return cached_value
                    
                    # Execute function
                    result = func(*args, **kwargs)
                    
                    # Check if result should be cached
                    if unless and unless(result):
                        return result
                    
                    # Determine expiration time
                    expiration = expire or self.cache_times.get(
                        prefix,
                        self.cache_times['default']
                    )
                    
                    # Cache the result
                    self.set(cache_key, result, expiration)
                    
                    return result
                except Exception as e:
                    logger.error(f"Cache decorator error: {str(e)}")
                    return func(*args, **kwargs)  # Fail open
                    
            return wrapper
        return decorator

# Example usage:
# cache = Cache(redis_host='localhost', redis_port=6379)
#
# @app.route('/api/historical-data')
# @cache.cached('historical', expire=timedelta(hours=24))
# def get_historical_data():
#     # Expensive operation here
#     return {'data': 'expensive_result'}
