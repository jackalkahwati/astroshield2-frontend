from functools import wraps
from typing import Callable, Any, Optional
import time
import logging
import asyncio
from collections import defaultdict

logger = logging.getLogger(__name__)

class CircuitBreakerState:
    """State machine for the circuit breaker pattern."""
    CLOSED = 'CLOSED'  # Normal operation, requests go through
    OPEN = 'OPEN'      # Circuit is open, requests fail fast
    HALF_OPEN = 'HALF_OPEN'  # Testing if service is back online
    
class CircuitBreakerRegistry:
    """Registry to track circuit breaker instances by function."""
    _instances = {}
    
    @classmethod
    def register(cls, func_name, instance):
        cls._instances[func_name] = instance
        
    @classmethod
    def get_instance(cls, func_name):
        return cls._instances.get(func_name)
    
    @classmethod
    def get_all_states(cls):
        return {name: instance.state for name, instance in cls._instances.items()}

class CircuitBreaker:
    """
    Circuit breaker implementation to prevent cascading failures.
    """
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
        name: Optional[str] = None
    ):
        """
        Initialize the circuit breaker.
        
        Args:
            failure_threshold: Number of failures before circuit opens
            recovery_timeout: Time in seconds before attempting recovery
            name: Optional name for this circuit breaker
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.name = name or 'unnamed'
        
    def __call__(self, func):
        """Make the class callable as a decorator."""
        # Register this instance
        func_name = f"{func.__module__}.{func.__name__}"
        self.name = func_name
        CircuitBreakerRegistry.register(func_name, self)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
            
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await self.async_call(func, *args, **kwargs)
            
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
        
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute the function with circuit breaker logic."""
        if not self._should_allow_execution():
            error_msg = f"Circuit breaker '{self.name}' is OPEN - failing fast"
            logger.warning(error_msg)
            raise CircuitBreakerOpenException(error_msg)
            
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
            
    async def async_call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute the async function with circuit breaker logic."""
        if not self._should_allow_execution():
            error_msg = f"Circuit breaker '{self.name}' is OPEN - failing fast"
            logger.warning(error_msg)
            raise CircuitBreakerOpenException(error_msg)
            
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_allow_execution(self) -> bool:
        """Determine if the current request should be allowed."""
        if self.state == CircuitBreakerState.CLOSED:
            return True
            
        if self.state == CircuitBreakerState.OPEN:
            # Check if recovery timeout has elapsed
            timeout_elapsed = time.time() - self.last_failure_time > self.recovery_timeout
            if timeout_elapsed:
                logger.info(f"Circuit breaker '{self.name}' transitioning from OPEN to HALF_OPEN")
                self.state = CircuitBreakerState.HALF_OPEN
                return True
            return False
            
        # In HALF_OPEN state, we allow a single test request
        return True
        
    def _on_success(self):
        """Handle successful execution."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            logger.info(f"Circuit breaker '{self.name}' recovered - transitioning to CLOSED")
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            
    def _on_failure(self):
        """Handle failed execution."""
        self.last_failure_time = time.time()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            logger.warning(f"Circuit breaker '{self.name}' test request failed - remaining OPEN")
            self.state = CircuitBreakerState.OPEN
            return
            
        self.failure_count += 1
        if self.state == CircuitBreakerState.CLOSED and self.failure_count >= self.failure_threshold:
            logger.warning(f"Circuit breaker '{self.name}' tripped after {self.failure_count} failures - transitioning to OPEN")
            self.state = CircuitBreakerState.OPEN
            
class CircuitBreakerOpenException(Exception):
    """Exception raised when a circuit breaker is open."""
    pass

# Simple decorator for backward compatibility
def circuit_breaker(func=None, failure_threshold=5, recovery_timeout=30):
    """
    Circuit breaker decorator that can be used with or without parameters.
    
    Example usage:
        @circuit_breaker
        def my_func():
            pass
            
        @circuit_breaker(failure_threshold=3, recovery_timeout=60)
        def my_other_func():
            pass
    """
    if func is None:
        # Called with parameters
        return CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout
        )
        
    # Called without parameters
    return CircuitBreaker()(func) 