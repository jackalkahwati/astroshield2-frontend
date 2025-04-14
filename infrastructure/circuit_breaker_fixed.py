from functools import wraps
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
import logging
import time

logger = logging.getLogger(__name__)

# Custom circuit breaker since CircuitBreaker is not available in the current tenacity version
class CircuitBreakerState:
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=30):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
    
    def __call__(self, retry_state):
        if self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                logger.info("Circuit half-open, allowing retry attempt")
                self.state = CircuitBreakerState.HALF_OPEN
                return True
            logger.info("Circuit open, blocking retry attempt")
            return False
            
        return True
    
    def reset(self):
        self.failure_count = 0
        self.state = CircuitBreakerState.CLOSED
        
    def record_success(self):
        if self.state == CircuitBreakerState.HALF_OPEN:
            logger.info("Circuit closing after successful retry")
            self.reset()
            
    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitBreakerState.HALF_OPEN or self.failure_count >= self.failure_threshold:
            logger.warning(f"Circuit opening after {self.failure_count} failures")
            self.state = CircuitBreakerState.OPEN

def circuit_breaker(
    failure_threshold=5,
    recovery_timeout=30,
    retry_max_attempts=3,
    retry_min_wait=1,
    retry_max_wait=10
):
    """
    Circuit breaker decorator with retry mechanism
    
    Args:
        failure_threshold (int): Number of failures before opening circuit
        recovery_timeout (int): Time in seconds before attempting recovery
        retry_max_attempts (int): Maximum number of retry attempts
        retry_min_wait (int): Minimum wait time between retries
        retry_max_wait (int): Maximum wait time between retries
    """
    def decorator(func):
        circuit = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout
        )
        
        @retry(
            stop=stop_after_attempt(retry_max_attempts),
            wait=wait_exponential(multiplier=retry_min_wait, max=retry_max_wait),
            retry=circuit,
            before=lambda retry_state: logger.info(f"Attempt {retry_state.attempt_number} for {func.__name__}")
        )
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                circuit.record_success()
                return result
            except Exception as e:
                logger.error(f"Circuit breaker caught error in {func.__name__}: {str(e)}")
                circuit.record_failure()
                raise
                
        return wrapper
    return decorator

# Example usage:
# @circuit_breaker(failure_threshold=3, recovery_timeout=60)
# def api_call():
#     pass 