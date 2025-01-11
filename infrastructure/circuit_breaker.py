from functools import wraps
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    CircuitBreaker,
    RetryError
)
import logging

logger = logging.getLogger(__name__)

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
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Circuit breaker caught error in {func.__name__}: {str(e)}")
                raise
                
        return wrapper
    return decorator

# Example usage:
# @circuit_breaker(failure_threshold=3, recovery_timeout=60)
# def api_call():
#     pass
