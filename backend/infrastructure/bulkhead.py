import asyncio
import logging
import functools
from typing import Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)

class BulkheadManager:
    """
    Implementation of the Bulkhead pattern to isolate failures
    by controlling the concurrent execution of operations.
    """
    
    def __init__(self, default_max_concurrent: int = 10, default_max_queue: int = 15):
        """
        Initialize the bulkhead manager.
        
        Args:
            default_max_concurrent: Default maximum concurrent executions
            default_max_queue: Default maximum queue size
        """
        self.bulkheads: Dict[str, Dict[str, Any]] = {}
        self.default_max_concurrent = default_max_concurrent
        self.default_max_queue = default_max_queue
    
    def configure(self, name: str, max_concurrent: int, max_queue: int = None):
        """
        Configure a named bulkhead.
        
        Args:
            name: Bulkhead name
            max_concurrent: Maximum concurrent executions
            max_queue: Maximum queue size (default: 1.5x max_concurrent)
        """
        if not max_queue:
            max_queue = int(max_concurrent * 1.5)
        
        self.bulkheads[name] = {
            "semaphore": asyncio.Semaphore(max_concurrent),
            "max_concurrent": max_concurrent,
            "max_queue": max_queue,
            "current_queue": 0,
            "active": 0,
            "rejected": 0,
            "total": 0,
        }
        
        logger.info(f"Configured bulkhead '{name}': max_concurrent={max_concurrent}, max_queue={max_queue}")
    
    def _get_or_create_bulkhead(self, name: str) -> Dict[str, Any]:
        """Get or create a bulkhead with the given name"""
        if name not in self.bulkheads:
            self.configure(name, self.default_max_concurrent, self.default_max_queue)
        return self.bulkheads[name]
    
    def limit(self, bulkhead_name: str):
        """
        Decorator to apply bulkhead pattern to an async function.
        
        Args:
            bulkhead_name: Name of the bulkhead to use
            
        Returns:
            Decorated function
        """
        def decorator(func: Callable):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                bulkhead = self._get_or_create_bulkhead(bulkhead_name)
                
                # Increment queue count and check if we can accept
                bulkhead["current_queue"] += 1
                bulkhead["total"] += 1
                
                if bulkhead["current_queue"] > bulkhead["max_queue"]:
                    # Queue full, reject the request
                    bulkhead["current_queue"] -= 1
                    bulkhead["rejected"] += 1
                    logger.warning(f"Bulkhead '{bulkhead_name}' rejected request - queue full")
                    raise Exception(f"Service overloaded - bulkhead '{bulkhead_name}' queue full")
                
                try:
                    # Acquire semaphore (will wait if max_concurrent reached)
                    async with bulkhead["semaphore"]:
                        # Decrement queue now that we're executing
                        bulkhead["current_queue"] -= 1
                        bulkhead["active"] += 1
                        
                        logger.debug(f"Executing in bulkhead '{bulkhead_name}' (active: {bulkhead['active']})")
                        try:
                            # Execute the function
                            return await func(*args, **kwargs)
                        finally:
                            bulkhead["active"] -= 1
                finally:
                    # Handle case when we're rejected while waiting
                    if bulkhead["current_queue"] > 0:
                        bulkhead["current_queue"] -= 1
            
            return wrapper
        return decorator 