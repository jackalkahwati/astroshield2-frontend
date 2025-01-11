from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Callable
import threading
import logging
import time

logger = logging.getLogger(__name__)

class BulkheadExecutor:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(BulkheadExecutor, cls).__new__(cls)
                    cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if not self.initialized:
            self.executors: Dict[str, ThreadPoolExecutor] = {}
            self.semaphores: Dict[str, threading.Semaphore] = {}
            self.metrics: Dict[str, Dict[str, int]] = {}
            self.initialized = True

    def create_bulkhead(
        self,
        name: str,
        max_concurrent: int = 10,
        max_queue_size: int = 20
    ):
        """Create a new bulkhead"""
        if name not in self.executors:
            self.executors[name] = ThreadPoolExecutor(max_workers=max_concurrent)
            self.semaphores[name] = threading.Semaphore(max_queue_size)
            self.metrics[name] = {
                'active_count': 0,
                'queue_size': 0,
                'rejection_count': 0
            }
            logger.info(f"Created bulkhead: {name}")

    def execute(
        self,
        bulkhead_name: str,
        func: Callable[..., Any],
        *args,
        **kwargs
    ) -> Any:
        """Execute a function within a bulkhead"""
        if bulkhead_name not in self.executors:
            raise ValueError(f"Bulkhead {bulkhead_name} not found")

        executor = self.executors[bulkhead_name]
        semaphore = self.semaphores[bulkhead_name]
        metrics = self.metrics[bulkhead_name]

        # Try to acquire a slot in the queue
        if not semaphore.acquire(blocking=False):
            metrics['rejection_count'] += 1
            logger.warning(f"Bulkhead {bulkhead_name} rejected execution: queue full")
            raise RuntimeError("Bulkhead queue is full")

        try:
            metrics['queue_size'] += 1
            future = executor.submit(self._wrapped_execution, bulkhead_name, func, *args, **kwargs)
            return future.result()
        finally:
            metrics['queue_size'] -= 1
            semaphore.release()

    def _wrapped_execution(
        self,
        bulkhead_name: str,
        func: Callable[..., Any],
        *args,
        **kwargs
    ) -> Any:
        """Wrapper for execution with metrics"""
        metrics = self.metrics[bulkhead_name]
        start_time = time.time()
        metrics['active_count'] += 1

        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in bulkhead {bulkhead_name}: {str(e)}")
            raise
        finally:
            metrics['active_count'] -= 1
            execution_time = time.time() - start_time
            logger.info(f"Bulkhead {bulkhead_name} execution completed in {execution_time:.2f}s")

    def get_metrics(self, bulkhead_name: str) -> Dict[str, int]:
        """Get metrics for a bulkhead"""
        if bulkhead_name not in self.metrics:
            raise ValueError(f"Bulkhead {bulkhead_name} not found")
        return self.metrics[bulkhead_name].copy()

    def shutdown(self, bulkhead_name: str = None):
        """Shutdown bulkhead(s)"""
        if bulkhead_name:
            if bulkhead_name in self.executors:
                self.executors[bulkhead_name].shutdown()
                del self.executors[bulkhead_name]
                del self.semaphores[bulkhead_name]
                del self.metrics[bulkhead_name]
                logger.info(f"Shutdown bulkhead: {bulkhead_name}")
        else:
            for executor in self.executors.values():
                executor.shutdown()
            self.executors.clear()
            self.semaphores.clear()
            self.metrics.clear()
            logger.info("All bulkheads shut down")

# Example usage:
# bulkhead = BulkheadExecutor()
# bulkhead.create_bulkhead("spacecraft_operations", max_concurrent=5, max_queue_size=10)
# 
# def process_telemetry(data):
#     # Process telemetry data
#     pass
# 
# try:
#     result = bulkhead.execute("spacecraft_operations", process_telemetry, telemetry_data)
# except RuntimeError as e:
#     # Handle rejection
#     pass
