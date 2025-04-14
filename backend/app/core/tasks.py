from fastapi import FastAPI
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from datetime import datetime
from app.core.security import key_store
import threading
import time
import logging
import queue
import uuid
from typing import Dict, Any, Callable, Optional, List, Tuple, Union
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class TaskStatus:
    """Task status constants"""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"

class TaskManager:
    """
    Task manager for handling background tasks.
    Implements a simple task queue with status tracking.
    """
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        """Singleton pattern implementation"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(TaskManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """Initialize the task manager"""
        if self._initialized:
            return
            
        self._initialized = True
        self.tasks = {}  # task_id -> task_info
        self.task_queue = queue.Queue()
        self.workers = []
        self.max_workers = 5
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.async_executor = None  # Will be set when async loop is available
        
        # Start worker threads
        self._start_workers()
        
        logger.info(f"TaskManager initialized with {self.max_workers} workers")
    
    def _start_workers(self):
        """Start worker threads to process tasks"""
        for i in range(self.max_workers):
            worker = threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            worker.start()
            self.workers.append(worker)
    
    def _worker_loop(self, worker_id: int):
        """Worker loop for processing tasks"""
        logger.info(f"Worker {worker_id} started")
        
        while True:
            try:
                # Get task from queue
                task_id = self.task_queue.get()
                
                if task_id is None:
                    # None is a signal to stop the worker
                    self.task_queue.task_done()
                    break
                
                # Get task info
                with self._lock:
                    if task_id not in self.tasks:
                        logger.warning(f"Task {task_id} not found in task list")
                        self.task_queue.task_done()
                        continue
                        
                    task_info = self.tasks[task_id]
                    task_info["status"] = TaskStatus.RUNNING
                    task_info["start_time"] = datetime.utcnow()
                
                logger.info(f"Worker {worker_id} processing task {task_id} ({task_info['name']})")
                
                # Execute task function
                try:
                    result = task_info["function"](*task_info["args"], **task_info["kwargs"])
                    
                    # Update task info with result
                    with self._lock:
                        if task_id in self.tasks:  # Check if task still exists
                            self.tasks[task_id].update({
                                "status": TaskStatus.COMPLETED,
                                "result": result,
                                "end_time": datetime.utcnow()
                            })
                            
                    logger.info(f"Task {task_id} completed successfully")
                    
                except Exception as e:
                    logger.exception(f"Error executing task {task_id}: {str(e)}")
                    
                    # Update task info with error
                    with self._lock:
                        if task_id in self.tasks:  # Check if task still exists
                            self.tasks[task_id].update({
                                "status": TaskStatus.FAILED,
                                "error": str(e),
                                "end_time": datetime.utcnow()
                            })
                
                # Mark task as done in queue
                self.task_queue.task_done()
                
            except Exception as e:
                logger.exception(f"Error in worker {worker_id}: {str(e)}")
    
    def submit_task(self, name: str, function: Callable, *args, **kwargs) -> str:
        """
        Submit a task for background processing.
        
        Args:
            name: Task name
            function: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            task_id: Unique ID for the task
        """
        task_id = str(uuid.uuid4())
        
        # Create task info
        task_info = {
            "id": task_id,
            "name": name,
            "status": TaskStatus.PENDING,
            "function": function,
            "args": args,
            "kwargs": kwargs,
            "submit_time": datetime.utcnow(),
            "start_time": None,
            "end_time": None,
            "result": None,
            "error": None
        }
        
        # Add task to list
        with self._lock:
            self.tasks[task_id] = task_info
        
        # Add task to queue
        self.task_queue.put(task_id)
        
        logger.info(f"Task {task_id} ({name}) submitted")
        
        return task_id
    
    async def submit_async_task(self, name: str, function: Callable, *args, **kwargs) -> str:
        """
        Submit an async task for background processing.
        
        Args:
            name: Task name
            function: Async function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            task_id: Unique ID for the task
        """
        if self.async_executor is None:
            # Get current event loop
            loop = asyncio.get_event_loop()
            self.async_executor = loop
        
        task_id = str(uuid.uuid4())
        
        # Create task info
        task_info = {
            "id": task_id,
            "name": name,
            "status": TaskStatus.PENDING,
            "function": function,
            "args": args,
            "kwargs": kwargs,
            "submit_time": datetime.utcnow(),
            "start_time": None,
            "end_time": None,
            "result": None,
            "error": None,
            "is_async": True
        }
        
        # Add task to list
        with self._lock:
            self.tasks[task_id] = task_info
        
        # Run async task in background
        asyncio.create_task(self._run_async_task(task_id))
        
        logger.info(f"Async task {task_id} ({name}) submitted")
        
        return task_id
    
    async def _run_async_task(self, task_id: str):
        """Run an async task in the background"""
        with self._lock:
            if task_id not in self.tasks:
                logger.warning(f"Async task {task_id} not found in task list")
                return
                
            task_info = self.tasks[task_id]
            task_info["status"] = TaskStatus.RUNNING
            task_info["start_time"] = datetime.utcnow()
        
        logger.info(f"Processing async task {task_id} ({task_info['name']})")
        
        # Execute task function
        try:
            result = await task_info["function"](*task_info["args"], **task_info["kwargs"])
            
            # Update task info with result
            with self._lock:
                if task_id in self.tasks:  # Check if task still exists
                    self.tasks[task_id].update({
                        "status": TaskStatus.COMPLETED,
                        "result": result,
                        "end_time": datetime.utcnow()
                    })
                    
            logger.info(f"Async task {task_id} completed successfully")
            
        except Exception as e:
            logger.exception(f"Error executing async task {task_id}: {str(e)}")
            
            # Update task info with error
            with self._lock:
                if task_id in self.tasks:  # Check if task still exists
                    self.tasks[task_id].update({
                        "status": TaskStatus.FAILED,
                        "error": str(e),
                        "end_time": datetime.utcnow()
                    })
    
    def get_task_info(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            Dict with task information or None if task not found
        """
        with self._lock:
            task_info = self.tasks.get(task_id)
            
            if task_info:
                # Return a copy without the function
                result = task_info.copy()
                result.pop("function", None)
                
                # Calculate duration if possible
                if result["start_time"] and result["end_time"]:
                    duration = (result["end_time"] - result["start_time"]).total_seconds()
                    result["duration_seconds"] = round(duration, 2)
                elif result["start_time"]:
                    duration = (datetime.utcnow() - result["start_time"]).total_seconds()
                    result["duration_seconds"] = round(duration, 2)
                    
                # Calculate estimated remaining time for running tasks
                if result["status"] == TaskStatus.RUNNING and "progress" in result:
                    progress = result["progress"]
                    if progress > 0:
                        elapsed = (datetime.utcnow() - result["start_time"]).total_seconds()
                        estimated_total = elapsed / progress
                        result["estimated_remaining_seconds"] = round(estimated_total - elapsed, 2)
                
                return result
                
            return None
    
    def get_all_tasks(self, limit: int = 100, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get information about all tasks.
        
        Args:
            limit: Maximum number of tasks to return
            status: Filter by status
            
        Returns:
            List of task information dictionaries
        """
        with self._lock:
            tasks = list(self.tasks.values())
            
            # Filter by status if specified
            if status:
                tasks = [t for t in tasks if t["status"] == status]
            
            # Sort by submit time (newest first)
            tasks.sort(key=lambda x: x["submit_time"], reverse=True)
            
            # Limit number of tasks
            tasks = tasks[:limit]
            
            # Remove function from result
            for task in tasks:
                task = task.copy()
                task.pop("function", None)
            
            return tasks
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a pending task.
        
        Args:
            task_id: Task ID
            
        Returns:
            True if task was canceled, False otherwise
        """
        with self._lock:
            if task_id not in self.tasks:
                return False
                
            task_info = self.tasks[task_id]
            
            if task_info["status"] == TaskStatus.PENDING:
                task_info["status"] = TaskStatus.CANCELED
                task_info["end_time"] = datetime.utcnow()
                return True
            
            return False
    
    def update_task_progress(self, task_id: str, progress: float, status_message: Optional[str] = None) -> bool:
        """
        Update progress for a task.
        
        Args:
            task_id: Task ID
            progress: Progress value (0.0 to 1.0)
            status_message: Optional status message
            
        Returns:
            True if task was updated, False otherwise
        """
        with self._lock:
            if task_id not in self.tasks:
                return False
                
            task_info = self.tasks[task_id]
            
            if task_info["status"] == TaskStatus.RUNNING:
                task_info["progress"] = progress
                
                if status_message:
                    task_info["status_message"] = status_message
                
                return True
            
            return False
    
    def cleanup_old_tasks(self, max_age_hours: int = 24) -> int:
        """
        Clean up old completed tasks.
        
        Args:
            max_age_hours: Maximum age of tasks to keep in hours
            
        Returns:
            Number of tasks removed
        """
        now = datetime.utcnow()
        removed = 0
        
        with self._lock:
            task_ids = list(self.tasks.keys())
            
            for task_id in task_ids:
                task_info = self.tasks[task_id]
                
                # Only remove completed, failed, or canceled tasks
                if task_info["status"] not in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELED]:
                    continue
                
                # Check if task is old enough
                end_time = task_info["end_time"]
                if end_time and (now - end_time).total_seconds() > max_age_hours * 3600:
                    del self.tasks[task_id]
                    removed += 1
        
        return removed
    
    def shutdown(self):
        """Shutdown the task manager"""
        logger.info("Shutting down TaskManager")
        
        # Stop all workers
        for _ in range(len(self.workers)):
            self.task_queue.put(None)
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5.0)
        
        # Shutdown executor
        self.executor.shutdown(wait=False)
        
        logger.info("TaskManager shutdown complete")

# Create singleton instance
task_manager = TaskManager()

# Helper functions for simpler usage
def submit_background_task(name: str, function: Callable, *args, **kwargs) -> str:
    """Submit a task for background processing"""
    return task_manager.submit_task(name, function, *args, **kwargs)

async def submit_async_background_task(name: str, function: Callable, *args, **kwargs) -> str:
    """Submit an async task for background processing"""
    return await task_manager.submit_async_task(name, function, *args, **kwargs)

def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    """Get task status information"""
    return task_manager.get_task_info(task_id)

def update_task_progress(task_id: str, progress: float, message: Optional[str] = None) -> bool:
    """Update task progress"""
    return task_manager.update_task_progress(task_id, progress, message)

def get_active_tasks() -> List[Dict[str, Any]]:
    """Get all active tasks"""
    return task_manager.get_all_tasks(status=TaskStatus.RUNNING)

def setup_periodic_tasks(app: FastAPI):
    """Setup periodic tasks for the application"""
    scheduler = AsyncIOScheduler()
    
    # Add periodic tasks
    scheduler.add_job(rotate_keys, 'interval', hours=24, args=[app])
    
    # Start the scheduler
    scheduler.start()
    
    # Store the scheduler instance
    app.state.scheduler = scheduler

async def rotate_keys(app: FastAPI):
    """Rotate encryption keys periodically"""
    key_store.rotate_keys()
    print(f"Rotated encryption keys at {datetime.utcnow().isoformat()}")

# Schedule key cleanup every week
@scheduler.scheduled_job("interval", days=7)
def cleanup_old_keys_job():
    print(f"[{datetime.utcnow()}] Cleaning up old encryption keys...")
    key_store.cleanup_old_keys(max_age_days=30)
    print(f"[{datetime.utcnow()}] Key cleanup complete") 