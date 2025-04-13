import logging
from typing import Dict, Any, List, Callable, Optional
import uuid
import time

logger = logging.getLogger(__name__)

class SagaStep:
    """Represents a step in a saga transaction with compensation"""
    
    def __init__(self, name: str, saga_id: str):
        """
        Initialize a saga step.
        
        Args:
            name: Name of the step
            saga_id: ID of the parent saga
        """
        self.name = name
        self.saga_id = saga_id
        self.step_id = str(uuid.uuid4())
        self.start_time = time.time()
        self.end_time = None
        self.status = "PENDING"  # PENDING, COMPLETED, FAILED, COMPENSATED
        self.compensation_data: Dict[str, Any] = {}
        self.result: Optional[Any] = None
        self.error: Optional[Exception] = None
    
    def complete(self, result: Any = None) -> None:
        """
        Mark step as completed.
        
        Args:
            result: Optional result data
        """
        self.status = "COMPLETED"
        self.end_time = time.time()
        self.result = result
        duration_ms = (self.end_time - self.start_time) * 1000
        logger.info(f"Saga {self.saga_id} - Step {self.name} completed in {duration_ms:.2f}ms")
    
    def fail(self, error: Exception) -> None:
        """
        Mark step as failed.
        
        Args:
            error: The error that caused the failure
        """
        self.status = "FAILED"
        self.end_time = time.time()
        self.error = error
        duration_ms = (self.end_time - self.start_time) * 1000
        logger.error(f"Saga {self.saga_id} - Step {self.name} failed after {duration_ms:.2f}ms: {str(error)}")
    
    def compensate(self) -> None:
        """Mark step as compensated"""
        self.status = "COMPENSATED"
        logger.info(f"Saga {self.saga_id} - Step {self.name} compensated")
    
    def set_compensation_data(self, data: Dict[str, Any]) -> None:
        """
        Set data needed for compensation.
        
        Args:
            data: Data for compensation handling
        """
        self.compensation_data = data
        
    def __str__(self) -> str:
        return f"SagaStep(name={self.name}, status={self.status})"


class Saga:
    """
    Implements the Saga pattern for distributed transactions.
    Provides coordination of steps with compensation on failure.
    """
    
    def __init__(self, saga_id: str, name: str):
        """
        Initialize a saga transaction.
        
        Args:
            saga_id: Unique ID for the saga
            name: Name of the saga
        """
        self.saga_id = saga_id
        self.name = name
        self.steps: List[SagaStep] = []
        self.compensations: Dict[str, Callable] = {}
        self.start_time = time.time()
        self.end_time = None
        self.status = "RUNNING"  # RUNNING, COMPLETED, FAILED
        self.current_step_index = -1
    
    def register_compensation(self, step_name: str, compensation_func: Callable) -> None:
        """
        Register a compensation function for a step.
        
        Args:
            step_name: Name of the step
            compensation_func: Function to call for compensation
        """
        self.compensations[step_name] = compensation_func
        logger.debug(f"Saga {self.saga_id} - Registered compensation for step {step_name}")
    
    def start_step(self, step_name: str) -> SagaStep:
        """
        Start a new step in the saga.
        
        Args:
            step_name: Name of the step
            
        Returns:
            The created step
        """
        step = SagaStep(step_name, self.saga_id)
        self.steps.append(step)
        self.current_step_index += 1
        logger.debug(f"Saga {self.saga_id} - Started step {step_name}")
        return step
    
    def complete_step(self, step_name: str, result: Any = None, compensation_data: Dict[str, Any] = None) -> None:
        """
        Complete the current step.
        
        Args:
            step_name: Name of the step to complete
            result: Optional result data
            compensation_data: Optional data for compensation
        """
        # Find the step by name
        step = self._get_current_step()
        
        if not step or step.name != step_name:
            raise ValueError(f"Cannot complete step {step_name} - not the current step")
        
        step.complete(result)
        
        if compensation_data:
            step.set_compensation_data(compensation_data)
    
    def fail_step(self, step_name: str, error: Exception) -> None:
        """
        Mark the current step as failed.
        
        Args:
            step_name: Name of the step
            error: The error that caused the failure
        """
        step = self._get_current_step()
        
        if not step or step.name != step_name:
            raise ValueError(f"Cannot fail step {step_name} - not the current step")
        
        step.fail(error)
        self.status = "FAILED"
    
    def complete(self) -> None:
        """Mark the saga as successfully completed"""
        self.status = "COMPLETED"
        self.end_time = time.time()
        duration_ms = (self.end_time - self.start_time) * 1000
        logger.info(f"Saga {self.saga_id} - {self.name} completed successfully in {duration_ms:.2f}ms")
    
    def compensate(self) -> None:
        """
        Compensate for a failed saga by running compensation
        functions for completed steps in reverse order.
        """
        if self.status != "FAILED":
            self.status = "FAILED"
        
        # Execute compensations in reverse order
        for step in reversed(self.steps):
            if step.status == "COMPLETED":
                logger.info(f"Saga {self.saga_id} - Compensating step {step.name}")
                
                try:
                    # Execute compensation function if registered
                    if step.name in self.compensations:
                        compensation_func = self.compensations[step.name]
                        compensation_func(step.compensation_data)
                    
                    step.compensate()
                except Exception as e:
                    logger.error(f"Saga {self.saga_id} - Error during compensation of {step.name}: {str(e)}")
        
        self.end_time = time.time()
        duration_ms = (self.end_time - self.start_time) * 1000
        logger.info(f"Saga {self.saga_id} - {self.name} compensated in {duration_ms:.2f}ms")
    
    def _get_current_step(self) -> Optional[SagaStep]:
        """Get the current active step"""
        if self.current_step_index < 0 or self.current_step_index >= len(self.steps):
            return None
        return self.steps[self.current_step_index]
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the saga"""
        completed_steps = sum(1 for step in self.steps if step.status == "COMPLETED")
        failed_steps = sum(1 for step in self.steps if step.status == "FAILED")
        
        return {
            "saga_id": self.saga_id,
            "name": self.name,
            "status": self.status,
            "total_steps": len(self.steps),
            "completed_steps": completed_steps,
            "failed_steps": failed_steps,
            "current_step": self._get_current_step().name if self._get_current_step() else None
        }


class SagaManager:
    """Manager for coordinating multiple sagas"""
    
    def __init__(self):
        """Initialize the saga manager"""
        self.sagas: Dict[str, Saga] = {}
    
    def create_saga(self, name: str) -> Saga:
        """
        Create a new saga.
        
        Args:
            name: Name of the saga
            
        Returns:
            The created saga
        """
        saga_id = str(uuid.uuid4())
        saga = Saga(saga_id, name)
        self.sagas[saga_id] = saga
        logger.info(f"Created saga {saga_id} - {name}")
        return saga
    
    def get_saga(self, saga_id: str) -> Optional[Saga]:
        """
        Get a saga by ID.
        
        Args:
            saga_id: ID of the saga
            
        Returns:
            The saga if found, None otherwise
        """
        return self.sagas.get(saga_id)
    
    def cleanup_completed(self, max_age_seconds: int = 3600) -> int:
        """
        Remove completed/failed sagas older than the specified age.
        
        Args:
            max_age_seconds: Maximum age in seconds
            
        Returns:
            Number of sagas removed
        """
        now = time.time()
        to_remove = []
        
        for saga_id, saga in self.sagas.items():
            if saga.status in ["COMPLETED", "FAILED"] and saga.end_time:
                age = now - saga.end_time
                if age > max_age_seconds:
                    to_remove.append(saga_id)
        
        for saga_id in to_remove:
            del self.sagas[saga_id]
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} completed sagas")
        
        return len(to_remove) 