from typing import Dict, List, Callable, Any
from enum import Enum
import logging
from .event_bus import EventBus

logger = logging.getLogger(__name__)

class SagaState(Enum):
    STARTED = "STARTED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    COMPENSATING = "COMPENSATING"

class SagaStep:
    def __init__(
        self,
        action: Callable[..., Any],
        compensation: Callable[..., Any],
        name: str
    ):
        self.action = action
        self.compensation = compensation
        self.name = name
        self.state = SagaState.STARTED
        self.data = {}

class Saga:
    def __init__(self, name: str):
        self.name = name
        self.steps: List[SagaStep] = []
        self.current_step = 0
        self.state = SagaState.STARTED
        self.event_bus = EventBus()

    def add_step(
        self,
        action: Callable[..., Any],
        compensation: Callable[..., Any],
        name: str
    ) -> 'Saga':
        """Add a step to the saga"""
        self.steps.append(SagaStep(action, compensation, name))
        return self

    def execute(self, context: Dict[str, Any] = None) -> bool:
        """Execute the saga"""
        if context is None:
            context = {}

        try:
            logger.info(f"Starting saga: {self.name}")
            self._publish_event("saga.started", {"name": self.name})

            # Execute each step
            while self.current_step < len(self.steps):
                step = self.steps[self.current_step]
                try:
                    logger.info(f"Executing step: {step.name}")
                    result = step.action(context)
                    step.data = result if result else {}
                    step.state = SagaState.COMPLETED
                    self._publish_event("saga.step.completed", {
                        "saga": self.name,
                        "step": step.name
                    })
                except Exception as e:
                    logger.error(f"Step {step.name} failed: {str(e)}")
                    step.state = SagaState.FAILED
                    self._publish_event("saga.step.failed", {
                        "saga": self.name,
                        "step": step.name,
                        "error": str(e)
                    })
                    self._compensate()
                    return False
                self.current_step += 1

            self.state = SagaState.COMPLETED
            self._publish_event("saga.completed", {"name": self.name})
            return True

        except Exception as e:
            logger.error(f"Saga {self.name} failed: {str(e)}")
            self.state = SagaState.FAILED
            self._publish_event("saga.failed", {
                "name": self.name,
                "error": str(e)
            })
            self._compensate()
            return False

    def _compensate(self):
        """Compensate for failed steps"""
        self.state = SagaState.COMPENSATING
        self._publish_event("saga.compensating", {"name": self.name})

        # Compensate steps in reverse order
        for i in range(self.current_step, -1, -1):
            step = self.steps[i]
            try:
                logger.info(f"Compensating step: {step.name}")
                step.compensation(step.data)
                self._publish_event("saga.step.compensated", {
                    "saga": self.name,
                    "step": step.name
                })
            except Exception as e:
                logger.error(f"Compensation failed for step {step.name}: {str(e)}")
                self._publish_event("saga.compensation.failed", {
                    "saga": self.name,
                    "step": step.name,
                    "error": str(e)
                })

    def _publish_event(self, topic: str, data: Dict[str, Any]):
        """Publish saga events"""
        try:
            self.event_bus.publish(topic, data)
        except Exception as e:
            logger.error(f"Failed to publish event {topic}: {str(e)}")

# Example usage:
# def book_spacecraft(context):
#     # Book spacecraft logic
#     return {"spacecraft_id": "123"}
#
# def cancel_spacecraft_booking(data):
#     # Cancel booking logic
#     pass
#
# saga = Saga("ManeuverOperation")
# saga.add_step(
#     action=book_spacecraft,
#     compensation=cancel_spacecraft_booking,
#     name="BookSpacecraft"
# )
# saga.execute({"mission_id": "ABC123"})
