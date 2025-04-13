"""Base class for all event processors."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import uuid
import logging

from app.models.events import (
    Event, EventType, EventStatus, EventDetection,
    EventProcessingStep, ThreatLevel, CourseOfAction,
    EventProcessingError
)

logger = logging.getLogger(__name__)

class EventProcessorBase(ABC):
    """Base class for all event processors."""
    
    def __init__(self):
        self.event_type = None  # Override in subclasses

    @abstractmethod
    async def detect_entry_criteria(self, data: Dict[str, Any]) -> Optional[EventDetection]:
        """
        Check if data meets entry criteria for this event type.
        
        Args:
            data: Input data to check against entry criteria
            
        Returns:
            EventDetection if criteria are met, None otherwise
        """
        pass
    
    @abstractmethod
    async def process_event(self, event: Event) -> Event:
        """
        Process an event from detection to completion.
        
        Args:
            event: The event to process
            
        Returns:
            Updated event with processing results
        """
        pass
    
    @abstractmethod
    async def assess_hostility(self, event: Event) -> Tuple[Dict[str, Any], ThreatLevel]:
        """
        Assess hostility level of an event.
        
        Args:
            event: The event to assess
            
        Returns:
            Tuple of (assessment_details, threat_level)
        """
        pass
    
    @abstractmethod
    async def generate_coa(self, event: Event, threat_level: ThreatLevel) -> CourseOfAction:
        """
        Generate course of action recommendation.
        
        Args:
            event: The event to generate recommendations for
            threat_level: Assessed threat level
            
        Returns:
            Recommended course of action
        """
        pass
    
    def create_event(self, detection: EventDetection) -> Event:
        """
        Create a new event from a detection.
        
        Args:
            detection: The detection data
            
        Returns:
            A new Event object
        """
        event_id = f"evt-{uuid.uuid4().hex[:12]}"
        
        return Event(
            id=event_id,
            event_type=detection.event_type,
            object_id=detection.object_id,
            status=EventStatus.DETECTED,
            creation_time=datetime.utcnow(),
            update_time=datetime.utcnow(),
            detection_data=detection,
            processing_steps=[]
        )
    
    async def add_processing_step(self, event: Event, step_name: str, 
                                 status: str, output: Optional[Dict[str, Any]] = None, 
                                 error: Optional[str] = None) -> Event:
        """
        Add a processing step to an event.
        
        Args:
            event: The event to update
            step_name: Name of the processing step
            status: Status of the step
            output: Optional output data
            error: Optional error message
            
        Returns:
            Updated event
        """
        step = EventProcessingStep(
            step_name=step_name,
            timestamp=datetime.utcnow(),
            status=status,
            output=output,
            error=error
        )
        
        # Create a new list with the additional step
        event.processing_steps.append(step)
        event.update_time = datetime.utcnow()
        
        return event
        
    async def complete_event(self, event: Event, threat_level: ThreatLevel, 
                            hostility_assessment: Dict[str, Any], 
                            coa: CourseOfAction) -> Event:
        """
        Mark an event as completed with final assessment and recommendation.
        
        Args:
            event: The event to complete
            threat_level: Assessed threat level
            hostility_assessment: Hostility assessment details
            coa: Recommended course of action
            
        Returns:
            Completed event
        """
        event.status = EventStatus.COMPLETED
        event.threat_level = threat_level
        event.hostility_assessment = hostility_assessment
        event.coa_recommendation = coa
        event.update_time = datetime.utcnow()
        
        # Add final processing step
        await self.add_processing_step(
            event=event, 
            step_name="event_completion", 
            status="success",
            output={
                "threat_level": threat_level,
                "coa_recommended": coa.title
            }
        )
        
        return event