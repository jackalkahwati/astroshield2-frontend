"""Service for event management and orchestration."""

from typing import Dict, List, Any, Optional, Tuple, Type
import logging
import json
from datetime import datetime
from sqlalchemy.orm import Session

from app.models.events import (
    Event, EventType, EventStatus, EventDetection,
    EventQuery, ThreatLevel, CourseOfAction
)
from app.models.event_store import EventORM, EventProcessingLogORM, EventMetricsORM
from app.services.event_processor_base import EventProcessorBase

# Import all event processors
from app.services.processors.launch_processor import LaunchProcessor
from app.services.processors.reentry_processor import ReentryProcessor
from app.services.processors.maneuver_processor import ManeuverProcessor 
from app.services.processors.separation_processor import SeparationProcessor
from app.services.processors.proximity_processor import ProximityProcessor
from app.services.processors.link_change_processor import LinkChangeProcessor
from app.services.processors.attitude_change_processor import AttitudeChangeProcessor

logger = logging.getLogger(__name__)

class EventService:
    """Service for managing and processing events."""
    
    def __init__(self, db: Session):
        """
        Initialize the event service.
        
        Args:
            db: Database session
        """
        self.db = db
        
        # Initialize all event processors
        self.processors: Dict[EventType, EventProcessorBase] = {
            EventType.LAUNCH: LaunchProcessor(),
            EventType.REENTRY: ReentryProcessor(),
            EventType.MANEUVER: ManeuverProcessor(),
            EventType.SEPARATION: SeparationProcessor(),
            EventType.PROXIMITY: ProximityProcessor(),
            EventType.LINK_CHANGE: LinkChangeProcessor(),
            EventType.ATTITUDE_CHANGE: AttitudeChangeProcessor()
        }
    
    async def process_sensor_data(self, data: Dict[str, Any]) -> Optional[Event]:
        """
        Process sensor data and detect any applicable events.
        
        Args:
            data: Raw sensor data
            
        Returns:
            Created event if entry criteria met, None otherwise
        """
        # Check each processor for entry criteria
        for processor in self.processors.values():
            try:
                detection = await processor.detect_entry_criteria(data)
                if detection:
                    # Create and save new event
                    event = processor.create_event(detection)
                    await self.save_event(event)
                    
                    # Start processing the event asynchronously
                    # In a real system, this would be a background task
                    event = await processor.process_event(event)
                    await self.update_event(event)
                    
                    return event
            except Exception as e:
                logger.error(f"Error detecting entry criteria: {str(e)}", exc_info=True)
        
        return None
    
    async def save_event(self, event: Event) -> Event:
        """
        Save an event to the database.
        
        Args:
            event: The event to save
            
        Returns:
            The saved event
        """
        # Convert the event to an ORM model
        event_orm = EventORM(
            id=event.id,
            event_type=event.event_type,
            object_id=event.object_id,
            status=event.status,
            creation_time=event.creation_time,
            update_time=event.update_time,
            detection_data=event.detection_data.dict(),
            processing_steps=[step.dict() for step in event.processing_steps],
            hostility_assessment=event.hostility_assessment,
            threat_level=event.threat_level,
            coa_recommendation=event.coa_recommendation.dict() if event.coa_recommendation else None
        )
        
        self.db.add(event_orm)
        self.db.commit()
        self.db.refresh(event_orm)
        
        return event
    
    async def update_event(self, event: Event) -> Event:
        """
        Update an existing event.
        
        Args:
            event: The event to update
            
        Returns:
            The updated event
        """
        # Get the existing event from the database
        event_orm = self.db.query(EventORM).filter(EventORM.id == event.id).first()
        
        if not event_orm:
            # If not found, create it
            return await self.save_event(event)
        
        # Update the fields
        event_orm.status = event.status
        event_orm.update_time = event.update_time
        event_orm.processing_steps = [step.dict() for step in event.processing_steps]
        event_orm.hostility_assessment = event.hostility_assessment
        event_orm.threat_level = event.threat_level
        event_orm.coa_recommendation = event.coa_recommendation.dict() if event.coa_recommendation else None
        
        self.db.commit()
        self.db.refresh(event_orm)
        
        return event
    
    async def get_event(self, event_id: str) -> Optional[Event]:
        """
        Get an event by ID.
        
        Args:
            event_id: ID of the event to get
            
        Returns:
            The event if found, None otherwise
        """
        event_orm = self.db.query(EventORM).filter(EventORM.id == event_id).first()
        
        if not event_orm:
            return None
        
        # Convert ORM model to Pydantic model
        return self._orm_to_model(event_orm)
    
    async def query_events(self, query: EventQuery) -> List[Event]:
        """
        Query events with filters.
        
        Args:
            query: Query parameters
            
        Returns:
            List of events matching the query
        """
        db_query = self.db.query(EventORM)
        
        # Apply filters
        if query.event_types:
            db_query = db_query.filter(EventORM.event_type.in_([et.value for et in query.event_types]))
        
        if query.object_ids:
            db_query = db_query.filter(EventORM.object_id.in_(query.object_ids))
        
        if query.status:
            db_query = db_query.filter(EventORM.status.in_([s.value for s in query.status]))
        
        if query.start_time:
            db_query = db_query.filter(EventORM.creation_time >= query.start_time)
        
        if query.end_time:
            db_query = db_query.filter(EventORM.creation_time <= query.end_time)
        
        # Apply pagination
        db_query = db_query.order_by(EventORM.creation_time.desc())
        db_query = db_query.offset(query.offset).limit(query.limit)
        
        # Convert results
        events = [self._orm_to_model(event_orm) for event_orm in db_query.all()]
        
        return events
    
    def _orm_to_model(self, event_orm: EventORM) -> Event:
        """
        Convert an ORM model to a Pydantic model.
        
        Args:
            event_orm: The ORM model to convert
            
        Returns:
            Pydantic model
        """
        # Convert processing steps
        processing_steps = []
        for step_dict in event_orm.processing_steps:
            from app.models.events import EventProcessingStep
            processing_steps.append(EventProcessingStep(**step_dict))
        
        # Convert detection data
        detection_data = EventDetection(**event_orm.detection_data)
        
        # Convert COA recommendation if present
        coa_recommendation = None
        if event_orm.coa_recommendation:
            from app.models.events import CourseOfAction
            coa_recommendation = CourseOfAction(**event_orm.coa_recommendation)
        
        return Event(
            id=event_orm.id,
            event_type=event_orm.event_type,
            object_id=event_orm.object_id,
            status=event_orm.status,
            creation_time=event_orm.creation_time,
            update_time=event_orm.update_time,
            detection_data=detection_data,
            processing_steps=processing_steps,
            hostility_assessment=event_orm.hostility_assessment,
            threat_level=event_orm.threat_level,
            coa_recommendation=coa_recommendation
        )
    
    async def process_pending_events(self) -> int:
        """
        Process all pending events.
        
        Returns:
            Number of events processed
        """
        # Query for detected but not completed events
        pending_events = self.db.query(EventORM).filter(
            EventORM.status.in_([EventStatus.DETECTED.value, EventStatus.PROCESSING.value])
        ).all()
        
        count = 0
        for event_orm in pending_events:
            # Convert to model
            event = self._orm_to_model(event_orm)
            
            # Get the appropriate processor
            processor = self.processors[event.event_type]
            
            try:
                # Process the event
                event = await processor.process_event(event)
                await self.update_event(event)
                count += 1
            except Exception as e:
                logger.error(f"Error processing event {event.id}: {str(e)}", exc_info=True)
                
                # Update event with error
                event.status = EventStatus.ERROR
                await self.add_processing_step(
                    event=event,
                    step_name="error",
                    status="failed",
                    error=str(e)
                )
                await self.update_event(event)
        
        return count
    
    async def add_processing_step(self, event: Event, step_name: str, 
                                 status: str, output: Optional[Dict[str, Any]] = None, 
                                 error: Optional[str] = None) -> Event:
        """
        Add a processing step to an event and save it.
        
        Args:
            event: The event to update
            step_name: Name of the processing step
            status: Status of the step
            output: Optional output data
            error: Optional error message
            
        Returns:
            Updated event
        """
        processor = self.processors[event.event_type]
        event = await processor.add_processing_step(event, step_name, status, output, error)
        return await self.update_event(event)