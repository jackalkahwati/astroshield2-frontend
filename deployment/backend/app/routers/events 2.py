"""API endpoints for event detection and management."""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from datetime import datetime
import logging

from app.db.session import get_db
from app.models.events import (
    Event, EventType, EventStatus, EventQuery, 
    EventDetection, ThreatLevel, CourseOfAction
)
from app.services.event_service import EventService

router = APIRouter(prefix="/events", tags=["events"])
logger = logging.getLogger(__name__)

@router.post("/detect", response_model=Optional[Event], summary="Submit sensor data for event detection")
async def detect_event(
    data: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """
    Submit sensor data for event detection.
    
    This endpoint analyzes the provided sensor data against the entry criteria
    for all event types. If an event is detected, it will be created, 
    processed, and returned.
    
    - **data**: Raw sensor data that may indicate an event
    """
    try:
        event_service = EventService(db)
        event = await event_service.process_sensor_data(data)
        
        if not event:
            return None
            
        return event
    except Exception as e:
        logger.error(f"Error processing sensor data: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing sensor data: {str(e)}")

@router.get("/{event_id}", response_model=Event, summary="Get event by ID")
async def get_event(
    event_id: str,
    db: Session = Depends(get_db)
):
    """
    Get an event by its ID.
    
    - **event_id**: The ID of the event to retrieve
    """
    try:
        event_service = EventService(db)
        event = await event_service.get_event(event_id)
        
        if not event:
            raise HTTPException(status_code=404, detail=f"Event {event_id} not found")
            
        return event
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving event: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving event: {str(e)}")

@router.post("/query", response_model=List[Event], summary="Query events with filters")
async def query_events(
    query: EventQuery,
    db: Session = Depends(get_db)
):
    """
    Query events with filters.
    
    - **query**: Query parameters including event types, object IDs, status, time range, etc.
    """
    try:
        event_service = EventService(db)
        events = await event_service.query_events(query)
        return events
    except Exception as e:
        logger.error(f"Error querying events: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error querying events: {str(e)}")

@router.get("/types", response_model=List[str], summary="Get all event types")
async def get_event_types():
    """
    Get a list of all supported event types.
    """
    return [e.value for e in EventType]

@router.get("/status", response_model=List[str], summary="Get all event status values")
async def get_event_status():
    """
    Get a list of all possible event status values.
    """
    return [s.value for s in EventStatus]

@router.get("/threat-levels", response_model=List[str], summary="Get all threat levels")
async def get_threat_levels():
    """
    Get a list of all possible threat levels.
    """
    return [t.value for t in ThreatLevel]

@router.post("/process-pending", response_model=int, summary="Process all pending events")
async def process_pending_events(
    db: Session = Depends(get_db)
):
    """
    Process all pending events.
    
    This endpoint triggers processing for all events that are in DETECTED or PROCESSING
    status but have not yet completed. Returns the number of events processed.
    """
    try:
        event_service = EventService(db)
        count = await event_service.process_pending_events()
        return count
    except Exception as e:
        logger.error(f"Error processing pending events: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing pending events: {str(e)}")

@router.get("/dashboard", response_model=Dict[str, Any], summary="Get event dashboard data")
async def get_dashboard_data(
    db: Session = Depends(get_db),
    start_time: Optional[datetime] = Query(None, description="Filter events after this time"),
    end_time: Optional[datetime] = Query(None, description="Filter events before this time")
):
    """
    Get aggregated event data for dashboard display.
    
    - **start_time**: Optional filter for events after this time
    - **end_time**: Optional filter for events before this time
    """
    try:
        # Create query for the given time range
        query = EventQuery(
            start_time=start_time,
            end_time=end_time,
            limit=1000  # Large limit to get comprehensive stats
        )
        
        # Get events for the time range
        event_service = EventService(db)
        events = await event_service.query_events(query)
        
        # Aggregate event counts by type
        events_by_type = {}
        for event_type in EventType:
            events_by_type[event_type.value] = len([e for e in events if e.event_type == event_type])
            
        # Aggregate event counts by status
        events_by_status = {}
        for status in EventStatus:
            events_by_status[status.value] = len([e for e in events if e.status == status])
            
        # Aggregate event counts by threat level
        events_by_threat = {}
        for threat in ThreatLevel:
            events_by_threat[threat.value] = len([e for e in events if e.threat_level == threat])
            
        # Get recent high-threat events
        high_threat_events = [e for e in events if e.threat_level in [ThreatLevel.HIGH, ThreatLevel.MODERATE]]
        high_threat_events.sort(key=lambda e: e.creation_time, reverse=True)
        recent_high_threats = high_threat_events[:5]  # Get 5 most recent
            
        # Compile dashboard data
        dashboard_data = {
            "total_events": len(events),
            "events_by_type": events_by_type,
            "events_by_status": events_by_status,
            "events_by_threat": events_by_threat,
            "recent_high_threats": recent_high_threats
        }
        
        return dashboard_data
    except Exception as e:
        logger.error(f"Error generating dashboard data: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating dashboard data: {str(e)}")