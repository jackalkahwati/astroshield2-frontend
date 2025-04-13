"""Models for discrete event processing in the Welder's Arc system."""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum

class EventType(str, Enum):
    """The seven discrete event types in Welder's Arc system."""
    LAUNCH = "launch"
    REENTRY = "reentry"
    MANEUVER = "maneuver"
    SEPARATION = "separation"
    PROXIMITY = "proximity"
    LINK_CHANGE = "link_change"
    ATTITUDE_CHANGE = "attitude_change"

class EventStatus(str, Enum):
    """Status of an event in the processing pipeline."""
    DETECTED = "detected"           # Entry criteria met, processing initiated
    PROCESSING = "processing"       # Being processed by the system
    AWAITING_DATA = "awaiting_data" # Waiting for additional data
    COMPLETED = "completed"         # Processing complete with COA recommendation
    REJECTED = "rejected"           # Determined to be a false positive
    ERROR = "error"                 # Error during processing

class ThreatLevel(str, Enum):
    """Threat level assessment."""
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"

class CourseOfAction(BaseModel):
    """Recommended course of action."""
    title: str
    description: str
    priority: int = Field(..., ge=1, le=5)
    actions: List[str]
    expiration: Optional[datetime] = None

class EventDetection(BaseModel):
    """Entry criteria detection for an event."""
    event_type: EventType
    object_id: str
    detection_time: datetime = Field(default_factory=datetime.utcnow)
    confidence: float = Field(..., ge=0.0, le=1.0)
    sensor_data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

class EventProcessingStep(BaseModel):
    """A step in the event processing pipeline."""
    step_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: str
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class Event(BaseModel):
    """A discrete event being processed by the system."""
    id: str
    event_type: EventType
    object_id: str
    status: EventStatus
    creation_time: datetime = Field(default_factory=datetime.utcnow)
    update_time: datetime = Field(default_factory=datetime.utcnow)
    detection_data: EventDetection
    processing_steps: List[EventProcessingStep] = []
    hostility_assessment: Optional[Dict[str, Any]] = None
    threat_level: Optional[ThreatLevel] = None
    coa_recommendation: Optional[CourseOfAction] = None
    
    class Config:
        schema_extra = {
            "example": {
                "id": "evt-1234567",
                "event_type": "maneuver",
                "object_id": "sat-45678",
                "status": "processing",
                "creation_time": "2025-03-23T12:34:56.789Z",
                "update_time": "2025-03-23T12:35:45.123Z",
                "detection_data": {
                    "event_type": "maneuver",
                    "object_id": "sat-45678",
                    "detection_time": "2025-03-23T12:34:56.789Z",
                    "confidence": 0.92,
                    "sensor_data": {
                        "delta_v": 1.5,
                        "direction": [0.1, 0.2, 0.7],
                        "sensor_id": "radar-12345"
                    }
                }
            }
        }

# Specific event detection models

class LaunchDetection(BaseModel):
    """Launch event detection criteria."""
    object_id: str
    launch_site: Dict[str, float]  # lat/lon coordinates
    launch_time: datetime
    initial_trajectory: List[float]
    confidence: float = Field(..., ge=0.0, le=1.0)
    sensor_id: str

class ReentryDetection(BaseModel):
    """Reentry event detection criteria."""
    object_id: str
    predicted_reentry_time: datetime
    predicted_location: Dict[str, float]  # lat/lon coordinates
    confidence: float = Field(..., ge=0.0, le=1.0)
    sensor_id: str

class ManeuverDetection(BaseModel):
    """Maneuver event detection criteria."""
    object_id: str
    maneuver_time: datetime
    delta_v: float  # m/s
    direction: List[float]  # unit vector
    confidence: float = Field(..., ge=0.0, le=1.0)
    sensor_id: str

class SeparationDetection(BaseModel):
    """Separation event detection criteria."""
    parent_object_id: str
    child_object_id: str
    separation_time: datetime
    relative_velocity: float  # m/s
    confidence: float = Field(..., ge=0.0, le=1.0)
    sensor_id: str

class ProximityDetection(BaseModel):
    """Proximity event detection criteria."""
    primary_object_id: str
    secondary_object_id: str
    closest_approach_time: datetime
    minimum_distance: float  # meters
    relative_velocity: float  # m/s
    confidence: float = Field(..., ge=0.0, le=1.0)
    sensor_id: str

class LinkChangeDetection(BaseModel):
    """Link change event detection criteria."""
    object_id: str
    link_change_time: datetime
    link_type: str  # e.g., "RF", "optical", etc.
    previous_state: str
    current_state: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    sensor_id: str

class AttitudeChangeDetection(BaseModel):
    """Attitude change event detection criteria."""
    object_id: str
    attitude_change_time: datetime
    change_magnitude: float  # degrees
    previous_attitude: List[float]  # quaternion
    current_attitude: List[float]  # quaternion
    confidence: float = Field(..., ge=0.0, le=1.0)
    sensor_id: str

# Event processing models

class EventProcessingError(Exception):
    """Exception raised during event processing."""
    pass

class EventQuery(BaseModel):
    """Query parameters for retrieving events."""
    event_types: Optional[List[EventType]] = None
    object_ids: Optional[List[str]] = None
    status: Optional[List[EventStatus]] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    limit: int = 100
    offset: int = 0