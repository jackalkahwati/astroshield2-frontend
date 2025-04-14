"""
CCDM data models
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

class ConjunctionObject(BaseModel):
    """Object involved in a conjunction"""
    object_id: str
    name: str = ""
    type: str = "UNKNOWN"
    size: float = 0.0  # Size in meters
    mass: Optional[float] = None  # Mass in kg if known
    owner: Optional[str] = None
    country: Optional[str] = None

class ConjunctionEvent(BaseModel):
    """Conjunction event data"""
    id: str
    time_of_closest_approach: str  # ISO format date string
    miss_distance: float  # Distance in meters
    probability_of_collision: float
    relative_velocity: float  # Relative velocity in m/s
    primary_object: ConjunctionObject
    secondary_object: ConjunctionObject
    created_at: str  # ISO format date string
    updated_at: Optional[str] = None
    status: str = "PENDING"  # PENDING, ANALYZING, RESOLVED, AVOIDED

class ConjunctionCreateRequest(BaseModel):
    """Request to create a new conjunction event"""
    time_of_closest_approach: str
    miss_distance: float
    probability_of_collision: float
    relative_velocity: float
    primary_object: ConjunctionObject
    secondary_object: ConjunctionObject

class ConjunctionFilterRequest(BaseModel):
    """Request to filter conjunction events"""
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    object_id: Optional[str] = None
    min_probability: Optional[float] = None
    status: Optional[str] = None 