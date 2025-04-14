"""
Maneuver data models
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timedelta

class ManeuverDetails(BaseModel):
    delta_v: Optional[float] = None
    duration: Optional[float] = None
    fuel_required: Optional[float] = None
    fuel_used: Optional[float] = None
    target_orbit: Optional[Dict[str, float]] = None

class ManeuverCreateRequest(BaseModel):
    satellite_id: Optional[str] = "SAT-001"
    type: str
    status: str = "scheduled"
    scheduledTime: str
    details: ManeuverDetails

class Maneuver(BaseModel):
    id: str
    satellite_id: str
    type: str
    status: str
    scheduledTime: str
    completedTime: Optional[str] = None
    created_by: Optional[str] = "system"
    created_at: Optional[str] = None
    details: ManeuverDetails 