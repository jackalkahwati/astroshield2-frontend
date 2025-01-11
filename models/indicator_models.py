from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

class SystemInteraction(BaseModel):
    """Model for system interaction data"""
    system_id: str
    interaction_type: str = Field(..., description="Type of interaction (RF/OPTICAL)")
    interaction_time: datetime
    frequency: Optional[float] = Field(None, description="RF frequency in MHz")
    wavelength: Optional[float] = Field(None, description="Optical wavelength in nm")
    power: float = Field(..., description="Interaction power in watts")
    duration: float = Field(..., description="Interaction duration in seconds")
    location: dict = Field(..., description="Location of the interaction")

class EclipsePeriod(BaseModel):
    """Model for eclipse period data"""
    start_time: datetime
    end_time: datetime
    eclipse_type: str = Field(..., description="Type of eclipse (UMBRA/PENUMBRA)")
    spacecraft_id: str
    orbit_position: dict = Field(..., description="Orbital position during eclipse")

class TrackingData(BaseModel):
    """Model for object tracking data"""
    object_id: str
    timestamp: datetime
    position: dict = Field(..., description="Object position")
    velocity: dict = Field(..., description="Object velocity")
    tracking_source: str
    confidence: float = Field(..., ge=0, le=1)
    uncorrelated_tracks: List[dict] = Field(default_factory=list)

class UNRegistryEntry(BaseModel):
    """Model for UN registry entry data"""
    international_designator: str
    registration_date: datetime
    state_of_registry: str
    launch_date: datetime
    orbital_parameters: dict
    function: str
    status: str = Field(..., description="Registration status")

class OrbitOccupancyData(BaseModel):
    """Model for orbit occupancy analysis"""
    region_id: str
    timestamp: datetime
    total_objects: int
    object_density: float  # objects per cubic km
    typical_density: float  # historical average
    orbital_band: dict = Field(..., description="Orbital band parameters")
    neighboring_objects: List[str] = Field(default_factory=list)

class StimulationEvent(BaseModel):
    """Model for system stimulation events"""
    event_id: str
    timestamp: datetime
    spacecraft_id: str
    stimulation_type: str = Field(..., description="Type of stimulation")
    source_system: str
    response_characteristics: dict
    confidence: float = Field(..., ge=0, le=1)
    evidence: dict

class LaunchTrackingData(BaseModel):
    """Model for launch tracking analysis"""
    launch_id: str
    timestamp: datetime
    expected_objects: int
    tracked_objects: List[str]
    tracking_status: str
    confidence_metrics: dict
    anomalies: List[dict] = Field(default_factory=list)
