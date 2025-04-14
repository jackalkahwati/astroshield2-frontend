"""
Trajectory data models
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class ObjectProperties(BaseModel):
    mass: float = Field(100.0, description="Mass of the object in kg")
    area: float = Field(1.2, description="Cross-sectional area in m²")
    cd: float = Field(2.2, description="Drag coefficient")

class BreakupModel(BaseModel):
    enabled: bool = Field(True, description="Whether breakup modeling is enabled")
    fragmentation_threshold: float = Field(50.0, description="Energy threshold for fragmentation in kJ")

class TrajectoryConfig(BaseModel):
    object_name: str = Field("Satellite Debris", description="Name of the object being analyzed")
    object_properties: ObjectProperties = Field(default_factory=ObjectProperties, description="Physical properties of the object")
    atmospheric_model: str = Field("exponential", description="Atmospheric model to use")
    wind_model: str = Field("custom", description="Wind model to use")
    monte_carlo_samples: int = Field(100, description="Number of Monte Carlo samples for uncertainty")
    breakup_model: BreakupModel = Field(default_factory=BreakupModel, description="Configuration for breakup modeling")

class TrajectoryRequest(BaseModel):
    config: TrajectoryConfig = Field(..., description="Configuration for the trajectory analysis")
    initial_state: List[float] = Field(..., description="Initial state vector [lon, lat, alt, vx, vy, vz]")

class TrajectoryPoint(BaseModel):
    time: float
    position: List[float]
    velocity: List[float]

class ImpactPrediction(BaseModel):
    time: float
    position: List[float]
    confidence: float
    energy: float
    area: float

class BreakupPoint(BaseModel):
    time: float
    position: List[float]
    fragments: int
    cause: str

class TrajectoryResult(BaseModel):
    trajectory: List[TrajectoryPoint]
    impactPrediction: ImpactPrediction
    breakupPoints: List[BreakupPoint] 