from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class ConfidenceLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class ObservationData(BaseModel):
    timestamp: datetime
    sensor_id: str
    measurements: Dict[str, float]
    metadata: Optional[Dict[str, Any]] = None

class ObjectAnalysisRequest(BaseModel):
    object_id: str
    observation_data: ObservationData

class ObjectAnalysisResponse(BaseModel):
    object_id: str
    ccdm_assessment: str
    confidence_level: float = Field(..., ge=0.0, le=1.0)
    timestamp: datetime
    details: Optional[Dict[str, Any]] = None

class ShapeChangeMetrics(BaseModel):
    volume_change: float
    surface_area_change: float
    aspect_ratio_change: float
    confidence: float = Field(..., ge=0.0, le=1.0)

class ShapeChangeResponse(BaseModel):
    object_id: str
    start_time: datetime
    end_time: datetime
    detected_changes: List[ShapeChangeMetrics]
    analysis_confidence: ConfidenceLevel
    timestamp: datetime

class ThermalSignatureMetrics(BaseModel):
    temperature_kelvin: float
    heat_signature_pattern: str
    emission_spectrum: Dict[str, float]
    anomaly_score: float = Field(..., ge=0.0, le=1.0)

class ThermalSignatureResponse(BaseModel):
    object_id: str
    timestamp: datetime
    metrics: ThermalSignatureMetrics
    historical_comparison: Optional[Dict[str, float]] = None
    confidence_level: ConfidenceLevel

class PropulsionType(str, Enum):
    CHEMICAL = "chemical"
    ELECTRIC = "electric"
    UNKNOWN = "unknown"

class PropulsiveCapabilityMetrics(BaseModel):
    estimated_thrust: float
    propulsion_type: PropulsionType
    maneuver_capability_score: float = Field(..., ge=0.0, le=1.0)
    fuel_reserve_estimate: Optional[float] = None

class PropulsiveCapabilityResponse(BaseModel):
    object_id: str
    analysis_period: int
    metrics: PropulsiveCapabilityMetrics
    historical_events: List[Dict[str, Any]]
    confidence_level: ConfidenceLevel
    timestamp: datetime 