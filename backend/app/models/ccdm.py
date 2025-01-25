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

class CCDMUpdate(BaseModel):
    """Model for CCDM real-time updates"""
    object_id: str
    timestamp: datetime
    update_type: str = Field(..., description="Type of update (e.g., 'shape_change', 'thermal_signature')")
    data: Dict[str, Any]
    confidence: float = Field(..., ge=0.0, le=1.0)
    severity: Optional[str] = Field(None, description="Severity level if applicable")

class CCDMAssessment(BaseModel):
    """Model for CCDM assessment results"""
    object_id: str
    assessment_type: str
    timestamp: datetime
    results: Dict[str, Any]
    confidence_level: float = Field(..., ge=0.0, le=1.0)
    recommendations: List[str]

class HistoricalAnalysis(BaseModel):
    """Model for historical analysis results"""
    object_id: str
    time_range: Dict[str, datetime]
    patterns: List[Dict[str, Any]]
    trend_analysis: Dict[str, Any]
    anomalies: List[Dict[str, Any]]

class CorrelationResult(BaseModel):
    """Model for correlation analysis results"""
    object_ids: List[str]
    timestamp: datetime
    correlations: List[Dict[str, Any]]
    relationship_strength: float = Field(..., ge=0.0, le=1.0)
    significance_level: float

class AnomalyDetection(BaseModel):
    """Model for anomaly detection results"""
    object_id: str
    timestamp: datetime
    anomaly_type: str
    details: Dict[str, Any]
    confidence: float = Field(..., ge=0.0, le=1.0)
    recommended_actions: List[str]

class ObservationRecommendation(BaseModel):
    """Model for observation recommendations"""
    object_id: str
    recommended_times: List[datetime]
    recommended_sensors: List[str]
    observation_parameters: Dict[str, Any]
    priority_level: int = Field(..., ge=1, le=5)

class CCDMReport(BaseModel):
    """Model for comprehensive CCDM reports"""
    object_id: str
    report_timestamp: datetime
    assessment_summary: CCDMAssessment
    historical_data: Optional[HistoricalAnalysis]
    anomalies: List[AnomalyDetection]
    recommendations: List[str]
    confidence_metrics: Dict[str, float] 