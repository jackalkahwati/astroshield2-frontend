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

class ShapeChangeResponse(BaseModel):
    detected: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    timestamp: datetime

class ThermalSignatureResponse(BaseModel):
    detected: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    timestamp: datetime

class PropulsiveCapabilityResponse(BaseModel):
    detected: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    timestamp: datetime

class ObjectAnalysisRequest(BaseModel):
    object_id: str
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

class ObjectAnalysisResponse(BaseModel):
    object_id: str
    timestamp: datetime
    analysis_complete: bool
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    shape_change: ShapeChangeResponse
    thermal_signature: ThermalSignatureResponse
    propulsive_capability: PropulsiveCapabilityResponse

class ShapeChangeMetrics(BaseModel):
    volume_change: float
    surface_area_change: float
    aspect_ratio_change: float
    confidence: float = Field(..., ge=0.0, le=1.0)

class ThermalSignatureMetrics(BaseModel):
    temperature_kelvin: float
    anomaly_score: float = Field(..., ge=0.0, le=1.0)

class PropulsionType(str, Enum):
    CHEMICAL = "chemical"
    ELECTRIC = "electric"
    NUCLEAR = "nuclear"
    UNKNOWN = "unknown"

class PropulsionMetrics(BaseModel):
    type: PropulsionType
    thrust_estimate: Optional[float] = None
    fuel_reserve_estimate: Optional[float] = None

class CCDMUpdate(BaseModel):
    """Model for CCDM real-time updates"""
    object_id: str
    timestamp: datetime
    update_type: str
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

class AnalysisPoint(BaseModel):
    """Model for a single analysis point in historical data"""
    timestamp: str
    threat_level: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    details: Dict[str, Any]

class HistoricalAnalysis(BaseModel):
    """Model for historical analysis results"""
    norad_id: int
    start_date: str
    end_date: str
    trend_summary: str
    analysis_points: List[AnalysisPoint]
    metadata: Dict[str, Any]

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