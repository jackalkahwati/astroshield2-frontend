from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
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
    CHEMICAL = "CHEMICAL"
    ELECTRIC = "ELECTRIC"
    NUCLEAR = "NUCLEAR"
    ION = "ION"
    UNKNOWN = "UNKNOWN"

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
    triggered_indicators: Optional[List[str]] = None
    summary: Optional[str] = None

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

class ThreatLevel(str, Enum):
    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class AnalysisResult(BaseModel):
    timestamp: datetime
    confidence: float
    threat_level: ThreatLevel
    details: Dict

class ObjectAnalysisRequest(BaseModel):
    norad_id: int
    analysis_type: str = "FULL"
    options: Dict = Field(default_factory=dict)

class ObjectAnalysisResponse(BaseModel):
    norad_id: int
    timestamp: datetime
    analysis_results: List[AnalysisResult]
    summary: str
    metadata: Optional[Dict] = None

class ThreatAssessmentRequest(BaseModel):
    norad_id: int
    assessment_factors: List[str] = ["COLLISION", "MANEUVER", "DEBRIS"]

class ObjectThreatAssessment(BaseModel):
    norad_id: int
    timestamp: datetime
    overall_threat: ThreatLevel
    confidence: float
    threat_components: Dict[str, str]
    recommendations: List[str]
    metadata: Optional[Dict] = None

class HistoricalAnalysisRequest(BaseModel):
    norad_id: int
    start_date: datetime
    end_date: datetime

class HistoricalAnalysisPoint(BaseModel):
    timestamp: datetime
    threat_level: ThreatLevel
    confidence: float
    details: Dict

class HistoricalAnalysisResponse(BaseModel):
    norad_id: int
    start_date: datetime
    end_date: datetime
    analysis_points: List[HistoricalAnalysisPoint]
    trend_summary: str
    metadata: Optional[Dict] = None

class ShapeChangeRequest(BaseModel):
    norad_id: int
    start_date: datetime
    end_date: datetime
    sensitivity: float = 0.5

class ShapeChangeDetection(BaseModel):
    timestamp: datetime
    description: str
    confidence: float
    before_shape: str
    after_shape: str
    significance: float

class ShapeChangeResponse(BaseModel):
    norad_id: int
    detected_changes: List[ShapeChangeDetection]
    summary: str
    metadata: Optional[Dict] = None 