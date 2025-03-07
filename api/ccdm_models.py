from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

class ObservationData(BaseModel):
    """Model for observation data"""
    timestamp: datetime
    sensor_id: str
    data_type: str
    measurements: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

class CCDMAssessment(BaseModel):
    """Model for CCDM assessment results"""
    object_id: str
    assessment_type: str
    confidence_level: float
    timestamp: datetime
    details: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

class HistoricalAnalysis(BaseModel):
    """Model for historical analysis results"""
    object_id: str
    time_range: Dict[str, datetime]
    patterns: List[Dict[str, Any]]
    trend_analysis: Dict[str, Any]
    confidence_levels: Dict[str, float]

class CorrelationResult(BaseModel):
    """Model for correlation analysis results"""
    primary_object_id: str
    related_objects: List[str]
    correlation_type: str
    confidence_level: float
    evidence: Dict[str, Any]

class ObservationRecommendation(BaseModel):
    """Model for observation recommendations"""
    object_id: str
    recommended_times: List[datetime]
    recommended_sensors: List[str]
    parameters: Dict[str, Any]
    priority_level: float = Field(ge=0.0, le=1.0)

class AnomalyDetection(BaseModel):
    """Model for anomaly detection results"""
    object_id: str
    anomaly_type: str
    confidence_level: float
    detection_time: datetime
    evidence: Dict[str, Any]
    recommended_actions: Optional[List[str]] = None

class BehaviorClassification(BaseModel):
    """Model for behavior classification results"""
    object_id: str
    behavior_type: str
    confidence_level: float
    classification_time: datetime
    supporting_evidence: Dict[str, Any]
    related_behaviors: Optional[List[str]] = None

class StateVector(BaseModel):
    """Model for object state vector"""
    position: List[float]
    velocity: List[float]
    attitude: Optional[List[float]] = None
    angular_velocity: Optional[List[float]] = None
    timestamp: datetime

class FutureStatePredict(BaseModel):
    """Model for future state predictions"""
    object_id: str
    current_state: StateVector
    predicted_state: StateVector
    prediction_window: float  # in seconds
    confidence_level: float
    uncertainty_metrics: Dict[str, float]

class CCDMReport(BaseModel):
    """Model for comprehensive CCDM reports"""
    object_id: str
    report_timestamp: datetime
    assessment_summary: CCDMAssessment
    historical_analysis: Optional[HistoricalAnalysis] = None
    anomalies: List[AnomalyDetection]
    behavior_classification: BehaviorClassification
    future_predictions: Optional[FutureStatePredict] = None
    recommendations: List[str]
    confidence_metrics: Dict[str, float]
    metadata: Optional[Dict[str, Any]] = None

class RFEmissionAnalysis(BaseModel):
    """Model for RF emission analysis"""
    object_id: str
    frequency_bands: List[Dict[str, float]]
    emission_patterns: Dict[str, Any]
    anomalies: List[Dict[str, Any]]
    confidence_level: float

class OpticalSignatureAnalysis(BaseModel):
    """Model for optical signature analysis"""
    object_id: str
    brightness_metrics: Dict[str, float]
    spectral_analysis: Dict[str, Any]
    temporal_variations: List[Dict[str, Any]]
    confidence_level: float

class RadarCrossSectionAnalysis(BaseModel):
    """Model for radar cross section analysis"""
    object_id: str
    rcs_measurements: List[float]
    aspect_angles: List[float]
    temporal_changes: Dict[str, Any]
    confidence_level: float

class ProximityOperation(BaseModel):
    """Model for proximity operation detection"""
    primary_object_id: str
    secondary_object_id: str
    minimum_distance: float
    relative_velocity: List[float]
    confidence_level: float
    operation_type: str
    detection_time: datetime

class PropulsiveCapability(BaseModel):
    """Model for propulsive capability assessment"""
    object_id: str
    maneuver_history: List[Dict[str, Any]]
    estimated_capabilities: Dict[str, float]
    confidence_level: float
    assessment_time: datetime

class ShapeChange(BaseModel):
    """Model for shape change detection"""
    object_id: str
    original_dimensions: Dict[str, float]
    current_dimensions: Dict[str, float]
    change_metrics: Dict[str, float]
    confidence_level: float
    detection_time: datetime

class ThermalSignature(BaseModel):
    """Model for thermal signature analysis"""
    object_id: str
    temperature_profile: Dict[str, float]
    temporal_variations: List[Dict[str, Any]]
    anomalies: List[Dict[str, Any]]
    confidence_level: float

class CCDMViolator(BaseModel):
    """Model for CCDM violator list entries"""
    time_utc: datetime
    object_id: str
    common_name: str
    country: str
    object_type: str
    priority: int
    violation_type: str = Field(description="Type of CCDM violation")
    violation_details: Optional[Dict[str, Any]] = None
    reported: bool = False
    
class CCDMViolatorList(BaseModel):
    """Model for CCDM violator list"""
    violators: List[CCDMViolator]
    last_updated: datetime
    total_count: int
    critical_count: int = Field(description="Count of priority 5 violations")
    high_count: int = Field(description="Count of priority 6 violations")
    medium_count: int = Field(description="Count of priority 7 violations")
    low_count: int = Field(description="Count of priority 8+ violations")

class AntiCCDMIndicator(BaseModel):
    """Model for Anti-CCDM indicators that help distinguish between actual maneuvers and environmental effects"""
    object_id: str
    timestamp: datetime
    indicator_type: str = Field(description="Type of anti-CCDM indicator (e.g., 'drag', 'solar_radiation')")
    confidence_level: float = Field(ge=0.0, le=1.0, description="Confidence level of this indicator")
    environmental_factor: str = Field(description="Environmental factor causing the behavior (e.g., 'atmospheric_drag', 'solar_activity')")
    expected_deviation: Dict[str, float] = Field(description="Expected deviation due to environmental factors")
    actual_deviation: Dict[str, float] = Field(description="Actual observed deviation")
    is_environmental: bool = Field(description="Whether the behavior is likely environmental rather than intentional")
    details: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class DragAnalysis(BaseModel):
    """Model for drag analysis results specifically for V-LEO objects"""
    object_id: str
    timestamp: datetime
    norad_id: Optional[str] = None
    tle_epoch: Optional[datetime] = None
    drag_coefficient: float
    atmospheric_density: float
    predicted_position_deviation: List[float]
    actual_position_deviation: List[float]
    is_anomalous: bool = Field(description="Whether the drag effect is outside expected parameters")
    confidence_level: float = Field(ge=0.0, le=1.0)
    forecast: Optional[Dict[str, Any]] = Field(description="Forecast for future drag effects (24h, 48h, 72h)")
    metadata: Optional[Dict[str, Any]] = None
