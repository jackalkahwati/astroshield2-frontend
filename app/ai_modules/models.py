"""
Data models and schemas for AstroShield AI modules.

These models define the structured data flow between AI components and integrate
with the existing Kafka event processing pipeline.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator
import uuid


class IntentClass(str, Enum):
    """Classification of satellite maneuver intent."""
    INSPECTION = "inspection"
    SHADOWING = "shadowing" 
    EVASION = "evasion"
    COLLISION_COURSE = "collision_course"
    IMAGING_PASS = "imaging_pass"
    STATION_KEEPING = "station_keeping"
    DEBRIS_AVOIDANCE = "debris_avoidance"
    RENDEZVOUS = "rendezvous"
    ROUTINE_MAINTENANCE = "routine_maintenance"
    UNKNOWN = "unknown"


class ThreatLevel(str, Enum):
    """Threat level classification."""
    BENIGN = "benign"
    SUSPECT = "suspect"
    HOSTILE = "hostile"
    CRITICAL = "critical"


class EventClass(str, Enum):
    """Event classification categories."""
    ROUTINE = "routine"
    ANOMALOUS = "anomalous"
    ENGINEERED = "engineered"
    THREAT_LIKELY = "threat_likely"


class ManeuverType(str, Enum):
    """Types of orbital maneuvers."""
    PROGRADE = "prograde"
    RETROGRADE = "retrograde"
    NORMAL = "normal"
    ANTI_NORMAL = "anti_normal"
    RADIAL = "radial"
    ANTI_RADIAL = "anti_radial"
    COMBINED = "combined"
    UNKNOWN = "unknown"


class ActorType(str, Enum):
    """Actor classification for satellites/objects."""
    US = "us"
    ALLY = "ally"
    ADVERSARY = "adversary"
    COMMERCIAL = "commercial"
    UNKNOWN = "unknown"


class SensorModality(str, Enum):
    """Sensor types for observation recommendations."""
    OPTICAL = "optical"
    INFRARED = "infrared"
    RADAR = "radar"
    RF = "rf"
    MULTI_SPECTRAL = "multi_spectral"


class ObservationPriority(str, Enum):
    """Priority levels for observation requests."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Base event models
class BaseEvent(BaseModel):
    """Base event model with common fields."""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source_data_lineage: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class ManeuverEvent(BaseEvent):
    """Detected satellite maneuver event."""
    sat_pair_id: str = Field(..., description="Satellite pair identifier")
    primary_norad_id: str = Field(..., description="Primary satellite NORAD ID")
    secondary_norad_id: Optional[str] = Field(None, description="Secondary satellite NORAD ID if applicable")
    maneuver_type: ManeuverType
    delta_v: float = Field(..., description="Delta-V magnitude in m/s")
    burn_duration: Optional[float] = Field(None, description="Burn duration in seconds")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    orbital_elements_before: Dict[str, float] = Field(..., description="Orbital elements before maneuver")
    orbital_elements_after: Dict[str, float] = Field(..., description="Orbital elements after maneuver")


class ProximityEvent(BaseEvent):
    """Proximity operation event between satellites."""
    sat_pair_id: str = Field(..., description="Satellite pair identifier")
    primary_norad_id: str
    secondary_norad_id: str
    closest_approach_time: datetime
    minimum_distance: float = Field(..., description="Minimum distance in meters")
    relative_velocity: float = Field(..., description="Relative velocity in m/s")
    duration_minutes: float = Field(..., description="Duration of proximity operation")
    approach_geometry: Dict[str, float] = Field(default_factory=dict)


# AI Analysis Results
class IntentClassificationResult(BaseModel):
    """Result of intent classification analysis."""
    event_id: str
    timestamp: datetime
    sat_pair_id: str
    intent_class: IntentClass
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    maneuver_type: ManeuverType
    reasoning: List[str] = Field(default_factory=list, description="Explanation of classification")
    model_version: str = Field(..., description="Model version used for classification")
    source_data_lineage: Dict[str, Any]
    
    @validator('confidence_score')
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence score must be between 0 and 1')
        return v


class HostilityAssessment(BaseModel):
    """Result of hostility scoring analysis."""
    event_id: str
    timestamp: datetime
    target_norad_id: str
    actor_type: ActorType
    threat_level: ThreatLevel
    hostility_score: float = Field(..., ge=0.0, le=1.0)
    contributing_factors: Dict[str, float] = Field(default_factory=dict)
    pattern_analysis: Dict[str, Any] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_version: str


class ObservationRecommendation(BaseModel):
    """Recommendation for sensor observation."""
    event_id: str
    timestamp: datetime
    target_norad_id: str
    recommended_sensors: List[str]
    sensor_modality: SensorModality
    observation_window_start: datetime
    observation_window_end: datetime
    priority: ObservationPriority
    predicted_phenomena: List[str] = Field(default_factory=list)
    success_probability: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    cue_parameters: Dict[str, Any] = Field(default_factory=dict)


class EventDetectionResult(BaseModel):
    """Result of event detection analysis."""
    event_id: str
    timestamp: datetime
    detection_type: str
    event_class: EventClass
    confidence: float = Field(..., ge=0.0, le=1.0)
    detected_patterns: List[str] = Field(default_factory=list)
    anomaly_score: float = Field(..., ge=0.0, le=1.0)
    requires_analysis: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FeedbackRecord(BaseModel):
    """Feedback record for model training and improvement."""
    feedback_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    original_event_id: str
    prediction_id: str
    ground_truth: Dict[str, Any]
    operator_action: Optional[str] = None
    outcome: Optional[str] = None
    correction_needed: bool = False
    feedback_type: str = Field(..., description="Type of feedback (classification, scoring, recommendation)")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    analyst_notes: Optional[str] = None


class ModelPerformanceMetrics(BaseModel):
    """Performance metrics for AI models."""
    model_name: str
    model_version: str
    timestamp: datetime
    accuracy: float = Field(..., ge=0.0, le=1.0)
    precision: float = Field(..., ge=0.0, le=1.0)
    recall: float = Field(..., ge=0.0, le=1.0)
    f1_score: float = Field(..., ge=0.0, le=1.0)
    confusion_matrix: Optional[List[List[int]]] = None
    sample_size: int
    evaluation_period_days: int


# Kafka message wrappers
class AIAnalysisMessage(BaseModel):
    """Wrapper for AI analysis results in Kafka messages."""
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    message_type: str
    analysis_type: str  # intent_classification, hostility_assessment, etc.
    payload: Dict[str, Any]
    correlation_id: Optional[str] = None
    
    class Config:
        use_enum_values = True


# Configuration models
class ModelConfig(BaseModel):
    """Configuration for AI models."""
    model_name: str
    model_version: str
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    batch_size: int = Field(default=32, gt=0)
    max_processing_time_seconds: int = Field(default=30, gt=0)
    enable_uncertainty_quantification: bool = True
    preprocessing_params: Dict[str, Any] = Field(default_factory=dict)
    postprocessing_params: Dict[str, Any] = Field(default_factory=dict)


class PipelineConfig(BaseModel):
    """Configuration for the AI processing pipeline."""
    enable_intent_classification: bool = True
    enable_hostility_scoring: bool = True
    enable_observation_recommendation: bool = True
    enable_feedback_loop: bool = True
    parallel_processing: bool = True
    max_concurrent_analyses: int = Field(default=10, gt=0)
    retry_failed_analyses: bool = True
    max_retries: int = Field(default=3, ge=0)
    models: Dict[str, ModelConfig] = Field(default_factory=dict) 