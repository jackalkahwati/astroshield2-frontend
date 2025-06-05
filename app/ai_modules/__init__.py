"""
AstroShield AI Modules Package

This package contains the AI-powered components for space domain awareness including:
- Intent classification for maneuver analysis
- Hostility scoring for threat assessment
- Event detection and pattern recognition
- Observation recommendation engine
- Sensor fusion interface
- Feedback and learning systems

All modules are designed to integrate with the existing Kafka event processing pipeline
and provide real-time analysis capabilities for space situational awareness.
"""

from .intent_classifier import IntentClassifier
from .hostility_scorer import HostilityScorer
from .event_detector import EventDetector
from .observation_recommender import ObservationRecommender
from .sensor_fusion import SensorFusionInterface
from .feedback_engine import FeedbackEngine
from .kafka_adapter import KafkaAdapter

__all__ = [
    "IntentClassifier",
    "HostilityScorer", 
    "EventDetector",
    "ObservationRecommender",
    "SensorFusionInterface",
    "FeedbackEngine",
    "KafkaAdapter"
] 