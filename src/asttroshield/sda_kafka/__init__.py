"""
AstroShield SDA Kafka Integration Module

This module provides integration with the SDA (Space Development Agency) Kafka message bus
for real-time event-driven architecture supporting orbital intelligence capabilities.

Key Features:
- AWS MSK connectivity with proper authentication
- SDA topic structure and message schemas
- Test environment support
- Message size optimization
- Integration with AstroShield TLE chat and orbital intelligence
"""

from .sda_message_bus import (
    SDAKafkaCredentials,
    SDAMessageSchema,
    SDATopicManager,
    SDAKafkaClient,
    AstroShieldSDAIntegration,
    SDASubsystem,
    MessagePriority
)

# Import SDA schemas
try:
    from .sda_schemas import (
        SDAManeuverDetected,
        SDALaunchDetected,
        SDATLEUpdate,
        SDALaunchIntentAssessment,
        SDAPezWezPrediction,
        SDAASATAssessment,
        SDASchemaFactory,
        validate_sda_schema
    )
    _SCHEMAS_AVAILABLE = True
except ImportError:
    _SCHEMAS_AVAILABLE = False

__all__ = [
    'SDAKafkaCredentials',
    'SDAMessageSchema', 
    'SDATopicManager',
    'SDAKafkaClient',
    'AstroShieldSDAIntegration',
    'SDASubsystem',
    'MessagePriority'
]

# Add schemas to exports if available
if _SCHEMAS_AVAILABLE:
    __all__.extend([
        'SDAManeuverDetected',
        'SDALaunchDetected', 
        'SDATLEUpdate',
        'SDALaunchIntentAssessment',
        'SDAPezWezPrediction',
        'SDAASATAssessment',
        'SDASchemaFactory',
        'validate_sda_schema'
    ])

__version__ = '1.0.0' 