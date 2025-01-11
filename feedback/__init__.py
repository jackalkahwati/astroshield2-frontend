from .strategy_evaluator import (
    StrategyEvaluator,
    StrategyEffectiveness,
    TelemetryData
)
from .telemetry_processor import (
    TelemetryProcessor,
    RawTelemetry,
    ProcessedTelemetry
)
from .adaptation_engine import (
    AdaptationEngine,
    AdaptationRule,
    AdaptationResult
)

__all__ = [
    # Strategy Evaluation
    'StrategyEvaluator',
    'StrategyEffectiveness',
    'TelemetryData',
    
    # Telemetry Processing
    'TelemetryProcessor',
    'RawTelemetry',
    'ProcessedTelemetry',
    
    # Adaptation Engine
    'AdaptationEngine',
    'AdaptationRule',
    'AdaptationResult'
]

# Module version
__version__ = '1.0.0'

# Module metadata
__author__ = 'AstroShield Development Team'
__description__ = '''
Feedback Layer for AstroShield CCDM System
Provides real-time strategy evaluation, telemetry processing,
and adaptive response capabilities for enhanced spacecraft protection.
'''
