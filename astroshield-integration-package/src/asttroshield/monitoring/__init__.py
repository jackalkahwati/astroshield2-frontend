"""
Monitoring and observability utilities for AstroShield.

This package provides tools for collecting metrics, tracing, and other
observability features for AstroShield components.
"""

from .metrics import initialize_metrics, MetricsMiddleware
from .metrics import MESSAGE_COUNTER, PROCESSING_TIME, CONSUMER_LAG

__all__ = [
    'initialize_metrics', 
    'MetricsMiddleware',
    'MESSAGE_COUNTER',
    'PROCESSING_TIME',
    'CONSUMER_LAG'
] 