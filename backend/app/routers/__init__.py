"""AstroShield API Routers."""

# Import all routers for FastAPI to find them
from . import health, analytics, maneuvers, satellites, advanced, dashboard, ccdm, trajectory, comparison, events

__all__ = [
    'health',
    'analytics',
    'maneuvers',
    'satellites',
    'advanced',
    'dashboard',
    'ccdm',
    'trajectory',
    'comparison',
    'events'
] 