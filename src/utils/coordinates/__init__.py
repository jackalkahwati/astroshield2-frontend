"""
AstroShield Coordinate Transformations Package

This package provides utilities for coordinate transformations commonly used in space applications.
"""

from .transformations import (
    # Core transformation functions
    eci_to_ecef,
    ecef_to_eci,
    ecef_to_lla,
    lla_to_ecef,
    eci_to_lla,
    lla_to_eci,
    
    # Convenience functions
    deg_to_rad,
    rad_to_deg,
    dict_eci_to_lla,
    dict_lla_to_eci,
    
    # Constants
    WGS84_A,
    WGS84_B,
    WGS84_F,
    WGS84_E,
    EARTH_ROTATION_RATE
) 