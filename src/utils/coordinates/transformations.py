"""
Coordinate Transformation Utilities

This module provides comprehensive coordinate transformation utilities for space applications,
including conversions between:
- Earth-Centered Inertial (ECI)
- Earth-Centered Earth-Fixed (ECEF)
- Latitude, Longitude, Altitude (LLA)
- Geodetic and Geocentric coordinates

All calculations use WGS-84 ellipsoid model for Earth.
"""

import numpy as np
from datetime import datetime
from typing import Tuple, Union, Dict

# WGS-84 Earth Constants
WGS84_A = 6378137.0  # Semi-major axis [m]
WGS84_F = 1/298.257223563  # Flattening
WGS84_B = WGS84_A * (1 - WGS84_F)  # Semi-minor axis [m]
WGS84_E = np.sqrt(2*WGS84_F - WGS84_F**2)  # Eccentricity
EARTH_ROTATION_RATE = 7.2921150e-5  # [rad/s]

def eci_to_ecef(r_eci: np.ndarray, v_eci: np.ndarray, epoch: datetime) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert position and velocity from ECI to ECEF coordinates.
    
    Args:
        r_eci: Position vector in ECI [m] (x, y, z)
        v_eci: Velocity vector in ECI [m/s] (vx, vy, vz)
        epoch: Time of the state vector
        
    Returns:
        Tuple of (position_ecef, velocity_ecef)
    """
    # Calculate GMST (Greenwich Mean Sidereal Time)
    jd = _datetime_to_jd(epoch)
    gmst = _calculate_gmst(jd)
    
    # Create rotation matrix
    R = _rotation_matrix_z(gmst)
    R_dot = _rotation_matrix_z_dot(gmst)
    
    # Transform position
    r_ecef = R @ r_eci
    
    # Transform velocity
    v_ecef = R @ v_eci + R_dot @ r_eci
    
    return r_ecef, v_ecef

def ecef_to_eci(r_ecef: np.ndarray, v_ecef: np.ndarray, epoch: datetime) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert position and velocity from ECEF to ECI coordinates.
    
    Args:
        r_ecef: Position vector in ECEF [m] (x, y, z)
        v_ecef: Velocity vector in ECEF [m/s] (vx, vy, vz)
        epoch: Time of the state vector
        
    Returns:
        Tuple of (position_eci, velocity_eci)
    """
    # Calculate GMST
    jd = _datetime_to_jd(epoch)
    gmst = _calculate_gmst(jd)
    
    # Create rotation matrices
    R = _rotation_matrix_z(gmst)
    R_dot = _rotation_matrix_z_dot(gmst)
    
    # Transform position
    r_eci = R.T @ r_ecef
    
    # Transform velocity
    v_eci = R.T @ v_ecef - R.T @ R_dot @ r_eci
    
    return r_eci, v_eci

def ecef_to_lla(r_ecef: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert ECEF coordinates to geodetic coordinates (LLA).
    
    Args:
        r_ecef: Position vector in ECEF [m] (x, y, z)
        
    Returns:
        Tuple of (latitude [rad], longitude [rad], altitude [m])
    """
    x, y, z = r_ecef
    
    # Calculate longitude
    lon = np.arctan2(y, x)
    
    # Calculate distance from Z axis
    p = np.sqrt(x**2 + y**2)
    
    # Initial estimate of latitude
    lat = np.arctan2(z, p * (1 - WGS84_E**2))
    
    # Iterative solution for latitude and altitude
    for _ in range(10):
        sin_lat = np.sin(lat)
        N = WGS84_A / np.sqrt(1 - WGS84_E**2 * sin_lat**2)
        alt = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1 - WGS84_E**2 * N/(N + alt)))
    
    return lat, lon, alt

def lla_to_ecef(lat: float, lon: float, alt: float) -> np.ndarray:
    """
    Convert geodetic coordinates (LLA) to ECEF coordinates.
    
    Args:
        lat: Latitude [rad]
        lon: Longitude [rad]
        alt: Altitude above WGS-84 ellipsoid [m]
        
    Returns:
        Position vector in ECEF [m] (x, y, z)
    """
    # Calculate Earth radius of curvature
    sin_lat = np.sin(lat)
    N = WGS84_A / np.sqrt(1 - WGS84_E**2 * sin_lat**2)
    
    # Calculate ECEF coordinates
    x = (N + alt) * np.cos(lat) * np.cos(lon)
    y = (N + alt) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - WGS84_E**2) + alt) * sin_lat
    
    return np.array([x, y, z])

def eci_to_lla(r_eci: np.ndarray, epoch: datetime) -> Tuple[float, float, float]:
    """
    Convert ECI coordinates to geodetic coordinates (LLA).
    
    Args:
        r_eci: Position vector in ECI [m] (x, y, z)
        epoch: Time of the state vector
        
    Returns:
        Tuple of (latitude [rad], longitude [rad], altitude [m])
    """
    r_ecef, _ = eci_to_ecef(r_eci, np.zeros(3), epoch)
    return ecef_to_lla(r_ecef)

def lla_to_eci(lat: float, lon: float, alt: float, epoch: datetime) -> np.ndarray:
    """
    Convert geodetic coordinates (LLA) to ECI coordinates.
    
    Args:
        lat: Latitude [rad]
        lon: Longitude [rad]
        alt: Altitude above WGS-84 ellipsoid [m]
        epoch: Time of the conversion
        
    Returns:
        Position vector in ECI [m] (x, y, z)
    """
    r_ecef = lla_to_ecef(lat, lon, alt)
    r_eci, _ = ecef_to_eci(r_ecef, np.zeros(3), epoch)
    return r_eci

# Helper functions

def _datetime_to_jd(dt: datetime) -> float:
    """Convert datetime to Julian Date."""
    year = dt.year
    month = dt.month
    day = dt.day + dt.hour/24.0 + dt.minute/1440.0 + dt.second/86400.0
    
    if month <= 2:
        year = year - 1
        month = month + 12
    
    A = int(year/100)
    B = 2 - A + int(A/4)
    
    jd = (int(365.25*(year + 4716)) + int(30.6001*(month + 1)) + 
          day + B - 1524.5)
    
    return jd

def _calculate_gmst(jd: float) -> float:
    """Calculate Greenwich Mean Sidereal Time in radians."""
    T = (jd - 2451545.0) / 36525.0
    gmst = 280.46061837 + 360.98564736629 * (jd - 2451545.0) + \
           0.000387933 * T**2 - T**3 / 38710000.0
    
    return np.radians(gmst % 360)

def _rotation_matrix_z(angle: float) -> np.ndarray:
    """Create rotation matrix around Z axis."""
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s, 0],
                    [s, c, 0],
                    [0, 0, 1]])

def _rotation_matrix_z_dot(angle: float) -> np.ndarray:
    """Create derivative of rotation matrix around Z axis."""
    c = np.cos(angle)
    s = np.sin(angle)
    return EARTH_ROTATION_RATE * np.array([[-s, -c, 0],
                                         [c, -s, 0],
                                         [0, 0, 0]])

# Convenience functions for degree/radian conversions
def deg_to_rad(deg: float) -> float:
    """Convert degrees to radians."""
    return np.radians(deg)

def rad_to_deg(rad: float) -> float:
    """Convert radians to degrees."""
    return np.degrees(rad)

# Dictionary-based interfaces for easier integration
def dict_eci_to_lla(state: Dict[str, float], epoch: datetime) -> Dict[str, float]:
    """
    Convert ECI state dictionary to LLA dictionary.
    
    Args:
        state: Dictionary with 'x', 'y', 'z' keys in meters
        epoch: Time of the state vector
        
    Returns:
        Dictionary with 'lat' (deg), 'lon' (deg), 'alt' (m) keys
    """
    r_eci = np.array([state['x'], state['y'], state['z']])
    lat, lon, alt = eci_to_lla(r_eci, epoch)
    return {
        'lat': rad_to_deg(lat),
        'lon': rad_to_deg(lon),
        'alt': alt
    }

def dict_lla_to_eci(state: Dict[str, float], epoch: datetime) -> Dict[str, float]:
    """
    Convert LLA state dictionary to ECI dictionary.
    
    Args:
        state: Dictionary with 'lat' (deg), 'lon' (deg), 'alt' (m) keys
        epoch: Time of the conversion
        
    Returns:
        Dictionary with 'x', 'y', 'z' keys in meters
    """
    lat = deg_to_rad(state['lat'])
    lon = deg_to_rad(state['lon'])
    alt = state['alt']
    r_eci = lla_to_eci(lat, lon, alt, epoch)
    return {
        'x': r_eci[0],
        'y': r_eci[1],
        'z': r_eci[2]
    } 