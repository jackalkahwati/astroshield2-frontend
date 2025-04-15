# AstroShield Coordinate Transformations

This module provides a comprehensive set of utilities for coordinate transformations commonly used in space applications. It implements accurate conversions between different coordinate systems while taking into account Earth's shape using the WGS-84 ellipsoid model.

## Supported Coordinate Systems

- **Earth-Centered Inertial (ECI)**: An inertial reference frame where the origin is at the Earth's center of mass, the Z-axis extends through the true north pole, and the X and Y axes lie in the equatorial plane (with X pointing towards the vernal equinox).

- **Earth-Centered Earth-Fixed (ECEF)**: Similar to ECI, but rotates with the Earth. The X-axis intersects the sphere of the Earth at 0° latitude and 0° longitude, Z-axis extends through the true north pole, and Y-axis completes the right-handed coordinate system.

- **Latitude, Longitude, Altitude (LLA)**: Geodetic coordinates using the WGS-84 ellipsoid model, where latitude and longitude are angular measurements from the equator and prime meridian respectively, and altitude is the height above the ellipsoid.

## Usage Examples

```python
from datetime import datetime
from src.utils.coordinates import (
    eci_to_ecef,
    ecef_to_lla,
    dict_eci_to_lla
)

# Example 1: Converting ECI position and velocity to ECEF
r_eci = np.array([6378137.0, 0.0, 0.0])  # Position vector in ECI
v_eci = np.array([0.0, 7.7, 0.0])        # Velocity vector in ECI
epoch = datetime.utcnow()

r_ecef, v_ecef = eci_to_ecef(r_eci, v_eci, epoch)

# Example 2: Converting ECEF to LLA
lat, lon, alt = ecef_to_lla(r_ecef)

# Example 3: Using dictionary interface for simpler integration
eci_state = {
    'x': 6378137.0,
    'y': 0.0,
    'z': 0.0
}

lla_state = dict_eci_to_lla(eci_state, epoch)
print(f"Latitude: {lla_state['lat']} deg")
print(f"Longitude: {lla_state['lon']} deg")
print(f"Altitude: {lla_state['alt']} m")
```

## Features

- Accurate coordinate transformations using WGS-84 ellipsoid model
- Support for both position and velocity transformations
- Proper handling of Earth rotation effects
- Convenience functions for degree/radian conversions
- Dictionary-based interfaces for easier integration
- Comprehensive type hints for better IDE support
- Well-documented functions with clear parameter descriptions

## Constants

The module provides several important constants based on the WGS-84 model:

- `WGS84_A`: Earth's semi-major axis (equatorial radius) in meters
- `WGS84_B`: Earth's semi-minor axis (polar radius) in meters
- `WGS84_F`: Earth's flattening factor
- `WGS84_E`: Earth's eccentricity
- `EARTH_ROTATION_RATE`: Earth's rotation rate in radians per second

## Implementation Details

- All linear measurements are in meters
- Angular measurements are in radians (unless using the dictionary interface, which uses degrees)
- Velocity measurements are in meters per second
- The implementation uses numpy for efficient matrix operations
- Iterative algorithm for accurate geodetic latitude calculation
- Proper handling of Greenwich Mean Sidereal Time for ECI-ECEF conversions 