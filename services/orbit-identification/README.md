# Orbit Family Identification API

This microservice identifies orbit families for space objects based on classical orbital elements. It assigns standardized tags like LEO, GEO, POLAR, etc., based on orbital parameters.

## Overview

The API analyzes orbital elements and assigns orbit family tags based on defined criteria:
- Altitude-based classifications (LEO, MEO, GEO, etc.)
- Direction-based classifications (PROGRADE, RETROGRADE)
- Eccentricity-based classifications (CIRCULAR, ELLIPTIC, HYPERBOLIC)
- Protected region classifications (IADC_LEO_PROTECTED_REGION, IADC_GEO_PROTECTED_REGION)
- Inclination-based classifications (EQUATORIAL, POLAR)
- Special orbit planes (ECLIPTIC, SSO)

## Features

- Single orbit identification
- Batch orbit identification
- Support for both semi-major axis and mean motion inputs
- Extensive coverage of orbit classification systems
- OpenAPI 3.1 compliant API

## API Endpoints

### Single Orbit Identification
- **Endpoint**: `POST /identify`
- **Description**: Identifies orbit families for a single set of orbital elements
- **Input**: MPE (Minimum Propagatable Element Set) object
- **Output**: List of applicable orbit tags

### Batch Orbit Identification
- **Endpoint**: `POST /identify-batch`
- **Description**: Identifies orbit families for multiple sets of orbital elements
- **Input**: Collection of MPE objects
- **Output**: Collection of orbit tag lists

## Data Models

### Input Data (MPE - Minimum Propagatable Element Set)

The API accepts the following orbital elements:
- `ENTITY_ID`: Optional identifier
- `EPOCH`: Optional epoch timestamp
- `SEMI_MAJOR_AXIS`: Semi-major axis in kilometers
- `MEAN_MOTION`: Mean motion in revolutions per day (alternative to SEMI_MAJOR_AXIS)
- `ECCENTRICITY`: Orbital eccentricity
- `INCLINATION`: Orbital inclination in degrees
- `RA_OF_ASC_NODE`: Right ascension of the ascending node in degrees
- `ARG_OF_PERICENTER`: Argument of pericenter in degrees
- `MEAN_ANOMALY`: Mean anomaly in degrees
- `BSTAR`: Drag term (1/Earth radii)

Either `SEMI_MAJOR_AXIS` or `MEAN_MOTION` must be provided.

### Output Data (OrbitTagMessage)

The API returns a list of applicable orbit tags from:

**Altitude Classes**:
- `LEO`: Low Earth Orbit (80-2000 km altitude)
- `VLEO`: Very Low Earth Orbit (80-450 km altitude)
- `MEO`: Medium Earth Orbit (2000-35,786 km altitude)
- `GSO`: Geosynchronous Orbit (period of 1 sidereal day ±5 minutes)
- `NEAR_GSO`: Near Geosynchronous Orbit (period within ±1 hour of GSO)
- `GEO`: Geostationary Orbit (circular equatorial GSO)
- `NEAR_GEO`: Near Geostationary Orbit (close to GEO belt)
- `HIGH_EARTH_ORBIT`: Above GEO altitude
- `GEO_GRAVEYARD`: Geostationary Graveyard Orbit (300-5000 km above GEO)

**Direction Classes**:
- `PROGRADE`: Prograde Orbit (inclination < 90°)
- `RETROGRADE`: Retrograde Orbit (inclination > 90°)

**Eccentricity Classes**:
- `CIRCULAR`: Circular Orbit (eccentricity ≤ 0.02)
- `NEAR_CIRCULAR`: Near Circular Orbit (eccentricity ≤ 0.1)
- `ELLIPTIC`: Elliptic Orbit (eccentricity < 1.0)
- `PARABOLIC`: Parabolic Orbit (eccentricity ≈ 1.0)
- `HYPERBOLIC`: Hyperbolic Orbit (eccentricity > 1.0)
- `HIGHLY_ELLIPTICAL_ORBIT`: Highly Elliptical Orbit (perigee in LEO, apogee near GSO or higher)

**IADC Protected Regions**:
- `IADC_LEO_PROTECTED_REGION`: IADC Low Earth Orbit Protected Region
- `IADC_LEO_PROTECTED_REGION_CROSSING`: Crosses IADC LEO Protected Region
- `IADC_GEO_PROTECTED_REGION`: IADC Geostationary Orbit Protected Region
- `IADC_GEO_PROTECTED_REGION_CROSSING`: Crosses IADC GEO Protected Region

**Inclination Classes**:
- `EQUATORIAL`: Equatorial Orbit (inclination within 0.1° of 0° or 180°)
- `NEAR_EQUATORIAL`: Near Equatorial Orbit (inclination within 5° of 0° or 180°)
- `POLAR`: Polar Orbit (inclination between 60° and 120°)

**Orbit Planes**:
- `ECLIPTIC`: Ecliptic Orbit (within 0.1° of the Ecliptic plane)
- `NEAR_ECLIPTIC`: Near Ecliptic Orbit (within 5° of the Ecliptic plane)
- `SSO`: Sun Synchronous Orbit (precesses at the same rate as Earth's orbit around the Sun)

## Examples

### Single Orbit Identification

#### Request:

```json
{
  "SEMI_MAJOR_AXIS": 7000,
  "ECCENTRICITY": 0.001,
  "INCLINATION": 98
}
```

#### Response:

```json
{
  "TAGS": [
    "LEO",
    "PROGRADE",
    "CIRCULAR",
    "NEAR_CIRCULAR",
    "ELLIPTIC",
    "IADC_LEO_PROTECTED_REGION",
    "IADC_LEO_PROTECTED_REGION_CROSSING",
    "POLAR",
    "SSO"
  ]
}
```

### Batch Orbit Identification

#### Request:

```json
{
  "REF_FRAME": "GCRF",
  "TIME_SYSTEM": "UTC",
  "MEAN_ELEMENT_THEORY": "SGP4",
  "RECORDS": [
    {
      "SEMI_MAJOR_AXIS": 7000,
      "ECCENTRICITY": 0.001,
      "INCLINATION": 98
    },
    {
      "SEMI_MAJOR_AXIS": 42164,
      "ECCENTRICITY": 0.0001,
      "INCLINATION": 0.05
    }
  ]
}
```

#### Response:

```json
{
  "RECORDS": [
    {
      "TAGS": [
        "LEO",
        "PROGRADE",
        "CIRCULAR",
        "NEAR_CIRCULAR",
        "ELLIPTIC",
        "IADC_LEO_PROTECTED_REGION",
        "IADC_LEO_PROTECTED_REGION_CROSSING",
        "POLAR",
        "SSO"
      ]
    },
    {
      "TAGS": [
        "GSO",
        "GEO",
        "NEAR_GSO",
        "NEAR_GEO",
        "PROGRADE",
        "CIRCULAR",
        "NEAR_CIRCULAR",
        "ELLIPTIC",
        "EQUATORIAL",
        "NEAR_EQUATORIAL"
      ]
    }
  ]
}
```

## Setup and Deployment

### Requirements

- Python 3.7+
- FastAPI
- Uvicorn

### Installation

1. Install required packages:
   ```
   pip install fastapi uvicorn
   ```

2. Run the service:
   ```
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

### API Documentation

When the service is running, you can access:
- Swagger UI: `/services/orbit-identification/docs`
- ReDoc: `/services/orbit-identification/redoc`
- OpenAPI Schema: `/services/orbit-identification/openapi.json`

## Implementation Notes

The orbit classification follows standard definitions from:

- Inter-Agency Space Debris Coordination Committee (IADC)
- NASA Orbital Debris Program Office
- International Telecommunications Union (ITU)
- Commercial Space Transportation Advisory Committee (COMSTAC)

The implementation uses accurate orbital mechanics formulas for calculating:
- Perigee and apogee altitudes
- Orbital periods
- Earth-relative positions

For accurate orbit family identification, provide as many orbital elements as possible. The minimum required elements are either `SEMI_MAJOR_AXIS` or `MEAN_MOTION`, along with `ECCENTRICITY` and `INCLINATION` for most classifications. 