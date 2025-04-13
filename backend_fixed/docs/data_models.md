# AstroShield API Data Models Reference

This document provides a comprehensive reference for all data models used in the AstroShield API, including their properties, relationships, and examples.

## Core Models

### Satellite

Represents a satellite tracked by the AstroShield system.

#### Properties

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| id | string | No (auto-generated) | Unique identifier for the satellite |
| name | string | Yes | Name of the satellite |
| norad_id | string | Yes | NORAD Catalog Number (unique) |
| international_designator | string | Yes | International designator (YYYY-NNNP) |
| orbit_type | enum | Yes | Type of orbit (LEO, MEO, GEO, HEO) |
| launch_date | datetime | Yes | Launch date and time (ISO 8601) |
| operational_status | enum | Yes | Operational status (ACTIVE, INACTIVE, DECOMMISSIONED, UNKNOWN) |
| owner | string | Yes | Organization that owns the satellite |
| mission | string | No | Primary mission of the satellite |
| orbital_parameters | object | Yes | Current orbital parameters |
| created_at | datetime | No (auto-generated) | When the record was created |
| updated_at | datetime | No (auto-generated) | When the record was last updated |

#### Orbit Types

- `LEO`: Low Earth Orbit (altitude < 2,000 km)
- `MEO`: Medium Earth Orbit (altitude 2,000-35,786 km)
- `GEO`: Geostationary Orbit (altitude = 35,786 km)
- `HEO`: Highly Elliptical Orbit (typically with high eccentricity)

#### Operational Status

- `ACTIVE`: Currently operational
- `INACTIVE`: Temporarily non-operational
- `DECOMMISSIONED`: End of mission, no longer operational
- `UNKNOWN`: Status cannot be determined

#### Example

```json
{
  "id": "sat-001",
  "name": "AstroShield-1",
  "norad_id": "43657",
  "international_designator": "2018-099A",
  "orbit_type": "LEO",
  "launch_date": "2023-05-15T00:00:00Z",
  "operational_status": "ACTIVE",
  "owner": "AstroShield Inc.",
  "mission": "Space Domain Awareness",
  "orbital_parameters": {
    "semi_major_axis": 7000.0,
    "eccentricity": 0.0001,
    "inclination": 51.6,
    "raan": 235.7,
    "argument_of_perigee": 90.0,
    "mean_anomaly": 0.0,
    "epoch": "2023-05-15T12:00:00Z"
  },
  "created_at": "2023-05-14T10:30:00Z",
  "updated_at": "2023-05-14T10:30:00Z"
}
```

### Orbital Parameters

Represents the Keplerian orbital elements that define a satellite's orbit.

#### Properties

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| semi_major_axis | float | Yes | Semi-major axis in kilometers |
| eccentricity | float | Yes | Orbital eccentricity (0-1) |
| inclination | float | Yes | Inclination in degrees (0-180) |
| raan | float | Yes | Right Ascension of Ascending Node in degrees (0-360) |
| argument_of_perigee | float | Yes | Argument of perigee in degrees (0-360) |
| mean_anomaly | float | Yes | Mean anomaly in degrees (0-360) |
| epoch | datetime | Yes | Reference time for orbital elements (ISO 8601) |

#### Example

```json
{
  "semi_major_axis": 7000.0,
  "eccentricity": 0.0001,
  "inclination": 51.6,
  "raan": 235.7,
  "argument_of_perigee": 90.0,
  "mean_anomaly": 0.0,
  "epoch": "2023-05-15T12:00:00Z"
}
```

### Maneuver

Represents a planned or executed satellite maneuver.

#### Properties

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| id | string | No (auto-generated) | Unique identifier for the maneuver |
| satellite_id | string | Yes | ID of the satellite performing the maneuver |
| type | enum | Yes | Type of maneuver |
| status | enum | No (default: SCHEDULED) | Current status of the maneuver |
| scheduled_start_time | datetime | Yes | Planned start time (ISO 8601) |
| actual_start_time | datetime | No | Actual start time if executed |
| end_time | datetime | No | End time if completed |
| parameters | object | Yes | Maneuver-specific parameters |
| resources | object | No | Resource usage for the maneuver |
| created_by | string | No | User who created the maneuver |
| created_at | datetime | No (auto-generated) | When the record was created |
| updated_at | datetime | No (auto-generated) | When the record was last updated |

#### Maneuver Types

- `COLLISION_AVOIDANCE`: Maneuver to avoid collision with another object
- `STATION_KEEPING`: Maneuver to maintain desired orbital position
- `ORBIT_RAISING`: Maneuver to increase orbital altitude
- `ORBIT_LOWERING`: Maneuver to decrease orbital altitude
- `DEORBIT`: Maneuver to initiate controlled reentry
- `INCLINATION_CHANGE`: Maneuver to change orbital plane
- `PHASING`: Maneuver to adjust position within orbital plane

#### Maneuver Status

- `SCHEDULED`: Planned for future execution
- `IN_PROGRESS`: Currently being executed
- `COMPLETED`: Successfully completed
- `FAILED`: Execution attempted but failed
- `CANCELLED`: Cancelled before execution

#### Maneuver Parameters

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| delta_v | float | Yes | Change in velocity (km/s) |
| burn_duration | float | Yes | Duration of thruster burn (seconds) |
| direction | object | Yes | Direction vector for maneuver |
| target_orbit | object | No | Target orbital parameters (if applicable) |

#### Direction Vector

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| x | float | Yes | X component of direction vector |
| y | float | Yes | Y component of direction vector |
| z | float | Yes | Z component of direction vector |

#### Maneuver Resources

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| fuel_remaining | float | No | Fuel remaining after maneuver (kg) |
| power_available | float | No | Power available during maneuver (W) |
| thruster_status | string | No | Status of thrusters |

#### Example

```json
{
  "id": "mnv-001",
  "satellite_id": "sat-001",
  "type": "COLLISION_AVOIDANCE",
  "status": "SCHEDULED",
  "scheduled_start_time": "2023-06-15T20:00:00Z",
  "parameters": {
    "delta_v": 0.02,
    "burn_duration": 15.0,
    "direction": {
      "x": 0.1,
      "y": 0.0,
      "z": -0.1
    },
    "target_orbit": {
      "altitude": 500.2,
      "inclination": 45.0,
      "eccentricity": 0.001
    }
  },
  "resources": {
    "fuel_remaining": 24.5,
    "power_available": 1800.0,
    "thruster_status": "NOMINAL"
  },
  "created_by": "operator@astroshield.com",
  "created_at": "2023-06-10T15:30:00Z",
  "updated_at": "2023-06-10T15:30:00Z"
}
```

### Conjunction

Represents a predicted close approach between a satellite and another space object.

#### Properties

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| id | string | No (auto-generated) | Unique identifier for the conjunction |
| satellite_id | string | Yes | ID of the primary satellite |
| object_type | enum | Yes | Type of secondary object |
| object_id | string | Yes | ID of the secondary object |
| tca | datetime | Yes | Time of closest approach (ISO 8601) |
| miss_distance | float | Yes | Distance at closest approach (km) |
| probability_of_collision | float | Yes | Calculated probability (0-1) |
| relative_velocity | float | Yes | Relative velocity at TCA (km/s) |
| severity_level | enum | Yes | Assessed severity of conjunction |
| status | enum | Yes | Current handling status |
| detection_time | datetime | Yes | When the conjunction was first detected |
| data_source | string | No | Source of conjunction data |
| created_at | datetime | No (auto-generated) | When the record was created |
| updated_at | datetime | No (auto-generated) | When the record was last updated |

#### Object Types

- `SATELLITE`: Active satellite
- `DEBRIS`: Orbital debris
- `ROCKET_BODY`: Spent rocket stage
- `UNKNOWN`: Unclassified object

#### Severity Levels

- `CRITICAL`: High risk requiring immediate action
- `HIGH`: Significant risk requiring action
- `MEDIUM`: Moderate risk requiring monitoring
- `LOW`: Low risk, no action required

#### Status

- `DETECTED`: Initial detection, analysis pending
- `MONITORING`: Under observation with regular updates
- `MANEUVER_PLANNED`: Avoidance maneuver has been planned
- `MANEUVER_EXECUTED`: Avoidance maneuver has been executed
- `RESOLVED`: Conjunction no longer poses a risk
- `FALSE_ALARM`: Determined to be a false detection

#### Example

```json
{
  "id": "conj-001",
  "satellite_id": "sat-001",
  "object_type": "DEBRIS",
  "object_id": "1989-006C",
  "tca": "2023-06-20T08:15:30Z",
  "miss_distance": 0.42,
  "probability_of_collision": 0.00025,
  "relative_velocity": 14.2,
  "severity_level": "HIGH",
  "status": "MONITORING",
  "detection_time": "2023-06-17T14:22:10Z",
  "data_source": "USSF Space Surveillance Network",
  "created_at": "2023-06-17T14:25:00Z",
  "updated_at": "2023-06-17T18:10:00Z"
}
```

### User

Represents a user of the AstroShield system.

#### Properties

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| id | string | No (auto-generated) | Unique identifier for the user |
| username | string | Yes | Username (unique) |
| email | string | Yes | Email address (unique) |
| full_name | string | No | User's full name |
| organization | string | No | User's organization |
| role | enum | No | User's role in the system |
| is_active | boolean | No | Whether the user account is active |
| is_superuser | boolean | No | Whether the user has superuser privileges |
| created_at | datetime | No (auto-generated) | When the user was created |
| updated_at | datetime | No (auto-generated) | When the user was last updated |

#### Roles

- `ADMIN`: System administrator with full access
- `OPERATOR`: Satellite operator with maneuver capabilities
- `ANALYST`: Analyst with read-only access
- `VIEWER`: Basic viewer with limited access

#### Example

```json
{
  "id": "usr-001",
  "username": "jdoe",
  "email": "john.doe@example.com",
  "full_name": "John Doe",
  "organization": "AstroShield Inc.",
  "role": "OPERATOR",
  "is_active": true,
  "is_superuser": false,
  "created_at": "2023-04-01T09:00:00Z",
  "updated_at": "2023-04-01T09:00:00Z"
}
```

## Analytics Models

### Satellite Analytics

Represents analytics data for a specific satellite.

#### Properties

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| satellite_id | string | Yes | ID of the satellite |
| orbit_stability | object | No | Orbital stability metrics |
| maneuver_history | array | No | Summary of past maneuvers |
| conjunction_statistics | object | No | Statistics about conjunctions |
| resource_utilization | object | No | Resource usage statistics |
| operational_metrics | object | No | Operational health metrics |
| data_period | object | Yes | Time period for the analytics |

#### Orbit Stability

| Property | Type | Description |
|----------|------|-------------|
| mean_eccentricity_change | float | Average change in eccentricity over time |
| mean_inclination_drift | float | Average drift in inclination (degrees/day) |
| orbital_period_variation | float | Variation in orbital period (seconds) |
| position_uncertainty | float | Uncertainty in position (meters) |

#### Maneuver History Summary

| Property | Type | Description |
|----------|------|-------------|
| total_count | integer | Total number of maneuvers |
| by_type | object | Count of maneuvers by type |
| total_delta_v | float | Total delta-v used (km/s) |
| success_rate | float | Percentage of successful maneuvers |

#### Conjunction Statistics

| Property | Type | Description |
|----------|------|-------------|
| total_count | integer | Total number of conjunctions |
| by_severity | object | Count of conjunctions by severity level |
| closest_approach | float | Closest recorded approach (km) |
| mean_miss_distance | float | Average miss distance (km) |

#### Resource Utilization

| Property | Type | Description |
|----------|------|-------------|
| fuel_consumption_rate | float | Fuel usage rate (kg/year) |
| estimated_remaining_life | float | Estimated remaining lifetime (years) |
| power_efficiency | float | Power efficiency metric |

#### Data Period

| Property | Type | Description |
|----------|------|-------------|
| start_date | datetime | Start of the data period |
| end_date | datetime | End of the data period |
| duration_days | integer | Duration in days |

#### Example

```json
{
  "satellite_id": "sat-001",
  "orbit_stability": {
    "mean_eccentricity_change": 0.0000025,
    "mean_inclination_drift": 0.0015,
    "orbital_period_variation": 0.42,
    "position_uncertainty": 15.3
  },
  "maneuver_history": {
    "total_count": 12,
    "by_type": {
      "COLLISION_AVOIDANCE": 3,
      "STATION_KEEPING": 9
    },
    "total_delta_v": 0.15,
    "success_rate": 100.0
  },
  "conjunction_statistics": {
    "total_count": 24,
    "by_severity": {
      "HIGH": 2,
      "MEDIUM": 8,
      "LOW": 14
    },
    "closest_approach": 0.42,
    "mean_miss_distance": 8.7
  },
  "resource_utilization": {
    "fuel_consumption_rate": 1.2,
    "estimated_remaining_life": 5.3,
    "power_efficiency": 0.92
  },
  "data_period": {
    "start_date": "2023-01-01T00:00:00Z",
    "end_date": "2023-06-01T00:00:00Z",
    "duration_days": 151
  }
}
```

### Dashboard Analytics

Represents the overall analytics dashboard data.

#### Properties

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| satellite_count | integer | Yes | Total number of satellites |
| satellite_status | object | Yes | Count of satellites by status |
| conjunction_summary | object | Yes | Summary of current conjunctions |
| recent_maneuvers | array | Yes | List of recent maneuvers |
| resource_summary | object | Yes | Summary of resource status |
| data_timestamp | datetime | Yes | Timestamp of the data |

#### Example

```json
{
  "satellite_count": 15,
  "satellite_status": {
    "ACTIVE": 12,
    "INACTIVE": 2,
    "DECOMMISSIONED": 1
  },
  "conjunction_summary": {
    "total_active": 8,
    "by_severity": {
      "CRITICAL": 0,
      "HIGH": 1,
      "MEDIUM": 3,
      "LOW": 4
    },
    "requiring_action": 1
  },
  "recent_maneuvers": [
    {
      "id": "mnv-015",
      "satellite_id": "sat-003",
      "type": "STATION_KEEPING",
      "status": "COMPLETED",
      "end_time": "2023-06-17T14:30:00Z"
    },
    {
      "id": "mnv-014",
      "satellite_id": "sat-008",
      "type": "COLLISION_AVOIDANCE",
      "status": "SCHEDULED",
      "scheduled_start_time": "2023-06-18T08:15:00Z"
    }
  ],
  "resource_summary": {
    "satellites_low_fuel": 2,
    "satellites_power_issues": 0,
    "overall_health_status": "GOOD"
  },
  "data_timestamp": "2023-06-17T18:00:00Z"
}
```

## Request/Response Models

### SatelliteCreate

Model for creating a new satellite.

#### Properties

Same as Satellite model, except:
- `id` is not included (auto-generated)
- `created_at` and `updated_at` are not included (auto-generated)

### SatelliteUpdate

Model for updating an existing satellite.

#### Properties

Same as Satellite model, except:
- All fields are optional
- Only the provided fields will be updated
- `id`, `created_at`, and `updated_at` cannot be modified

### SatelliteResponse

Model for satellite data in responses.

#### Properties

Same as Satellite model, with all fields included.

### ManeuverCreate

Model for creating a new maneuver.

#### Properties

Same as Maneuver model, except:
- `id` is not included (auto-generated)
- `status` is not included (defaults to "SCHEDULED")
- `actual_start_time` and `end_time` are not included
- `created_at` and `updated_at` are not included (auto-generated)

### ManeuverUpdate

Model for updating an existing maneuver.

#### Properties

Same as Maneuver model, except:
- All fields are optional
- Only the provided fields will be updated
- `id`, `satellite_id`, `created_by`, `created_at`, and `updated_at` cannot be modified

### ManeuverResponse

Model for maneuver data in responses.

#### Properties

Same as Maneuver model, with all fields included.

## Pagination and Filtering

### PaginatedResponse

Generic model for paginated responses.

#### Properties

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| data | array | Yes | Array of items (depends on endpoint) |
| meta | object | Yes | Pagination metadata |

#### Meta

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| page | integer | Yes | Current page number |
| limit | integer | Yes | Items per page |
| total | integer | Yes | Total number of items |
| pages | integer | Yes | Total number of pages |

#### Example

```json
{
  "data": [
    {
      "id": "sat-001",
      "name": "AstroShield-1",
      "norad_id": "43657",
      "international_designator": "2018-099A",
      "orbit_type": "LEO",
      "launch_date": "2023-05-15T00:00:00Z",
      "operational_status": "ACTIVE",
      "owner": "AstroShield Inc.",
      "mission": "Space Domain Awareness",
      "orbital_parameters": {
        "semi_major_axis": 7000.0,
        "eccentricity": 0.0001,
        "inclination": 51.6,
        "raan": 235.7,
        "argument_of_perigee": 90.0,
        "mean_anomaly": 0.0,
        "epoch": "2023-05-15T12:00:00Z"
      },
      "created_at": "2023-05-14T10:30:00Z",
      "updated_at": "2023-05-14T10:30:00Z"
    },
    // ... more items
  ],
  "meta": {
    "page": 1,
    "limit": 20,
    "total": 45,
    "pages": 3
  }
}
```

### Filter Parameters

Common filter parameters for collection endpoints.

#### Satellite Filters

| Parameter | Type | Description |
|-----------|------|-------------|
| name | string | Filter by satellite name (partial match) |
| norad_id | string | Filter by NORAD ID (exact match) |
| orbit_type | string | Filter by orbit type |
| operational_status | string | Filter by operational status |
| owner | string | Filter by owner (partial match) |
| launch_date_from | datetime | Filter by launch date (greater than or equal) |
| launch_date_to | datetime | Filter by launch date (less than or equal) |

#### Maneuver Filters

| Parameter | Type | Description |
|-----------|------|-------------|
| satellite_id | string | Filter by satellite ID |
| type | string | Filter by maneuver type |
| status | string | Filter by maneuver status |
| scheduled_from | datetime | Filter by scheduled start time (greater than or equal) |
| scheduled_to | datetime | Filter by scheduled start time (less than or equal) |

#### Conjunction Filters

| Parameter | Type | Description |
|-----------|------|-------------|
| satellite_id | string | Filter by satellite ID |
| severity_level | string | Filter by severity level |
| status | string | Filter by status |
| tca_from | datetime | Filter by TCA (greater than or equal) |
| tca_to | datetime | Filter by TCA (less than or equal) |
| min_probability | float | Filter by minimum probability |

## Error Models

### ErrorResponse

Model for error responses.

#### Properties

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| detail | string | Yes | Human-readable error message |
| status_code | integer | Yes | HTTP status code |
| error_code | string | No | Machine-readable error code |
| path | string | No | Path that caused the error |
| timestamp | datetime | No | Time when the error occurred |

#### Example

```json
{
  "detail": "Satellite with ID 'sat-999' not found",
  "status_code": 404,
  "error_code": "resource_not_found",
  "path": "/api/v1/satellites/sat-999",
  "timestamp": "2023-06-18T14:25:30Z"
}
```

### ValidationError

Model for validation error responses.

#### Properties

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| detail | object | Yes | Validation errors by field |
| status_code | integer | Yes | HTTP status code (typically 400) |
| error_code | string | Yes | Error code (typically "validation_error") |

#### Example

```json
{
  "detail": {
    "orbital_parameters.semi_major_axis": [
      "value is less than minimum allowed (6378.0)"
    ],
    "launch_date": [
      "invalid date format, expected ISO 8601"
    ]
  },
  "status_code": 400,
  "error_code": "validation_error"
}
```

## Relationships

### Data Model Relationships

- **Satellite** to **Maneuver**: One-to-many (a satellite can have multiple maneuvers)
- **Satellite** to **Conjunction**: One-to-many (a satellite can have multiple conjunctions)
- **User** to **Maneuver**: One-to-many (a user can create multiple maneuvers)
- **Satellite** to **SatelliteAnalytics**: One-to-one (a satellite has one analytics record)

## Data Type Details

### Date and Time

All date/time values use ISO 8601 format in UTC:
- Format: `YYYY-MM-DDThh:mm:ssZ`
- Example: `2023-06-18T14:30:00Z`

### Coordinates and Vectors

- Orbital parameters use the Earth-centered inertial (ECI) reference frame
- Direction vectors are unitless and normalized (magnitude = 1)
- Angles are in degrees (0-360)

### Identifiers

- IDs are strings with a prefix indicating the entity type:
  - `sat-XXX` for satellites
  - `mnv-XXX` for maneuvers
  - `conj-XXX` for conjunctions
  - `usr-XXX` for users

## Best Practices

### Working with Orbital Data

- Always include epoch when working with orbital elements
- Be aware that orbital parameters degrade in accuracy over time
- Use propagation models for predicting future positions

### Resource Management

- Monitor fuel usage and remaining fuel estimates
- Track power availability, especially for maneuvers
- Consider thruster duty cycles and limitations

### Conjunction Assessment

- Consider both miss distance and probability of collision
- Higher relative velocity generally means lower collision risk for the same miss distance
- Always check conjunction data source and quality 