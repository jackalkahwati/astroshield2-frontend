# AstroShield API Documentation

## Overview

AstroShield provides a RESTful API for spacecraft collision avoidance and space traffic management. The API is organized around standard HTTP methods and uses JSON for request and response bodies.

## Base URL

```
Production: https://api.astroshield.com
Development: http://localhost:8000
```

## Authentication

The API uses JWT (JSON Web Token) for authentication. Include the token in the Authorization header:

```
Authorization: Bearer <your_token>
```

## Rate Limiting

The API implements rate limiting to ensure fair usage:
- 10 requests per second per IP address
- Rate limit headers are included in responses
- When exceeded, returns 429 Too Many Requests

## Endpoints

### Stability Analysis

#### GET /stability/data
Returns current stability metrics for the spacecraft.

**Response**
```json
{
  "timestamp": "2024-01-03T12:00:00Z",
  "overall_stability": 0.85,
  "metrics": {
    "attitude_stability": 0.9,
    "orbit_stability": 0.8,
    "power_stability": 0.85
  },
  "warnings": []
}
```

### Maneuvers

#### GET /maneuvers/active
Returns list of active maneuvers.

**Response**
```json
{
  "maneuvers": [
    {
      "id": "m123",
      "type": "collision_avoidance",
      "status": "in_progress",
      "start_time": "2024-01-03T12:00:00Z",
      "end_time": "2024-01-03T12:30:00Z"
    }
  ]
}
```

#### POST /maneuvers/plan
Plan a new maneuver.

**Request**
```json
{
  "type": "collision_avoidance",
  "target_orbit": {
    "semi_major_axis": 7000,
    "eccentricity": 0.001,
    "inclination": 51.6
  }
}
```

### Analytics

#### GET /analytics/data
Returns analytics data for specified time range.

**Parameters**
- `start_time`: ISO 8601 timestamp
- `end_time`: ISO 8601 timestamp
- `metrics`: Comma-separated list of metrics

**Response**
```json
{
  "time_range": {
    "start": "2024-01-01T00:00:00Z",
    "end": "2024-01-02T00:00:00Z"
  },
  "data_points": [
    {
      "timestamp": "2024-01-01T00:00:00Z",
      "metrics": {
        "orbit_stability": 0.95,
        "power_consumption": 80.5
      }
    }
  ]
}
```

### Tracking

#### GET /tracking/status
Returns current tracking status.

**Response**
```json
{
  "timestamp": "2024-01-03T12:00:00Z",
  "status": "active",
  "current_position": {
    "latitude": 45.5,
    "longitude": -122.6,
    "altitude": 400
  }
}
```

## Error Handling

The API uses conventional HTTP response codes:
- 2xx: Success
- 4xx: Client errors
- 5xx: Server errors

Error responses include:
```json
{
  "detail": "Error message",
  "timestamp": "2024-01-03T12:00:00Z",
  "error_code": "VALIDATION_ERROR"
}
```

## Health Checks

#### GET /health
Basic health check endpoint.

**Response**
```json
{
  "status": "ok",
  "timestamp": "2024-01-03T12:00:00Z"
}
```

#### GET /health/details
Detailed health check with component status.

**Response**
```json
{
  "status": "ok",
  "components": {
    "database": "ok",
    "redis": "ok"
  },
  "version": "1.0.0"
}
``` 