# API Documentation

## Overview

This document outlines the API endpoints and their usage for the AstroShield platform.

## Base URL

- Development: `http://localhost:8000`
- Production: `https://api.astroshield.com`

## Authentication

### JWT Authentication

```bash
# Request token
POST /auth/token
Content-Type: application/json

{
  "username": "user@example.com",
  "password": "password123"
}

# Response
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}

# Using token
GET /api/protected-endpoint
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...
```

## Endpoints

### Stability Analysis

1. Get Stability Data
   ```bash
   GET /stability/data
   
   # Query Parameters
   start_time: ISO8601 timestamp
   end_time: ISO8601 timestamp
   satellite_id: string (optional)
   
   # Response
   {
     "data": [
       {
         "timestamp": "2024-01-01T00:00:00Z",
         "satellite_id": "SAT-001",
         "stability_score": 0.95,
         "attitude_error": 0.02,
         "orbital_parameters": {
           "semi_major_axis": 7000,
           "eccentricity": 0.001,
           "inclination": 45.0
         }
       }
     ],
     "metadata": {
       "total_records": 100,
       "start_time": "2024-01-01T00:00:00Z",
       "end_time": "2024-01-02T00:00:00Z"
     }
   }
   ```

2. Calculate Stability Score
   ```bash
   POST /stability/calculate
   Content-Type: application/json
   
   {
     "satellite_id": "SAT-001",
     "timestamp": "2024-01-01T00:00:00Z",
     "orbital_parameters": {
       "position": [1000, 2000, 3000],
       "velocity": [-1, 2, -3]
     }
   }
   
   # Response
   {
     "stability_score": 0.95,
     "confidence": 0.98,
     "recommendations": [
       {
         "type": "attitude_adjustment",
         "description": "Adjust satellite attitude by 0.5 degrees",
         "priority": "medium"
       }
     ]
   }
   ```

### Maneuvers

1. List Active Maneuvers
   ```bash
   GET /maneuvers/active
   
   # Response
   {
     "maneuvers": [
       {
         "id": "MNV-001",
         "satellite_id": "SAT-001",
         "type": "orbit_correction",
         "start_time": "2024-01-01T00:00:00Z",
         "end_time": "2024-01-01T01:00:00Z",
         "status": "in_progress",
         "parameters": {
           "delta_v": 10.5,
           "direction": [1, 0, 0]
         }
       }
     ],
     "total": 1
   }
   ```

2. Plan Maneuver
   ```bash
   POST /maneuvers/plan
   Content-Type: application/json
   
   {
     "satellite_id": "SAT-001",
     "type": "orbit_correction",
     "target_parameters": {
       "altitude": 500,
       "inclination": 45,
       "eccentricity": 0
     },
     "constraints": {
       "max_delta_v": 50,
       "max_duration": 3600
     }
   }
   
   # Response
   {
     "maneuver_id": "MNV-002",
     "execution_plan": [
       {
         "sequence": 1,
         "type": "burn",
         "delta_v": 10.5,
         "direction": [1, 0, 0],
         "duration": 300
       }
     ],
     "estimated_resources": {
       "fuel_consumption": 2.5,
       "power_usage": 1000
     }
   }
   ```

### Analytics

1. Get Analytics Data
   ```bash
   GET /analytics/data
   
   # Query Parameters
   metric: string (required)
   start_time: ISO8601 timestamp
   end_time: ISO8601 timestamp
   interval: string (1h, 1d, 1w)
   
   # Response
   {
     "data": [
       {
         "timestamp": "2024-01-01T00:00:00Z",
         "value": 42.5,
         "metadata": {
           "confidence": 0.95,
           "source": "sensor_data"
         }
       }
     ],
     "summary": {
       "min": 40.0,
       "max": 45.0,
       "mean": 42.5,
       "std_dev": 1.2
     }
   }
   ```

2. Generate Report
   ```bash
   POST /analytics/report
   Content-Type: application/json
   
   {
     "report_type": "stability_analysis",
     "time_range": {
       "start": "2024-01-01T00:00:00Z",
       "end": "2024-01-02T00:00:00Z"
     },
     "metrics": ["stability_score", "attitude_error"],
     "format": "pdf"
   }
   
   # Response
   {
     "report_id": "RPT-001",
     "status": "generating",
     "estimated_completion": "2024-01-01T00:05:00Z",
     "download_url": null
   }
   ```

### Settings

1. Get System Settings
   ```bash
   GET /settings/system
   
   # Response
   {
     "settings": {
       "stability_threshold": 0.8,
       "alert_sensitivity": "medium",
       "data_retention_days": 90,
       "automatic_maneuvers": false
     },
     "last_updated": "2024-01-01T00:00:00Z",
     "updated_by": "admin@example.com"
   }
   ```

2. Update Settings
   ```bash
   PATCH /settings/system
   Content-Type: application/json
   
   {
     "stability_threshold": 0.85,
     "alert_sensitivity": "high"
   }
   
   # Response
   {
     "updated": [
       "stability_threshold",
       "alert_sensitivity"
     ],
     "timestamp": "2024-01-01T00:00:00Z"
   }
   ```

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input parameters",
    "details": [
      {
        "field": "timestamp",
        "error": "Must be a valid ISO8601 timestamp"
      }
    ],
    "request_id": "req-123-456"
  }
}
```

### Common Error Codes

1. Authentication Errors
   - `UNAUTHORIZED`: Missing or invalid authentication
   - `TOKEN_EXPIRED`: Authentication token has expired
   - `INVALID_CREDENTIALS`: Invalid username or password

2. Validation Errors
   - `VALIDATION_ERROR`: Invalid input parameters
   - `INVALID_FORMAT`: Invalid data format
   - `MISSING_REQUIRED`: Required field missing

3. Resource Errors
   - `NOT_FOUND`: Requested resource not found
   - `ALREADY_EXISTS`: Resource already exists
   - `CONFLICT`: Resource state conflict

4. System Errors
   - `INTERNAL_ERROR`: Internal server error
   - `SERVICE_UNAVAILABLE`: Service temporarily unavailable
   - `RATE_LIMITED`: Too many requests

## Rate Limiting

```bash
# Rate limit headers
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1704067200

# Rate limit error response
{
  "error": {
    "code": "RATE_LIMITED",
    "message": "Rate limit exceeded",
    "details": {
      "limit": 100,
      "remaining": 0,
      "reset": 1704067200
    }
  }
}
```

## Pagination

```bash
# Request with pagination
GET /maneuvers/history?page=2&per_page=20

# Response with pagination metadata
{
  "data": [...],
  "pagination": {
    "current_page": 2,
    "per_page": 20,
    "total_pages": 5,
    "total_items": 100,
    "links": {
      "first": "/maneuvers/history?page=1&per_page=20",
      "prev": "/maneuvers/history?page=1&per_page=20",
      "next": "/maneuvers/history?page=3&per_page=20",
      "last": "/maneuvers/history?page=5&per_page=20"
    }
  }
}
```

## Versioning

```bash
# Specify API version
Accept: application/json; version=1.0

# Version in URL
GET /v1/stability/data

# Version header in response
API-Version: 1.0
```

## Webhooks

1. Register Webhook
   ```bash
   POST /webhooks/register
   Content-Type: application/json
   
   {
     "url": "https://example.com/webhook",
     "events": ["maneuver.completed", "stability.alert"],
     "secret": "webhook_secret_123"
   }
   
   # Response
   {
     "webhook_id": "WHK-001",
     "status": "active",
     "created_at": "2024-01-01T00:00:00Z"
   }
   ```

2. Webhook Payload
   ```json
   {
     "event": "maneuver.completed",
     "timestamp": "2024-01-01T00:00:00Z",
     "data": {
       "maneuver_id": "MNV-001",
       "status": "completed",
       "results": {
         "success": true,
         "actual_delta_v": 10.2
       }
     },
     "webhook_id": "WHK-001"
   }
   ```

# CCDM Service API Documentation

## Overview

The Conjunction and Collision Data Management (CCDM) service provides a RESTful API for accessing space object conjunction data, collision risk assessments, and related space situational awareness information. This document outlines the available endpoints, request/response formats, authentication methods, and usage examples.

## Base URL

```
https://api.ccdm.example.com/v1
```

## Authentication

The API uses JWT (JSON Web Token) authentication. Include the token in the Authorization header:

```
Authorization: Bearer <your_jwt_token>
```

To obtain a token, use the authentication endpoint:

```
POST /auth/login
```

### Authentication Request

```json
{
  "username": "your_username",
  "password": "your_password"
}
```

### Authentication Response

```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "expiresIn": 3600
}
```

## Rate Limiting

API requests are rate-limited to prevent abuse. The current limits are:

- 100 requests per minute for standard users
- 300 requests per minute for premium users

Rate limit information is included in the response headers:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1620000000
```

## API Endpoints

### Conjunction Data

#### Get Conjunction Events

```
GET /conjunctions
```

Query Parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| startTime | ISO8601 | Start time for conjunction events |
| endTime | ISO8601 | End time for conjunction events |
| minPc | number | Minimum probability of collision |
| primaryNorad | string | NORAD ID of primary object |
| limit | number | Maximum number of results (default: 100, max: 1000) |
| offset | number | Result offset for pagination |

Response:

```json
{
  "data": [
    {
      "id": "conj-12345",
      "tca": "2023-06-15T08:22:15Z",
      "primaryObject": {
        "noradId": "25544",
        "name": "ISS (ZARYA)",
        "type": "PAYLOAD"
      },
      "secondaryObject": {
        "noradId": "48274",
        "name": "COSMOS 2251 DEB",
        "type": "DEBRIS"
      },
      "missDistance": 325,
      "probabilityOfCollision": 0.00015,
      "relativeVelocity": 14.2
    },
    // Additional conjunction events...
  ],
  "pagination": {
    "total": 2345,
    "limit": 100,
    "offset": 0,
    "nextOffset": 100
  }
}
```

#### Get Conjunction Event Details

```
GET /conjunctions/{id}
```

Response:

```json
{
  "id": "conj-12345",
  "tca": "2023-06-15T08:22:15Z",
  "primaryObject": {
    "noradId": "25544",
    "name": "ISS (ZARYA)",
    "type": "PAYLOAD",
    "size": 109.43,
    "mass": 420000,
    "rcs": 399.1
  },
  "secondaryObject": {
    "noradId": "48274",
    "name": "COSMOS 2251 DEB",
    "type": "DEBRIS",
    "size": 0.1,
    "mass": 0.5,
    "rcs": 0.02
  },
  "missDistance": 325,
  "probabilityOfCollision": 0.00015,
  "relativeVelocity": 14.2,
  "covariance": {
    "primary": [[0.001, 0, 0], [0, 0.001, 0], [0, 0, 0.001]],
    "secondary": [[0.05, 0, 0], [0, 0.05, 0], [0, 0, 0.05]]
  },
  "createdAt": "2023-06-14T10:15:22Z",
  "updatedAt": "2023-06-14T16:20:15Z",
  "source": "JSpOC"
}
```

### Space Objects

#### Get Space Object

```
GET /objects/{noradId}
```

Response:

```json
{
  "noradId": "25544",
  "name": "ISS (ZARYA)",
  "internationalDesignator": "1998-067A",
  "type": "PAYLOAD",
  "size": 109.43,
  "mass": 420000,
  "rcs": 399.1,
  "launchDate": "1998-11-20T00:00:00Z",
  "country": "ISS",
  "orbit": {
    "type": "LEO",
    "semiMajorAxis": 6783.4,
    "eccentricity": 0.0004768,
    "inclination": 51.6426,
    "raan": 247.4627,
    "argOfPerigee": 130.5360,
    "meanAnomaly": 325.0288,
    "epoch": "2023-06-15T06:00:00Z"
  },
  "status": "ACTIVE"
}
```

#### Search Space Objects

```
GET /objects
```

Query Parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| name | string | Object name (partial match) |
| type | string | Object type (PAYLOAD, DEBRIS, ROCKET_BODY) |
| orbit | string | Orbit type (LEO, MEO, GEO, HEO) |
| status | string | Object status (ACTIVE, INACTIVE) |
| launchDateStart | ISO8601 | Minimum launch date |
| launchDateEnd | ISO8601 | Maximum launch date |
| limit | number | Maximum number of results (default: 100, max: 1000) |
| offset | number | Result offset for pagination |

Response:

```json
{
  "data": [
    {
      "noradId": "25544",
      "name": "ISS (ZARYA)",
      "internationalDesignator": "1998-067A",
      "type": "PAYLOAD",
      "orbit": {
        "type": "LEO"
      },
      "status": "ACTIVE"
    },
    // Additional space objects...
  ],
  "pagination": {
    "total": 23456,
    "limit": 100,
    "offset": 0,
    "nextOffset": 100
  }
}
```

### Maneuver Planning

#### Calculate Avoidance Maneuver

```
POST /maneuvers/calculate
```

Request:

```json
{
  "conjunctionId": "conj-12345",
  "maneuverType": "PROGRADE",
  "maneuverTime": "2023-06-14T18:00:00Z",
  "deltaV": 0.1
}
```

Response:

```json
{
  "id": "man-6789",
  "conjunctionId": "conj-12345",
  "maneuverType": "PROGRADE",
  "maneuverTime": "2023-06-14T18:00:00Z",
  "deltaV": 0.1,
  "fuelConsumption": 0.75,
  "newProbabilityOfCollision": 0.000001,
  "newMissDistance": 2450,
  "postManeuverOrbit": {
    "semiMajorAxis": 6783.6,
    "eccentricity": 0.0004772,
    "inclination": 51.6426,
    "raan": 247.4627,
    "argOfPerigee": 130.5360,
    "meanAnomaly": 325.0300
  }
}
```

### Reporting

#### Generate Conjunction Summary Report

```
GET /reports/conjunction-summary
```

Query Parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| startTime | ISO8601 | Start time for report period |
| endTime | ISO8601 | End time for report period |
| minPc | number | Minimum probability of collision |
| format | string | Output format (JSON, CSV, PDF) |

Response (JSON format):

```json
{
  "reportId": "rep-34567",
  "generatedAt": "2023-06-15T10:00:00Z",
  "period": {
    "start": "2023-06-01T00:00:00Z",
    "end": "2023-06-15T00:00:00Z"
  },
  "summary": {
    "totalConjunctions": 156,
    "highRiskConjunctions": 3,
    "objectsInvolved": 212,
    "maneuversPlanned": 2,
    "maneuversExecuted": 1
  },
  "highRiskEvents": [
    {
      "id": "conj-12345",
      "tca": "2023-06-15T08:22:15Z",
      "primaryObject": {
        "noradId": "25544",
        "name": "ISS (ZARYA)"
      },
      "secondaryObject": {
        "noradId": "48274",
        "name": "COSMOS 2251 DEB"
      },
      "probabilityOfCollision": 0.00015,
      "status": "MONITORING"
    },
    // Additional high-risk events...
  ],
  "downloadUrl": "https://api.ccdm.example.com/v1/reports/downloads/rep-34567"
}
```

## Error Handling

The API uses standard HTTP status codes and returns error details in the response body:

```json
{
  "error": {
    "code": "INVALID_PARAMETER",
    "message": "Invalid value for parameter 'minPc': must be a number between 0 and 1",
    "details": {
      "parameter": "minPc",
      "value": "xyz",
      "constraint": "0 <= minPc <= 1"
    }
  }
}
```

Common error codes:

- `AUTHENTICATION_REQUIRED`: Authentication is required but was not provided
- `INVALID_CREDENTIALS`: The provided credentials are invalid
- `INSUFFICIENT_PERMISSIONS`: The user lacks required permissions
- `RESOURCE_NOT_FOUND`: The requested resource does not exist
- `INVALID_PARAMETER`: A request parameter has an invalid value
- `RATE_LIMIT_EXCEEDED`: The user has exceeded their rate limit
- `INTERNAL_ERROR`: An internal server error occurred

## Webhooks

The CCDM service supports webhooks for real-time notifications. Configure webhooks in the user dashboard.

Available event types:

- `conjunction.new`: A new conjunction event is detected
- `conjunction.updated`: An existing conjunction's data is updated
- `conjunction.risk_increased`: A conjunction's risk level increased
- `maneuver.planned`: A new maneuver has been planned
- `maneuver.executed`: A planned maneuver has been executed

Webhook payload example:

```json
{
  "eventType": "conjunction.risk_increased",
  "timestamp": "2023-06-15T06:14:22Z",
  "data": {
    "conjunctionId": "conj-12345",
    "previousPc": 0.00005,
    "newPc": 0.00015,
    "tca": "2023-06-15T08:22:15Z",
    "objects": ["25544", "48274"]
  }
}
```

## Client Libraries

Official client libraries are available for:

- Python: [GitHub - ccdm/ccdm-python](https://github.com/ccdm/ccdm-python)
- JavaScript: [GitHub - ccdm/ccdm-js](https://github.com/ccdm/ccdm-js)
- Java: [GitHub - ccdm/ccdm-java](https://github.com/ccdm/ccdm-java)

## API Versioning

The API uses URL versioning (e.g., `/v1/`). We maintain backward compatibility within major versions and provide deprecation notices at least 6 months before removing features.

## Support

For API support, contact:

- Email: api-support@ccdm.example.com
- Documentation: https://docs.ccdm.example.com
- Status page: https://status.ccdm.example.com 