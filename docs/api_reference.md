# AstroShield API Reference

## Overview

This document provides a comprehensive reference for the AstroShield API, including all endpoints, request/response formats, authentication methods, and usage examples.

## Base URLs

- **Development**: `http://localhost:8000`
- **Staging**: `https://api-staging.astroshield.com`
- **Production**: `https://api.astroshield.com`

## API Versioning

The API is versioned through the URL path. The current version is `v1`.

Example: `https://api.astroshield.com/api/v1/spacecraft`

## Authentication

### JWT Authentication

All API requests (except public endpoints) require authentication using JSON Web Tokens (JWT).

#### Obtaining a Token

```http
POST /auth/token
Content-Type: application/json

{
  "username": "user@example.com",
  "password": "password123"
}
```

**Response:**

```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

#### Using the Token

Include the token in the Authorization header for all authenticated requests:

```http
GET /api/v1/spacecraft
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...
```

### API Key Authentication

For service-to-service communication, API key authentication is also supported.

```http
GET /api/v1/spacecraft
X-API-Key: your-api-key-here
```

## Rate Limiting

API requests are rate-limited to prevent abuse. The current limits are:

- **Authenticated users**: 100 requests per minute
- **Anonymous users**: 20 requests per minute

Rate limit information is included in the response headers:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1620000000
```

## Common Response Codes

| Code | Description |
|------|-------------|
| 200  | Success |
| 201  | Resource created |
| 400  | Bad request |
| 401  | Unauthorized |
| 403  | Forbidden |
| 404  | Resource not found |
| 429  | Too many requests |
| 500  | Internal server error |

## Error Responses

All error responses follow a standard format:

```json
{
  "error": {
    "code": "RESOURCE_NOT_FOUND",
    "message": "The requested resource was not found",
    "details": {
      "resource_id": "123",
      "resource_type": "spacecraft"
    }
  }
}
```

## API Endpoints

### Health Check

#### Get API Health Status

```http
GET /health
```

**Response:**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2023-05-01T12:00:00Z"
}
```

### Spacecraft

#### List Spacecraft

```http
GET /api/v1/spacecraft
```

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| page | integer | Page number (default: 1) |
| limit | integer | Items per page (default: 20, max: 100) |
| status | string | Filter by status (active, inactive, all) |

**Response:**

```json
{
  "data": [
    {
      "id": 1,
      "name": "ISS",
      "norad_id": 25544,
      "status": "active",
      "orbit_type": "LEO",
      "launch_date": "1998-11-20T06:40:00Z",
      "last_updated": "2023-05-01T12:00:00Z"
    },
    {
      "id": 2,
      "name": "Hubble Space Telescope",
      "norad_id": 20580,
      "status": "active",
      "orbit_type": "LEO",
      "launch_date": "1990-04-24T12:33:51Z",
      "last_updated": "2023-05-01T12:00:00Z"
    }
  ],
  "meta": {
    "page": 1,
    "limit": 20,
    "total": 2
  }
}
```

#### Get Spacecraft Details

```http
GET /api/v1/spacecraft/{spacecraft_id}
```

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| spacecraft_id | integer | Unique identifier of the spacecraft |

**Response:**

```json
{
  "id": 1,
  "name": "ISS",
  "norad_id": 25544,
  "status": "active",
  "orbit_type": "LEO",
  "launch_date": "1998-11-20T06:40:00Z",
  "last_updated": "2023-05-01T12:00:00Z",
  "orbital_parameters": {
    "semi_major_axis": 6783.5,
    "eccentricity": 0.0001,
    "inclination": 51.64,
    "raan": 247.89,
    "argument_of_perigee": 283.9,
    "mean_anomaly": 120.87,
    "epoch": "2023-05-01T12:00:00Z"
  },
  "physical_parameters": {
    "mass": 420000,
    "length": 73,
    "width": 109,
    "height": 27.5,
    "cross_section": 2477.5
  }
}
```

#### Create Spacecraft

```http
POST /api/v1/spacecraft
Content-Type: application/json
Authorization: Bearer {token}

{
  "name": "New Satellite",
  "norad_id": 12345,
  "status": "active",
  "orbit_type": "GEO",
  "launch_date": "2023-01-01T00:00:00Z",
  "orbital_parameters": {
    "semi_major_axis": 42164,
    "eccentricity": 0.0002,
    "inclination": 0.1,
    "raan": 95.5,
    "argument_of_perigee": 270,
    "mean_anomaly": 0,
    "epoch": "2023-05-01T12:00:00Z"
  },
  "physical_parameters": {
    "mass": 5000,
    "length": 15,
    "width": 3,
    "height": 3,
    "cross_section": 45
  }
}
```

**Response:**

```json
{
  "id": 3,
  "name": "New Satellite",
  "norad_id": 12345,
  "status": "active",
  "orbit_type": "GEO",
  "launch_date": "2023-01-01T00:00:00Z",
  "last_updated": "2023-05-01T12:00:00Z",
  "orbital_parameters": {
    "semi_major_axis": 42164,
    "eccentricity": 0.0002,
    "inclination": 0.1,
    "raan": 95.5,
    "argument_of_perigee": 270,
    "mean_anomaly": 0,
    "epoch": "2023-05-01T12:00:00Z"
  },
  "physical_parameters": {
    "mass": 5000,
    "length": 15,
    "width": 3,
    "height": 3,
    "cross_section": 45
  }
}
```

#### Update Spacecraft

```http
PUT /api/v1/spacecraft/{spacecraft_id}
Content-Type: application/json
Authorization: Bearer {token}

{
  "status": "inactive",
  "orbital_parameters": {
    "semi_major_axis": 42165,
    "eccentricity": 0.0003,
    "inclination": 0.15,
    "raan": 95.6,
    "argument_of_perigee": 271,
    "mean_anomaly": 1,
    "epoch": "2023-05-02T12:00:00Z"
  }
}
```

**Response:**

```json
{
  "id": 3,
  "name": "New Satellite",
  "norad_id": 12345,
  "status": "inactive",
  "orbit_type": "GEO",
  "launch_date": "2023-01-01T00:00:00Z",
  "last_updated": "2023-05-02T12:00:00Z",
  "orbital_parameters": {
    "semi_major_axis": 42165,
    "eccentricity": 0.0003,
    "inclination": 0.15,
    "raan": 95.6,
    "argument_of_perigee": 271,
    "mean_anomaly": 1,
    "epoch": "2023-05-02T12:00:00Z"
  },
  "physical_parameters": {
    "mass": 5000,
    "length": 15,
    "width": 3,
    "height": 3,
    "cross_section": 45
  }
}
```

#### Delete Spacecraft

```http
DELETE /api/v1/spacecraft/{spacecraft_id}
Authorization: Bearer {token}
```

**Response:**

```
204 No Content
```

### Conjunction Analysis

#### Get Conjunction Events

```http
GET /api/v1/conjunctions
```

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| spacecraft_id | integer | Filter by spacecraft ID |
| start_time | string | Start time in ISO 8601 format |
| end_time | string | End time in ISO 8601 format |
| min_probability | number | Minimum collision probability |
| page | integer | Page number (default: 1) |
| limit | integer | Items per page (default: 20, max: 100) |

**Response:**

```json
{
  "data": [
    {
      "id": 1,
      "primary_object": {
        "id": 1,
        "name": "ISS",
        "norad_id": 25544
      },
      "secondary_object": {
        "id": null,
        "name": "Debris",
        "norad_id": 45678
      },
      "tca": "2023-05-10T15:30:00Z",
      "miss_distance": 0.5,
      "collision_probability": 0.00001,
      "relative_velocity": 10.5,
      "created_at": "2023-05-01T12:00:00Z",
      "updated_at": "2023-05-01T12:00:00Z"
    }
  ],
  "meta": {
    "page": 1,
    "limit": 20,
    "total": 1
  }
}
```

#### Get Conjunction Details

```http
GET /api/v1/conjunctions/{conjunction_id}
```

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| conjunction_id | integer | Unique identifier of the conjunction event |

**Response:**

```json
{
  "id": 1,
  "primary_object": {
    "id": 1,
    "name": "ISS",
    "norad_id": 25544,
    "state_vector": {
      "position": [6500, 1000, 100],
      "velocity": [0.1, 7.5, 0.5],
      "epoch": "2023-05-10T15:30:00Z"
    }
  },
  "secondary_object": {
    "id": null,
    "name": "Debris",
    "norad_id": 45678,
    "state_vector": {
      "position": [6500.5, 1000.1, 100.2],
      "velocity": [0.2, 7.6, 0.6],
      "epoch": "2023-05-10T15:30:00Z"
    }
  },
  "tca": "2023-05-10T15:30:00Z",
  "miss_distance": 0.5,
  "collision_probability": 0.00001,
  "relative_velocity": 10.5,
  "covariance_primary": [
    [0.01, 0.001, 0.0001],
    [0.001, 0.01, 0.0001],
    [0.0001, 0.0001, 0.01]
  ],
  "covariance_secondary": [
    [0.1, 0.01, 0.001],
    [0.01, 0.1, 0.001],
    [0.001, 0.001, 0.1]
  ],
  "created_at": "2023-05-01T12:00:00Z",
  "updated_at": "2023-05-01T12:00:00Z"
}
```

### Cyber Threats

#### Get Cyber Threats

```http
GET /api/v1/spacecraft/{spacecraft_id}/cyber-threats
```

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| spacecraft_id | integer | Unique identifier of the spacecraft |

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| start_time | string | Start time in ISO 8601 format |
| end_time | string | End time in ISO 8601 format |
| severity | string | Filter by severity (low, medium, high, critical) |
| page | integer | Page number (default: 1) |
| limit | integer | Items per page (default: 20, max: 100) |

**Response:**

```json
{
  "data": [
    {
      "id": 1,
      "spacecraft_id": 1,
      "type": "jamming",
      "severity": "high",
      "confidence": 0.85,
      "start_time": "2023-05-01T10:00:00Z",
      "end_time": "2023-05-01T10:30:00Z",
      "description": "Suspected jamming attack on communication subsystem",
      "affected_subsystems": ["communications"],
      "mitigation_status": "resolved",
      "created_at": "2023-05-01T10:05:00Z",
      "updated_at": "2023-05-01T11:00:00Z"
    }
  ],
  "meta": {
    "page": 1,
    "limit": 20,
    "total": 1
  }
}
```

### Stability Analysis

#### Get Stability Data

```http
GET /api/v1/stability/data
```

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| start_time | string | Start time in ISO 8601 format |
| end_time | string | End time in ISO 8601 format |
| spacecraft_id | integer | Filter by spacecraft ID (optional) |
| page | integer | Page number (default: 1) |
| limit | integer | Items per page (default: 20, max: 100) |

**Response:**

```json
{
  "data": [
    {
      "id": 1,
      "spacecraft_id": 1,
      "timestamp": "2023-05-01T12:00:00Z",
      "stability_index": 0.95,
      "anomaly_score": 0.02,
      "subsystem_metrics": {
        "power": {
          "stability": 0.98,
          "anomaly_score": 0.01
        },
        "thermal": {
          "stability": 0.97,
          "anomaly_score": 0.02
        },
        "attitude": {
          "stability": 0.94,
          "anomaly_score": 0.03
        },
        "communications": {
          "stability": 0.99,
          "anomaly_score": 0.01
        }
      }
    }
  ],
  "meta": {
    "page": 1,
    "limit": 20,
    "total": 1
  }
}
```

### Telemetry

#### Get Telemetry Data

```http
GET /api/v1/spacecraft/{spacecraft_id}/telemetry
```

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| spacecraft_id | integer | Unique identifier of the spacecraft |

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| start_time | string | Start time in ISO 8601 format |
| end_time | string | End time in ISO 8601 format |
| subsystem | string | Filter by subsystem (power, thermal, attitude, communications) |
| page | integer | Page number (default: 1) |
| limit | integer | Items per page (default: 20, max: 100) |

**Response:**

```json
{
  "data": [
    {
      "id": 1,
      "spacecraft_id": 1,
      "timestamp": "2023-05-01T12:00:00Z",
      "subsystem": "power",
      "metrics": {
        "battery_voltage": 28.5,
        "battery_current": 2.1,
        "battery_temperature": 25.3,
        "solar_array_voltage": 32.1,
        "solar_array_current": 5.2
      }
    },
    {
      "id": 2,
      "spacecraft_id": 1,
      "timestamp": "2023-05-01T12:00:00Z",
      "subsystem": "thermal",
      "metrics": {
        "internal_temperature": 22.5,
        "external_temperature": -15.3,
        "radiator_temperature": 10.2
      }
    }
  ],
  "meta": {
    "page": 1,
    "limit": 20,
    "total": 2
  }
}
```

## Webhooks

AstroShield supports webhooks for real-time notifications of events.

### Webhook Events

| Event Type | Description |
|------------|-------------|
| conjunction.detected | A new conjunction event has been detected |
| conjunction.updated | An existing conjunction event has been updated |
| cyber_threat.detected | A new cyber threat has been detected |
| stability.anomaly | A stability anomaly has been detected |

### Webhook Payload

```json
{
  "event_type": "conjunction.detected",
  "timestamp": "2023-05-01T12:00:00Z",
  "data": {
    "conjunction_id": 1,
    "primary_object": {
      "id": 1,
      "name": "ISS",
      "norad_id": 25544
    },
    "secondary_object": {
      "id": null,
      "name": "Debris",
      "norad_id": 45678
    },
    "tca": "2023-05-10T15:30:00Z",
    "miss_distance": 0.5,
    "collision_probability": 0.00001
  }
}
```

### Webhook Configuration

To configure webhooks, use the Webhooks API:

```http
POST /api/v1/webhooks
Content-Type: application/json
Authorization: Bearer {token}

{
  "url": "https://example.com/webhook",
  "events": ["conjunction.detected", "cyber_threat.detected"],
  "secret": "your-webhook-secret"
}
```

**Response:**

```json
{
  "id": 1,
  "url": "https://example.com/webhook",
  "events": ["conjunction.detected", "cyber_threat.detected"],
  "created_at": "2023-05-01T12:00:00Z",
  "updated_at": "2023-05-01T12:00:00Z"
}
```

## SDKs and Client Libraries

AstroShield provides official client libraries for the following languages:

- Python: [astroshield-python](https://github.com/astroshield/astroshield-python)
- JavaScript: [astroshield-js](https://github.com/astroshield/astroshield-js)
- Java: [astroshield-java](https://github.com/astroshield/astroshield-java)

## API Changelog

### v1.0.0 (2023-05-01)

- Initial release of the AstroShield API

### v1.1.0 (2023-06-15)

- Added webhook support
- Added telemetry endpoints
- Improved error handling and validation

## Support

For API support, please contact api-support@astroshield.com or visit our [developer portal](https://developers.astroshield.com). 