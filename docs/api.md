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