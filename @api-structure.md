# AstroShield Microservice API Structure

## Base URL
- Development: `http://localhost:3000`
- Test: `http://localhost:${TEST_PORT}`
- Production: `https://api.astroshield.com`

## Authentication
All endpoints except health checks require authentication via JWT token.

### Headers
```
Authorization: Bearer <jwt_token>
Content-Type: application/json
CSRF-Token: <csrf_token>  // Required for POST/PUT/DELETE requests
```

## Rate Limiting
- Default: 100 requests per 15 minutes per IP
- Burst: 200 requests per minute for authenticated users
- Response Headers:
  - `X-RateLimit-Limit`
  - `X-RateLimit-Remaining`
  - `X-RateLimit-Reset`

## API Endpoints

### Health Checks
#### GET /health/live
Check if service is running.
```json
Response 200:
{
  "status": "alive"
}
```

#### GET /health/ready
Check if service is ready to handle requests.
```json
Response 200:
{
  "status": "ready"
}
```

### Spacecraft Operations

#### GET /spacecraft/:id/status
Get current status of a spacecraft.
```json
Response 200:
{
  "spacecraft_id": "string",
  "status": "operational | inactive | error",
  "last_update": "ISO8601 timestamp"
}

Response 404:
{
  "status": "error",
  "message": "Spacecraft not found"
}
```

#### POST /spacecraft/telemetry
Submit telemetry data for a spacecraft.
```json
Request:
{
  "spacecraft_id": "string",
  "timestamp": "ISO8601",
  "measurements": {
    "position": [number, number, number],
    "velocity": [number, number, number],
    "temperature": number,
    "pressure": number
  }
}

Response 200:
{
  "status": "success",
  "message": "Telemetry data received"
}
```

#### POST /spacecraft/telemetry/batch
Submit batch telemetry data.
```json
Request:
{
  "spacecraft_id": "string",
  "data": [
    {
      "timestamp": "ISO8601",
      "readings": {
        "position": [number, number, number],
        "velocity": [number, number, number]
      }
    }
  ]
}

Response 200:
{
  "status": "success",
  "message": "Batch telemetry data processed"
}
```

#### POST /spacecraft/analyze/trajectory
Analyze spacecraft trajectory.
```json
Request:
{
  "data_points": [
    {
      "time": number,
      "position": [number, number, number]
    }
  ]
}

Response 200:
{
  "status": "success",
  "analysis": {
    "risk_level": "low | medium | high",
    "confidence": number
  }
}
```

### ML-Enhanced Operations (Planned)

#### POST /spacecraft/intent
Analyze spacecraft intent using ML models.
```json
Request:
{
  "spacecraft_id": "string",
  "telemetry_window": {
    "start": "ISO8601",
    "end": "ISO8601"
  }
}

Response 200:
{
  "intent_analysis": {
    "predicted_behavior": "string",
    "confidence": number,
    "risk_factors": [string]
  }
}
```

#### GET /conjunction/active
Get active conjunctions with ML-enhanced risk assessment.
```json
Response 200:
{
  "active_conjunctions": [
    {
      "id": "string",
      "objects": ["string"],
      "time_of_closest_approach": "ISO8601",
      "miss_distance": number,
      "collision_probability": number,
      "ml_risk_assessment": {
        "risk_level": "string",
        "confidence": number,
        "factors": [string]
      }
    }
  ]
}
```

## Error Responses

### Standard Error Format
```json
{
  "status": "error",
  "message": "string",
  "code": "string",
  "details": {}  // Optional additional information
}
```

### Common Error Codes
- 400: Bad Request
- 401: Unauthorized
- 403: Forbidden
- 404: Not Found
- 429: Too Many Requests
- 500: Internal Server Error

## Response Headers
All responses include:
```
X-Request-ID: <uuid>
X-Response-Time: <ms>
```

## Monitoring Endpoints

### GET /metrics
Prometheus metrics endpoint (internal use only).
```
# HELP api_latency_seconds API endpoint latency in seconds
# TYPE api_latency_seconds histogram
...
```

### GET /health/metrics
Basic health metrics (authenticated access).
```json
Response 200:
{
  "uptime": number,
  "memory_usage": number,
  "cpu_usage": number,
  "active_connections": number,
  "error_rate": number
}
```

## Data Validation

### Telemetry Schema
```json
{
  "type": "object",
  "required": ["spacecraft_id", "timestamp", "measurements"],
  "properties": {
    "spacecraft_id": { "type": "string" },
    "timestamp": { 
      "type": "string",
      "format": "date-time"
    },
    "measurements": {
      "type": "object",
      "required": ["position", "velocity"],
      "properties": {
        "position": {
          "type": "array",
          "items": { "type": "number" },
          "minItems": 3,
          "maxItems": 3
        },
        "velocity": {
          "type": "array",
          "items": { "type": "number" },
          "minItems": 3,
          "maxItems": 3
        }
      }
    }
  }
}
```

## Rate Limit Categories
| Endpoint Category | Unauthenticated | Basic Auth | Premium |
|------------------|-----------------|------------|---------|
| Health Checks    | 100/min         | 200/min    | 500/min |
| Status Queries   | 20/min          | 100/min    | 300/min |
| Telemetry        | Not Allowed     | 50/min     | 200/min |
| Analysis         | Not Allowed     | 10/min     | 50/min  |

## Versioning
- Current Version: v1
- Version Header: `Accept: application/vnd.astroshield.v1+json`
- Deprecation Notice: Included in response headers when applicable 