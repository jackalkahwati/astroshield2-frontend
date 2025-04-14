# CCDM API Documentation

This document describes the REST API endpoints provided by the Conjunction and Collision Data Management (CCDM) service.

## Authentication

All API requests require authentication using a JWT token. The token should be included in the `Authorization` header:

```
Authorization: Bearer <your_token>
```

To obtain a token, use the `/auth/login` endpoint with your credentials.

## Base URL

```
https://api.ccdm.example.org/v1
```

## Rate Limiting

API requests are subject to rate limiting based on your subscription tier. The following headers are included in API responses:

- `X-RateLimit-Limit`: Maximum requests per minute
- `X-RateLimit-Remaining`: Remaining requests in the current window
- `X-RateLimit-Reset`: Time when the rate limit resets (Unix timestamp)

## Endpoints

### Authentication

#### POST /auth/login

Authenticates a user and returns a JWT token.

**Request Body:**
```json
{
  "username": "string",
  "password": "string"
}
```

**Response:**
```json
{
  "token": "string",
  "expiresAt": "ISO8601 timestamp"
}
```

**Status Codes:**
- 200: Success
- 401: Invalid credentials

### Conjunction Data

#### GET /conjunctions

Retrieves a list of conjunction events based on specified filters.

**Query Parameters:**
- `startTime` (required): ISO8601 timestamp
- `endTime` (required): ISO8601 timestamp
- `minPc` (optional): Minimum collision probability (0.0-1.0)
- `objectId` (optional): Filter by primary object ID
- `limit` (optional): Maximum number of results (default: 100)
- `offset` (optional): Pagination offset (default: 0)

**Response:**
```json
{
  "conjunctions": [
    {
      "id": "string",
      "tca": "ISO8601 timestamp",
      "primaryObject": {
        "id": "string",
        "name": "string",
        "type": "string"
      },
      "secondaryObject": {
        "id": "string",
        "name": "string",
        "type": "string"
      },
      "probabilityOfCollision": "number",
      "missDistance": "number",
      "relativeVelocity": "number",
      "createdAt": "ISO8601 timestamp",
      "updatedAt": "ISO8601 timestamp"
    }
  ],
  "totalCount": "number",
  "limit": "number",
  "offset": "number"
}
```

**Status Codes:**
- 200: Success
- 400: Invalid parameters
- 401: Unauthorized
- 429: Rate limit exceeded

#### GET /conjunctions/{id}

Retrieves detailed information about a specific conjunction event.

**Path Parameters:**
- `id` (required): Conjunction event ID

**Response:**
```json
{
  "id": "string",
  "tca": "ISO8601 timestamp",
  "primaryObject": {
    "id": "string",
    "name": "string",
    "type": "string",
    "noradId": "string",
    "internationalDesignator": "string",
    "orbitalParameters": {
      "epoch": "ISO8601 timestamp",
      "semiMajorAxis": "number",
      "eccentricity": "number",
      "inclination": "number",
      "raan": "number",
      "argumentOfPerigee": "number",
      "meanAnomaly": "number"
    }
  },
  "secondaryObject": {
    "id": "string",
    "name": "string",
    "type": "string",
    "noradId": "string",
    "internationalDesignator": "string",
    "orbitalParameters": {
      "epoch": "ISO8601 timestamp",
      "semiMajorAxis": "number",
      "eccentricity": "number",
      "inclination": "number",
      "raan": "number",
      "argumentOfPerigee": "number",
      "meanAnomaly": "number"
    }
  },
  "probabilityOfCollision": "number",
  "missDistance": "number",
  "relativeVelocity": "number",
  "covariance": {
    "primary": [...],
    "secondary": [...]
  },
  "createdAt": "ISO8601 timestamp",
  "updatedAt": "ISO8601 timestamp"
}
```

**Status Codes:**
- 200: Success
- 404: Conjunction not found
- 401: Unauthorized
- 429: Rate limit exceeded

### Space Objects

#### GET /objects

Retrieves a list of space objects based on specified filters.

**Query Parameters:**
- `type` (optional): Object type (e.g., "PAYLOAD", "ROCKET_BODY", "DEBRIS")
- `name` (optional): Object name (partial match)
- `noradId` (optional): NORAD catalog ID
- `internationalDesignator` (optional): International designator
- `limit` (optional): Maximum number of results (default: 100)
- `offset` (optional): Pagination offset (default: 0)

**Response:**
```json
{
  "objects": [
    {
      "id": "string",
      "name": "string",
      "type": "string",
      "noradId": "string",
      "internationalDesignator": "string",
      "launchDate": "ISO8601 timestamp",
      "createdAt": "ISO8601 timestamp",
      "updatedAt": "ISO8601 timestamp"
    }
  ],
  "totalCount": "number",
  "limit": "number",
  "offset": "number"
}
```

**Status Codes:**
- 200: Success
- 400: Invalid parameters
- 401: Unauthorized
- 429: Rate limit exceeded

#### GET /objects/{id}

Retrieves detailed information about a specific space object.

**Path Parameters:**
- `id` (required): Space object ID

**Response:**
```json
{
  "id": "string",
  "name": "string",
  "type": "string",
  "noradId": "string",
  "internationalDesignator": "string",
  "launchDate": "ISO8601 timestamp",
  "orbitalParameters": {
    "epoch": "ISO8601 timestamp",
    "semiMajorAxis": "number",
    "eccentricity": "number",
    "inclination": "number",
    "raan": "number",
    "argumentOfPerigee": "number",
    "meanAnomaly": "number"
  },
  "physicalParameters": {
    "mass": "number",
    "size": {
      "length": "number",
      "width": "number",
      "height": "number"
    },
    "crossSectionalArea": "number",
    "dragCoefficient": "number"
  },
  "createdAt": "ISO8601 timestamp",
  "updatedAt": "ISO8601 timestamp"
}
```

**Status Codes:**
- 200: Success
- 404: Object not found
- 401: Unauthorized
- 429: Rate limit exceeded

### Maneuver Planning

#### POST /maneuvers/plan

Computes an optimal maneuver plan to mitigate a conjunction event.

**Request Body:**
```json
{
  "conjunctionId": "string",
  "constraints": {
    "maxDeltaV": "number",
    "earliestManeuverTime": "ISO8601 timestamp",
    "latestManeuverTime": "ISO8601 timestamp",
    "targetMinimumMissDistance": "number"
  }
}
```

**Response:**
```json
{
  "id": "string",
  "conjunctionId": "string",
  "status": "string",
  "maneuverOptions": [
    {
      "id": "string",
      "maneuverTime": "ISO8601 timestamp",
      "deltaV": {
        "magnitude": "number",
        "radial": "number",
        "inTrack": "number",
        "crossTrack": "number"
      },
      "postManeuverMissDistance": "number",
      "postManeuverProbabilityOfCollision": "number",
      "fuelConsumption": "number"
    }
  ],
  "recommendedManeuver": "string",
  "createdAt": "ISO8601 timestamp"
}
```

**Status Codes:**
- 200: Success
- 400: Invalid parameters
- 401: Unauthorized
- 404: Conjunction not found
- 429: Rate limit exceeded

### Analytics

#### GET /analytics/conjunctions/summary

Retrieves summary statistics for conjunction events over a specified time period.

**Query Parameters:**
- `startTime` (required): ISO8601 timestamp
- `endTime` (required): ISO8601 timestamp
- `objectId` (optional): Filter by primary object ID
- `interval` (optional): Aggregation interval (daily, weekly, monthly)

**Response:**
```json
{
  "interval": "string",
  "data": [
    {
      "timeBlock": "string",
      "conjunctionCount": "number",
      "highRiskCount": "number",
      "averagePc": "number",
      "averageMissDistance": "number"
    }
  ]
}
```

**Status Codes:**
- 200: Success
- 400: Invalid parameters
- 401: Unauthorized
- 429: Rate limit exceeded

### Notifications

#### GET /notifications/settings

Retrieves the current notification settings for the authenticated user.

**Response:**
```json
{
  "email": {
    "enabled": "boolean",
    "addresses": ["string"],
    "minPc": "number"
  },
  "webhook": {
    "enabled": "boolean",
    "url": "string",
    "secret": "string",
    "minPc": "number"
  },
  "sms": {
    "enabled": "boolean",
    "phoneNumbers": ["string"],
    "minPc": "number"
  }
}
```

**Status Codes:**
- 200: Success
- 401: Unauthorized
- 429: Rate limit exceeded

#### PUT /notifications/settings

Updates notification settings for the authenticated user.

**Request Body:**
```json
{
  "email": {
    "enabled": "boolean",
    "addresses": ["string"],
    "minPc": "number"
  },
  "webhook": {
    "enabled": "boolean",
    "url": "string",
    "secret": "string",
    "minPc": "number"
  },
  "sms": {
    "enabled": "boolean",
    "phoneNumbers": ["string"],
    "minPc": "number"
  }
}
```

**Response:**
```json
{
  "success": "boolean",
  "message": "string"
}
```

**Status Codes:**
- 200: Success
- 400: Invalid parameters
- 401: Unauthorized
- 429: Rate limit exceeded

## Error Responses

All error responses follow this format:

```json
{
  "error": {
    "code": "string",
    "message": "string",
    "details": {}
  }
}
```

Common error codes:
- `AUTHENTICATION_REQUIRED`: Authentication is required
- `INVALID_CREDENTIALS`: Invalid username or password
- `INVALID_TOKEN`: Authentication token is invalid or expired
- `PERMISSION_DENIED`: User does not have permission for the requested operation
- `RESOURCE_NOT_FOUND`: The requested resource was not found
- `INVALID_PARAMETERS`: The request contains invalid parameters
- `RATE_LIMIT_EXCEEDED`: API rate limit has been exceeded
- `INTERNAL_ERROR`: An internal server error occurred 