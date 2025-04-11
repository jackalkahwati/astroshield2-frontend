# AstroShield API Documentation Guide

This guide will help you understand how to use the interactive API documentation for the AstroShield API.

## Accessing the Documentation

The API documentation is available at the following URLs:

- Documentation Landing Page: `/api/v1/documentation`
- Swagger UI: `/api/v1/docs`
- ReDoc: `/api/v1/redoc`
- OpenAPI Specification: `/api/v1/openapi.json`

## Using Swagger UI

Swagger UI provides an interactive documentation interface that allows you to:

1. Browse API endpoints organized by tags
2. Understand request parameters and response formats
3. Test API endpoints directly from your browser

![Swagger UI Overview](../assets/images/swagger-ui-overview.png)
*(Image placeholder: Screenshot of AstroShield Swagger UI)*

### Authentication

AstroShield API supports two authentication methods:

#### Option 1: Bearer Token Authentication

For protected endpoints, you can authenticate using JWT tokens:

1. Navigate to the `/api/v1/token` endpoint in the "auth" section
2. Click "Try it out" and enter your username and password:

```json
{
  "username": "your_username",
  "password": "your_password"
}
```

3. Execute the request to get a token
4. Copy the access token from the response:

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_at": "2024-03-26T15:30:45.123456"
}
```

5. Click the "Authorize" button at the top of the page
6. Enter your token in the format `Bearer <your_token>` in the "Value" field under the "bearerAuth" section
7. Click "Authorize"

#### Option 2: API Key Authentication

For easier testing, you can also use an API key:

1. Click the "Authorize" button at the top of the page
2. Enter the API key `d8533cd1-a315-408f-bf4f-fcd898863daf` in the "Value" field under the "apiKeyAuth" section
3. Click "Authorize"

![Authentication Flow](../assets/images/swagger-auth-flow.png)
*(Image placeholder: Screenshot of authentication workflow)*

Now your requests will include the authentication credentials.

### Testing Endpoints

To test an endpoint:

1. Navigate to the endpoint you want to test
2. Click "Try it out"
3. Fill in the required parameters
4. Click "Execute"
5. Review the response

#### Example: Creating a New Satellite

Here's an example of creating a new satellite using the Swagger UI:

1. Navigate to the `/api/v1/satellites` POST endpoint
2. Click "Try it out"
3. Enter the following JSON in the request body:

```json
{
  "name": "AstroShield-1",
  "norad_id": "43657",
  "international_designator": "2018-099A",
  "orbit_type": "LEO",
  "launch_date": "2023-05-15T00:00:00Z",
  "operational_status": "active",
  "owner": "AstroShield Inc.",
  "mission": "Space Domain Awareness",
  "orbital_parameters": {
    "semi_major_axis": 7000,
    "eccentricity": 0.0001,
    "inclination": 51.6,
    "raan": 235.7,
    "argument_of_perigee": 90.0,
    "mean_anomaly": 0.0,
    "epoch": "2023-05-15T12:00:00Z"
  }
}
```

4. Click "Execute"
5. You should receive a successful response with a 201 status code and the created satellite details.

### Using Models

Swagger UI provides detailed information about the data models used in the API:

1. Scroll down to the "Schemas" section or click on a schema link in an endpoint
2. Explore the model structure, required fields, and data types
3. Use the "Model" and "Example Value" toggles to view different representations

![Data Models](../assets/images/swagger-models.png)
*(Image placeholder: Screenshot of models section)*

### Filter and Search

Swagger UI includes powerful filtering capabilities:

1. Use the filter box at the top to search for specific endpoints or tags
2. Filter by tags using the tag list on the left side
3. Use the "Deep Linking" feature by adding the endpoint name to the URL after a hash, e.g., `/api/v1/docs#/satellites/get_satellites`

## Using ReDoc

ReDoc provides a more reader-friendly documentation interface that is better for:

1. Reading API documentation
2. Understanding complex request and response schemas
3. Navigating through a large API

![ReDoc Overview](../assets/images/redoc-overview.png)
*(Image placeholder: Screenshot of AstroShield ReDoc UI)*

ReDoc features:

- Table of contents navigation
- Request/response examples
- Schema definitions with all properties
- Search functionality
- Responsive design for mobile devices

Unlike Swagger UI, ReDoc doesn't allow you to test endpoints directly.

## Programmatic API Access

### Using cURL

Here are examples of accessing the API using cURL:

**Authentication with username/password:**
```bash
curl -X POST http://localhost:3001/api/v1/token \
  -H "Content-Type: application/json" \
  -d '{"username": "your_username", "password": "your_password"}'
```

**Authentication with API key:**
```bash
curl -X GET http://localhost:3001/api/v1/satellites \
  -H "X-API-Key: d8533cd1-a315-408f-bf4f-fcd898863daf"
```

**Retrieving Satellites with Bearer Token:**
```bash
curl -X GET http://localhost:3001/api/v1/satellites \
  -H "Authorization: Bearer your_token_here"
```

**Creating a Maneuver:**
```bash
curl -X POST http://localhost:3001/api/v1/maneuvers \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_token_here" \
  -d '{
    "satellite_id": "sat-001",
    "type": "collision_avoidance",
    "scheduled_start_time": "2023-06-15T20:00:00Z",
    "parameters": {
      "delta_v": 0.02,
      "burn_duration": 15.0,
      "direction": {"x": 0.1, "y": 0.0, "z": -0.1}
    }
  }'
```

### Using Python

Here's an example of accessing the API using Python requests:

#### Token Authentication

```python
import requests
import json

# Authentication
auth_url = "http://localhost:3001/api/v1/token"
auth_payload = {
    "username": "your_username",
    "password": "your_password"
}
auth_response = requests.post(auth_url, json=auth_payload)
token = auth_response.json()["access_token"]

# Set up headers with authentication token
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

# Get satellites
satellites_url = "http://localhost:3001/api/v1/satellites"
response = requests.get(satellites_url, headers=headers)
satellites = response.json()

print(f"Retrieved {len(satellites)} satellites")
```

#### API Key Authentication

```python
import requests

# Set up headers with API key
headers = {
    "X-API-Key": "d8533cd1-a315-408f-bf4f-fcd898863daf",
    "Content-Type": "application/json"
}

# Get satellites
satellites_url = "http://localhost:3001/api/v1/satellites"
response = requests.get(satellites_url, headers=headers)
satellites = response.json()

print(f"Retrieved {len(satellites)} satellites")
```

### Using JavaScript

Here's an example of accessing the API using JavaScript fetch:

#### Token Authentication

```javascript
// Authentication
async function getToken() {
  const response = await fetch('http://localhost:3001/api/v1/token', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      username: 'your_username',
      password: 'your_password'
    })
  });
  
  const data = await response.json();
  return data.access_token;
}

// Use the API with authentication
async function getSatellites() {
  const token = await getToken();
  
  const response = await fetch('http://localhost:3001/api/v1/satellites', {
    method: 'GET',
    headers: {
      'Authorization': `Bearer ${token}`
    }
  });
  
  const satellites = await response.json();
  console.log(`Retrieved ${satellites.length} satellites`);
  return satellites;
}
```

#### API Key Authentication

```javascript
// Use the API with API key
async function getSatellitesWithApiKey() {
  const response = await fetch('http://localhost:3001/api/v1/satellites', {
    method: 'GET',
    headers: {
      'X-API-Key': 'd8533cd1-a315-408f-bf4f-fcd898863daf'
    }
  });
  
  const satellites = await response.json();
  console.log(`Retrieved ${satellites.length} satellites`);
  return satellites;
}
```

## API Structure

The AstroShield API is organized into the following sections:

- **Health**: Basic health check endpoints
- **Analytics**: Data analysis and reporting
- **Maneuvers**: Satellite maneuver planning and execution
- **Satellites**: Satellite management and information
- **Advanced**: Advanced operations and analysis
- **Dashboard**: User dashboard data
- **CCDM**: Concealment, Camouflage, Deception, and Maneuvering capabilities
- **Trajectory**: Trajectory analysis and planning
- **Comparison**: Data comparison operations
- **Events**: Event management and tracking
- **Auth**: Authentication endpoints

## Status Codes

The API uses standard HTTP status codes to indicate success or failure:

| Code | Description | Example |
|------|-------------|---------|
| 200 | OK - The request was successful | GET /api/v1/satellites |
| 201 | Created - Resource created successfully | POST /api/v1/satellites |
| 204 | No Content - Request successful, no content returned | DELETE /api/v1/satellites/{id} |
| 400 | Bad Request - Invalid parameters or data | Missing required fields |
| 401 | Unauthorized - Authentication required | Missing or invalid token |
| 403 | Forbidden - Insufficient permissions | Trying to access admin endpoint |
| 404 | Not Found - Resource doesn't exist | Invalid satellite ID |
| 409 | Conflict - Resource already exists | Duplicate satellite name |
| 429 | Too Many Requests - Rate limit exceeded | Exceeded API rate limits |
| 500 | Internal Server Error - Something went wrong | Server-side error |

## OpenAPI Specification

The complete OpenAPI specification is available at `/api/v1/openapi.json`. You can use this specification with:

- API client generators
- Testing tools
- CI/CD pipelines
- Other API documentation systems

### Generating Client Libraries

You can use the OpenAPI specification to generate client libraries for various languages:

1. Download the OpenAPI specification from `/api/v1/openapi.json`
2. Use a tool like [OpenAPI Generator](https://openapi-generator.tech/) to generate client code:

```bash
# Install OpenAPI Generator
npm install @openapitools/openapi-generator-cli -g

# Generate a Python client
openapi-generator-cli generate -i openapi.json -g python -o ./generated-client

# Generate a TypeScript client
openapi-generator-cli generate -i openapi.json -g typescript-fetch -o ./typescript-client
```

## Rate Limiting

The API implements rate limiting to prevent abuse:

- 1000 requests per hour for authenticated users
- 60 requests per hour for unauthenticated users

Rate limit headers are included in all responses:

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 998
X-RateLimit-Reset: 3600
```

## Troubleshooting

If you encounter issues with the API documentation:

1. Make sure you're using a modern browser (Chrome, Firefox, Edge, or Safari)
2. Clear your browser cache and reload the page
3. Check your authentication token if you're getting 401 Unauthorized responses
4. Ensure your request payload matches the expected schema
5. Check that your token hasn't expired, and request a new one if needed
6. Look for detailed error messages in the response body
7. Contact the API support team if you continue to have issues

## Common Errors and Solutions

| Error | Possible Cause | Solution |
|-------|----------------|----------|
| 401 Unauthorized | Invalid or expired token | Re-authenticate to get a new token |
| 400 Bad Request | Invalid request body | Check your request against the schema |
| 403 Forbidden | Insufficient permissions | Request access to the required role |
| 404 Not Found | Invalid resource ID | Verify the ID is correct |
| 429 Too Many Requests | Rate limit exceeded | Wait until rate limit resets |

## Support

For questions or support with the AstroShield API, please contact the development team at support@astroshield.com. 