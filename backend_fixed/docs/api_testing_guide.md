# AstroShield API Testing Guide

This guide provides instructions for testing the AstroShield API, including manual testing with tools like Postman, automated testing, and troubleshooting common issues.

## Test Environments

AstroShield provides several environments for testing:

| Environment | Base URL | Purpose |
|-------------|----------|---------|
| Development | http://localhost:3001/api/v1 | Local development testing |
| Testing | https://api-test.astroshield.com/api/v1 | CI/CD automated tests |
| Staging | https://api-staging.astroshield.com/api/v1 | Pre-production testing |
| Production | https://api.astroshield.com/api/v1 | Production verification |

## Testing with Postman

The [AstroShield Postman Collection](../postman_collection.json) provides a pre-configured set of API requests for testing.

### Setup

1. Import the Postman collection:
   - Open Postman
   - Click "Import" and select the collection file
   - The collection includes environment variables for different environments

2. Configure environment variables:
   - Select the appropriate environment (Development, Staging, Production)
   - Update `baseUrl` to match your test environment
   - Set `username` and `password` for authentication

3. Get an access token:
   - Run the "Get Access Token" request in the Authentication folder
   - The token will be automatically saved to the `accessToken` variable

### Testing Workflow

Follow this workflow for comprehensive testing:

1. **Authentication**
   - Test token acquisition with valid credentials
   - Test token acquisition with invalid credentials
   - Verify token expiration behavior

2. **Satellite Operations**
   - List satellites (verify pagination)
   - Get a specific satellite
   - Create a new satellite
   - Update a satellite
   - Test error handling for non-existent satellites

3. **Maneuver Operations**
   - List maneuvers for a satellite
   - Create a new maneuver
   - Update a maneuver
   - Cancel a maneuver
   - Simulate a maneuver
   - Test various maneuver types

4. **Analytics**
   - Test dashboard analytics
   - Test satellite-specific analytics

5. **System Operations**
   - Health check
   - System information

### Testing Response Status Codes

Verify appropriate status codes for different scenarios:

| Scenario | Expected Status Code |
|----------|---------------------|
| Successful GET | 200 OK |
| Successful POST | 201 Created |
| Successful PUT/PATCH | 200 OK |
| Successful DELETE | 204 No Content |
| Invalid request | 400 Bad Request |
| Unauthorized access | 401 Unauthorized |
| Forbidden action | 403 Forbidden |
| Resource not found | 404 Not Found |
| Rate limit exceeded | 429 Too Many Requests |
| Server error | 500 Internal Server Error |

## Automated Testing

### API Integration Tests

The repository includes automated tests using pytest for API integration testing.

To run the tests:

```bash
# Set up test environment
export TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/astroshield_test
export TEST_SECRET_KEY=test_secret_key

# Run the tests
pytest backend/app/tests/integration/
```

### Performance Testing with Locust

We use Locust for performance testing. The Locust file is available in `backend/app/tests/performance/locustfile.py`.

To run a performance test:

```bash
cd backend/app/tests/performance
locust -f locustfile.py --host=https://api-staging.astroshield.com
```

Then open http://localhost:8089 in your browser to configure and start the test.

### Security Testing

We recommend using OWASP ZAP for security testing of the API. A pre-configured ZAP context file is available in `backend/app/tests/security/astroshield-api.context`.

## Common Test Scenarios

### Authentication Tests

- **Valid Credentials**: Should return a token
- **Invalid Credentials**: Should return 401 Unauthorized
- **Missing Credentials**: Should return 400 Bad Request
- **Expired Token**: Should return 401 Unauthorized
- **Invalid Token Format**: Should return 401 Unauthorized

### Satellite Tests

- **Create Valid Satellite**: Should return 201 Created with satellite data
- **Create Invalid Satellite** (missing required fields): Should return 400 Bad Request
- **Create Duplicate Satellite** (same NORAD ID): Should return 409 Conflict
- **Update Satellite**: Should return 200 OK with updated data
- **Get Non-existent Satellite**: Should return 404 Not Found

### Maneuver Tests

- **Create Valid Maneuver**: Should return 201 Created with maneuver data
- **Create Maneuver for Non-existent Satellite**: Should return 404 Not Found
- **Create Invalid Maneuver** (invalid parameters): Should return 400 Bad Request
- **Update Maneuver**: Should return 200 OK with updated data
- **Cancel Maneuver**: Should return 200 OK with updated status
- **Cancel Completed Maneuver**: Should return 400 Bad Request
- **Create Maneuver with Insufficient Resources**: Should return 422 Unprocessable Entity

### Rate Limiting Tests

- **Exceed Rate Limit**: Send more than the allowed requests per minute, should return 429 Too Many Requests
- **Check Rate Limit Headers**: Verify `X-RateLimit-*` headers are present and accurate

## Testing with cURL

### Basic Authentication

```bash
# Get token
curl -X POST https://api.astroshield.com/api/v1/token \
  -H "Content-Type: application/json" \
  -d '{"username":"your_username","password":"your_password"}'

# Store token in a variable
TOKEN=$(curl -s -X POST https://api.astroshield.com/api/v1/token \
  -H "Content-Type: application/json" \
  -d '{"username":"your_username","password":"your_password"}' | jq -r .access_token)
```

### Satellite Operations

```bash
# List satellites
curl -X GET https://api.astroshield.com/api/v1/satellites \
  -H "Authorization: Bearer $TOKEN"

# Get specific satellite
curl -X GET https://api.astroshield.com/api/v1/satellites/sat-001 \
  -H "Authorization: Bearer $TOKEN"

# Create satellite
curl -X POST https://api.astroshield.com/api/v1/satellites \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "AstroShield-1",
    "norad_id": "43657",
    "international_designator": "2018-099A",
    "orbit_type": "LEO",
    "launch_date": "2023-05-15T00:00:00Z",
    "operational_status": "active",
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
    }
  }'
```

### Maneuver Operations

```bash
# List maneuvers
curl -X GET https://api.astroshield.com/api/v1/maneuvers?satellite_id=sat-001 \
  -H "Authorization: Bearer $TOKEN"

# Create maneuver
curl -X POST https://api.astroshield.com/api/v1/maneuvers \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "satellite_id": "sat-001",
    "type": "collision_avoidance",
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
    }
  }'
```

## Data Validation Tests

For each API endpoint, test the following validation scenarios:

- **Required fields**: Omit required fields to verify validation
- **Field types**: Provide incorrect data types (string instead of number, etc.)
- **Field ranges**: Test minimum/maximum values and out-of-range values
- **String formats**: Test invalid formats for dates, IDs, etc.
- **Enumerated values**: Test invalid enum values

## Error Response Tests

Verify error responses include:

- HTTP status code
- Error message
- Error code
- Additional details where applicable

Example validation:

```javascript
// Example error response check in JavaScript/Jest
test('Invalid satellite data returns proper error', async () => {
  const response = await fetch(`${API_BASE_URL}/satellites`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      // Missing required fields
      "name": "Test Satellite"
    }),
  });
  
  expect(response.status).toBe(400);
  
  const errorData = await response.json();
  expect(errorData).toHaveProperty('detail');
  expect(errorData).toHaveProperty('error_code');
  expect(errorData.error_code).toBe('validation_error');
});
```

## End-to-End Test Scenarios

1. **Satellite Lifecycle**
   - Create a satellite
   - Retrieve the satellite
   - Update the satellite
   - Create a maneuver for the satellite
   - Execute the maneuver
   - Verify new orbital parameters

2. **Collision Avoidance Workflow**
   - Create two satellites on collision course
   - Detect collision (via analytics API)
   - Create avoidance maneuver
   - Simulate maneuver
   - Execute maneuver
   - Verify collision avoided

## Troubleshooting

### Common Issues

#### Authentication Failures

- **Issue**: 401 Unauthorized when using a token
- **Check**: Token expiration, correct token format, token in the correct header format

#### Request Validation Errors

- **Issue**: 400 Bad Request with validation errors
- **Check**: Request payload structure, required fields, data types, field formats

#### Rate Limiting

- **Issue**: 429 Too Many Requests
- **Check**: Rate limit headers, implement backoff strategy

#### Server Errors

- **Issue**: 500 Internal Server Error
- **Check**: API logs, request payload, try simplified request

### Debugging Tips

1. Use verbose logging in API clients:
   ```python
   import requests
   import logging
   
   # Set up logging
   logging.basicConfig(level=logging.DEBUG)
   logging.getLogger("requests").setLevel(logging.DEBUG)
   logging.getLogger("urllib3").setLevel(logging.DEBUG)
   ```

2. Inspect full request/response details:
   ```bash
   # Using curl with verbose output
   curl -v -X GET https://api.astroshield.com/api/v1/satellites \
     -H "Authorization: Bearer $TOKEN"
   ```

3. Use API debugging proxies like Charles Proxy or Fiddler to inspect API traffic

## Testing Checklist

- [ ] Authentication flows
- [ ] CRUD operations for all resources
- [ ] Error handling for all endpoints
- [ ] Rate limiting behavior
- [ ] Pagination of collection endpoints
- [ ] Search and filtering functionality
- [ ] Data validation for all inputs
- [ ] Performance under expected load
- [ ] Security (authentication, authorization, input validation)
- [ ] Logging and monitoring

## Additional Resources

- [Swagger Documentation](https://api.astroshield.com/docs)
- [Postman Collection](../postman_collection.json)
- [API Integration Guide](./integration_guide.md)
- [API Development Guide](./api_development_guide.md) 