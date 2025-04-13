# AstroShield API Integration Guide

This guide provides instructions for integrating your application with the AstroShield API. It includes examples in multiple programming languages and best practices for a successful integration.

## API Overview

The AstroShield API is a RESTful service that provides access to satellite data, collision avoidance maneuvers, and space domain awareness analytics. The API uses standard HTTP methods and returns JSON responses.

## Base URL

```
Production: https://api.astroshield.com/api/v1
Staging: https://api-staging.astroshield.com/api/v1
Development: http://localhost:3001/api/v1
```

## Authentication

The AstroShield API uses JWT (JSON Web Token) for authentication. To access protected endpoints, you need to:

1. Obtain an access token by sending your credentials to the `/token` endpoint
2. Include the token in the `Authorization` header of subsequent requests

### Getting an Access Token

#### Request

```http
POST /api/v1/token
Content-Type: application/json

{
  "username": "your_username",
  "password": "your_password"
}
```

#### Response

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_at": "2023-06-22T12:00:00Z"
}
```

### Using the Token

Include the token in the `Authorization` header:

```http
GET /api/v1/satellites
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### Token Expiration

Tokens expire after 24 hours. When a token expires, request a new one using the same authentication process.

## Rate Limiting

The API implements rate limiting to ensure fair usage:

- Authenticated requests: 100 requests per minute
- Unauthenticated requests: 20 requests per minute

Rate limit information is included in the response headers:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1623456789
```

## Error Handling

The API returns standard HTTP status codes and detailed error messages:

```json
{
  "detail": "Satellite with ID 'sat-999' not found",
  "status_code": 404,
  "error_code": "resource_not_found"
}
```

Common error status codes:

- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Missing or invalid authentication
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server-side error

## Pagination

Collection endpoints support pagination with the following query parameters:

- `page`: Page number (default: 1)
- `limit`: Items per page (default: 20, max: 100)

Example:

```http
GET /api/v1/satellites?page=2&limit=50
```

Response includes pagination metadata:

```json
{
  "data": [...],
  "meta": {
    "page": 2,
    "limit": 50,
    "total": 345,
    "pages": 7
  }
}
```

## Integration Examples

### Python

```python
import requests

BASE_URL = "https://api.astroshield.com/api/v1"

def get_token(username, password):
    response = requests.post(
        f"{BASE_URL}/token",
        json={"username": username, "password": password}
    )
    response.raise_for_status()
    return response.json()["access_token"]

def get_satellites(token):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{BASE_URL}/satellites", headers=headers)
    response.raise_for_status()
    return response.json()["data"]

def create_maneuver(token, satellite_id, maneuver_data):
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    response = requests.post(
        f"{BASE_URL}/maneuvers",
        headers=headers,
        json={
            "satellite_id": satellite_id,
            **maneuver_data
        }
    )
    response.raise_for_status()
    return response.json()["data"]

# Example usage
token = get_token("user@example.com", "password123")
satellites = get_satellites(token)
print(f"Found {len(satellites)} satellites")

# Create a maneuver
maneuver = create_maneuver(
    token,
    "sat-001",
    {
        "type": "collision_avoidance",
        "scheduled_start_time": "2023-06-15T20:00:00Z",
        "parameters": {
            "delta_v": 0.02,
            "burn_duration": 15.0,
            "direction": {"x": 0.1, "y": 0.0, "z": -0.1}
        }
    }
)
print(f"Created maneuver: {maneuver['id']}")
```

### JavaScript

```javascript
const API_BASE_URL = 'https://api.astroshield.com/api/v1';

async function getToken(username, password) {
  const response = await fetch(`${API_BASE_URL}/token`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      username,
      password,
    }),
  });
  
  if (!response.ok) {
    throw new Error(`Authentication failed: ${response.statusText}`);
  }
  
  const data = await response.json();
  return data.access_token;
}

async function getSatellites(token) {
  const response = await fetch(`${API_BASE_URL}/satellites`, {
    headers: {
      'Authorization': `Bearer ${token}`,
    },
  });
  
  if (!response.ok) {
    throw new Error(`Failed to fetch satellites: ${response.statusText}`);
  }
  
  const data = await response.json();
  return data.data;
}

async function createManeuver(token, satelliteId, maneuverData) {
  const response = await fetch(`${API_BASE_URL}/maneuvers`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      satellite_id: satelliteId,
      ...maneuverData,
    }),
  });
  
  if (!response.ok) {
    throw new Error(`Failed to create maneuver: ${response.statusText}`);
  }
  
  const data = await response.json();
  return data.data;
}

// Example usage
async function example() {
  try {
    const token = await getToken('user@example.com', 'password123');
    
    const satellites = await getSatellites(token);
    console.log(`Found ${satellites.length} satellites`);
    
    const maneuver = await createManeuver(token, 'sat-001', {
      type: 'collision_avoidance',
      scheduled_start_time: '2023-06-15T20:00:00Z',
      parameters: {
        delta_v: 0.02,
        burn_duration: 15.0,
        direction: { x: 0.1, y: 0.0, z: -0.1 }
      }
    });
    
    console.log(`Created maneuver: ${maneuver.id}`);
  } catch (error) {
    console.error('Integration error:', error);
  }
}

example();
```

### Java

```java
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import org.json.JSONObject;
import org.json.JSONArray;

public class AstroShieldClient {
    private static final String API_BASE_URL = "https://api.astroshield.com/api/v1";
    private final HttpClient httpClient;
    private String token;

    public AstroShieldClient() {
        this.httpClient = HttpClient.newBuilder()
                .version(HttpClient.Version.HTTP_2)
                .connectTimeout(Duration.ofSeconds(10))
                .build();
    }

    public void authenticate(String username, String password) throws IOException, InterruptedException {
        JSONObject authRequest = new JSONObject();
        authRequest.put("username", username);
        authRequest.put("password", password);

        HttpRequest request = HttpRequest.newBuilder()
                .POST(HttpRequest.BodyPublishers.ofString(authRequest.toString()))
                .uri(URI.create(API_BASE_URL + "/token"))
                .header("Content-Type", "application/json")
                .build();

        HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());

        if (response.statusCode() != 200) {
            throw new IOException("Authentication failed: " + response.statusCode());
        }

        JSONObject jsonResponse = new JSONObject(response.body());
        this.token = jsonResponse.getString("access_token");
    }

    public JSONArray getSatellites() throws IOException, InterruptedException {
        HttpRequest request = HttpRequest.newBuilder()
                .GET()
                .uri(URI.create(API_BASE_URL + "/satellites"))
                .header("Authorization", "Bearer " + token)
                .build();

        HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());

        if (response.statusCode() != 200) {
            throw new IOException("Failed to fetch satellites: " + response.statusCode());
        }

        JSONObject jsonResponse = new JSONObject(response.body());
        return jsonResponse.getJSONArray("data");
    }

    public JSONObject createManeuver(String satelliteId, JSONObject maneuverData) 
            throws IOException, InterruptedException {
        
        maneuverData.put("satellite_id", satelliteId);

        HttpRequest request = HttpRequest.newBuilder()
                .POST(HttpRequest.BodyPublishers.ofString(maneuverData.toString()))
                .uri(URI.create(API_BASE_URL + "/maneuvers"))
                .header("Content-Type", "application/json")
                .header("Authorization", "Bearer " + token)
                .build();

        HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());

        if (response.statusCode() != 201) {
            throw new IOException("Failed to create maneuver: " + response.statusCode());
        }

        JSONObject jsonResponse = new JSONObject(response.body());
        return jsonResponse.getJSONObject("data");
    }

    public static void main(String[] args) {
        try {
            AstroShieldClient client = new AstroShieldClient();
            client.authenticate("user@example.com", "password123");

            JSONArray satellites = client.getSatellites();
            System.out.println("Found " + satellites.length() + " satellites");

            // Create a maneuver
            JSONObject parameters = new JSONObject();
            parameters.put("delta_v", 0.02);
            parameters.put("burn_duration", 15.0);
            
            JSONObject direction = new JSONObject();
            direction.put("x", 0.1);
            direction.put("y", 0.0);
            direction.put("z", -0.1);
            parameters.put("direction", direction);

            JSONObject maneuverData = new JSONObject();
            maneuverData.put("type", "collision_avoidance");
            maneuverData.put("scheduled_start_time", "2023-06-15T20:00:00Z");
            maneuverData.put("parameters", parameters);

            JSONObject maneuver = client.createManeuver("sat-001", maneuverData);
            System.out.println("Created maneuver: " + maneuver.getString("id"));

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
```

## Best Practices

### Error Handling

Always implement robust error handling in your integration:

```python
try:
    # API call
except requests.exceptions.HTTPError as http_err:
    if http_err.response.status_code == 401:
        # Handle authentication errors
    elif http_err.response.status_code == 404:
        # Handle resource not found
    else:
        # Handle other HTTP errors
except requests.exceptions.ConnectionError:
    # Handle connection issues
except requests.exceptions.Timeout:
    # Handle timeout issues
except requests.exceptions.RequestException as err:
    # Handle any other request issues
```

### Token Management

- Store tokens securely
- Implement token refresh before expiration
- Don't embed tokens in client-side code

### Connection Pooling

For high-volume integrations, use connection pooling:

```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

session = requests.Session()
retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=20))
```

### Pagination

Handle pagination properly for large collections:

```python
def get_all_satellites(token):
    all_satellites = []
    page = 1
    more_pages = True
    
    while more_pages:
        response = requests.get(
            f"{BASE_URL}/satellites",
            headers={"Authorization": f"Bearer {token}"},
            params={"page": page, "limit": 100}
        )
        response.raise_for_status()
        result = response.json()
        
        all_satellites.extend(result["data"])
        
        if page >= result["meta"]["pages"]:
            more_pages = False
        else:
            page += 1
    
    return all_satellites
```

### Rate Limiting

Respect rate limits by implementing exponential backoff:

```python
def make_request_with_backoff(url, headers, max_retries=5):
    for attempt in range(max_retries):
        response = requests.get(url, headers=headers)
        
        if response.status_code != 429:  # Not rate limited
            return response
        
        # Get retry-after header or use exponential backoff
        retry_after = int(response.headers.get('Retry-After', 2 ** attempt))
        time.sleep(retry_after)
    
    # If we get here, we've exceeded max retries
    response.raise_for_status()
```

## Webhooks

AstroShield API supports webhooks for real-time notifications. Configure webhooks in your account settings to receive events for:

- Satellite status changes
- Maneuver status updates
- Collision warnings
- System notifications

Example webhook payload:

```json
{
  "event": "maneuver.completed",
  "timestamp": "2023-06-15T20:30:00Z",
  "data": {
    "maneuver_id": "mnv-001",
    "satellite_id": "sat-001",
    "status": "completed",
    "execution_time": 180,
    "success": true
  }
}
```

To verify webhook authenticity, check the `X-AstroShield-Signature` header against an HMAC of the request body using your webhook secret.

## API Client Libraries

Official client libraries are available for:

- Python: `pip install astroshield-client`
- JavaScript/Node.js: `npm install astroshield-client`
- Java: Available on Maven Central as `com.astroshield:astroshield-client`

## Migration Guide

### Migrating from v0 to v1

- The base URL has changed from `/api` to `/api/v1`
- Authentication now uses JWT instead of API keys
- Response format has been standardized with `data` and `meta` fields
- Pagination parameters have changed from `offset/limit` to `page/limit`

## Support

For integration support:

- Documentation: https://docs.astroshield.com
- API Status: https://status.astroshield.com
- Email: support@astroshield.com
- Developer Forum: https://community.astroshield.com 