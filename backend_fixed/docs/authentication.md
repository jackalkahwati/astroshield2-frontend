# AstroShield API Authentication Guide

This guide explains how to authenticate with the AstroShield API, manage tokens, and implement secure authentication in your applications.

## Authentication Overview

The AstroShield API uses JSON Web Tokens (JWT) for authentication. To access protected endpoints, you need to:

1. Obtain an access token by authenticating with your credentials
2. Include the token in the `Authorization` header of subsequent requests
3. Refresh the token before it expires to maintain continuous access

## Getting an Access Token

To obtain an access token, send a POST request to the `/api/v1/token` endpoint with your credentials:

### Request

```http
POST /api/v1/token
Content-Type: application/json

{
  "username": "your_username",
  "password": "your_password"
}
```

### Response

```http
HTTP/1.1 200 OK
Content-Type: application/json

{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyQGV4YW1wbGUuY29tIiwiaWF0IjoxNTE2MjM5MDIyLCJleHAiOjE1MTYyMzk5MjJ9.L8aTFyMXg9_PqJ-OlbSRqvR7qlC8LROrJ_xkFpN0C1A",
  "token_type": "bearer",
  "expires_at": "2023-06-22T12:00:00Z"
}
```

### Python Example

```python
import requests

def get_token(username, password):
    response = requests.post(
        "https://api.astroshield.com/api/v1/token",
        json={"username": username, "password": password}
    )
    response.raise_for_status()
    return response.json()["access_token"]

token = get_token("user@example.com", "password123")
```

### JavaScript Example

```javascript
async function getToken(username, password) {
  const response = await fetch('https://api.astroshield.com/api/v1/token', {
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

// Usage
getToken('user@example.com', 'password123')
  .then(token => console.log('Token:', token))
  .catch(error => console.error('Error:', error));
```

## Using the Access Token

Once you have an access token, include it in the `Authorization` header of your API requests:

```http
GET /api/v1/satellites
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### Python Example

```python
import requests

def get_satellites(token):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(
        "https://api.astroshield.com/api/v1/satellites",
        headers=headers
    )
    response.raise_for_status()
    return response.json()["data"]

satellites = get_satellites(token)
```

### JavaScript Example

```javascript
async function getSatellites(token) {
  const response = await fetch('https://api.astroshield.com/api/v1/satellites', {
    headers: {
      'Authorization': `Bearer ${token}`,
    },
  });
  
  if (!response.ok) {
    throw new Error(`Request failed: ${response.statusText}`);
  }
  
  const data = await response.json();
  return data.data;
}
```

## Token Expiration and Refresh

Access tokens expire after 24 hours. The expiration time is included in the token response as `expires_at`.

### Checking the Current User

You can verify that your token is valid by calling the `/api/v1/users/me` endpoint:

```http
GET /api/v1/users/me
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

Response:

```json
{
  "id": "usr-001",
  "username": "user@example.com",
  "email": "user@example.com",
  "full_name": "John Doe",
  "organization": "AstroShield Inc.",
  "role": "OPERATOR",
  "is_active": true
}
```

### Refreshing a Token

To refresh a token before it expires, use the `/api/v1/token/refresh` endpoint:

```http
POST /api/v1/token/refresh
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

Response:

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_at": "2023-06-23T12:00:00Z"
}
```

## Token Management Best Practices

### Secure Storage

Store tokens securely:

- **Server-side applications**: Use environment variables or a secure credential store
- **Browser applications**: Use secure HTTP-only cookies or secure storage mechanisms
- **Mobile applications**: Use secure storage APIs (Keychain for iOS, Keystore for Android)

### Automatic Refresh

Implement automatic token refresh to ensure continuous access:

```python
import time
import requests
from datetime import datetime, timedelta

class TokenManager:
    def __init__(self, username, password, api_base_url):
        self.username = username
        self.password = password
        self.api_base_url = api_base_url
        self.token = None
        self.expires_at = None
        self.refresh_token()
    
    def refresh_token(self):
        response = requests.post(
            f"{self.api_base_url}/token",
            json={"username": self.username, "password": self.password}
        )
        response.raise_for_status()
        data = response.json()
        self.token = data["access_token"]
        self.expires_at = datetime.fromisoformat(data["expires_at"].replace('Z', '+00:00'))
    
    def get_valid_token(self):
        # Refresh if token is expired or will expire in the next 5 minutes
        if not self.expires_at or self.expires_at - timedelta(minutes=5) < datetime.now(self.expires_at.tzinfo):
            self.refresh_token()
        return self.token
```

### JavaScript Implementation

```javascript
class TokenManager {
  constructor(username, password, apiBaseUrl) {
    this.username = username;
    this.password = password;
    this.apiBaseUrl = apiBaseUrl;
    this.token = null;
    this.expiresAt = null;
  }
  
  async refreshToken() {
    const response = await fetch(`${this.apiBaseUrl}/token`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        username: this.username,
        password: this.password,
      }),
    });
    
    if (!response.ok) {
      throw new Error(`Token refresh failed: ${response.statusText}`);
    }
    
    const data = await response.json();
    this.token = data.access_token;
    this.expiresAt = new Date(data.expires_at);
  }
  
  async getValidToken() {
    // Refresh if token is expired or will expire in the next 5 minutes
    if (!this.expiresAt || this.expiresAt - 5 * 60 * 1000 < Date.now()) {
      await this.refreshToken();
    }
    return this.token;
  }
}
```

## Authentication Errors

When authentication fails, the API will return appropriate error responses:

### Invalid Credentials

```http
HTTP/1.1 401 Unauthorized
Content-Type: application/json

{
  "detail": "Incorrect username or password",
  "status_code": 401,
  "error_code": "invalid_credentials"
}
```

### Missing Token

```http
HTTP/1.1 401 Unauthorized
Content-Type: application/json

{
  "detail": "Not authenticated",
  "status_code": 401,
  "error_code": "not_authenticated"
}
```

### Invalid Token

```http
HTTP/1.1 401 Unauthorized
Content-Type: application/json

{
  "detail": "Invalid authentication credentials",
  "status_code": 401,
  "error_code": "invalid_token"
}
```

### Expired Token

```http
HTTP/1.1 401 Unauthorized
Content-Type: application/json

{
  "detail": "Token has expired",
  "status_code": 401,
  "error_code": "token_expired"
}
```

## Authentication for Specific Use Cases

### Server-to-Server Integration

For server-to-server integration, we recommend:

1. Creating a dedicated API user with appropriate permissions
2. Using environment variables to store credentials
3. Implementing automatic token refresh
4. Setting up request retry logic for authentication failures

```python
import os
import requests

# Load credentials from environment variables
API_USERNAME = os.environ.get("ASTROSHIELD_API_USERNAME")
API_PASSWORD = os.environ.get("ASTROSHIELD_API_PASSWORD")
API_BASE_URL = os.environ.get("ASTROSHIELD_API_URL", "https://api.astroshield.com/api/v1")

# Initialize token manager
token_manager = TokenManager(API_USERNAME, API_PASSWORD, API_BASE_URL)

def make_api_request(method, endpoint, data=None):
    token = token_manager.get_valid_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    url = f"{API_BASE_URL}/{endpoint}"
    response = requests.request(method, url, headers=headers, json=data)
    
    # Handle token expiration
    if response.status_code == 401 and "token_expired" in response.text:
        token_manager.refresh_token()
        token = token_manager.get_valid_token()
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.request(method, url, headers=headers, json=data)
    
    response.raise_for_status()
    return response.json()
```

### Web Applications

For web applications, consider:

1. Storing tokens in HTTP-only cookies or secure client-side storage
2. Implementing a token refresh mechanism on the client
3. Using a centralized auth service to manage tokens

```javascript
// Example using axios in a web application
import axios from 'axios';

// Create an axios instance for API requests
const api = axios.create({
  baseURL: 'https://api.astroshield.com/api/v1',
});

// Add a request interceptor to include the token
api.interceptors.request.use(async (config) => {
  // Get token from storage
  const token = localStorage.getItem('astroshield_token');
  const expiresAt = localStorage.getItem('astroshield_token_expires');
  
  // Check if token is about to expire
  if (token && expiresAt && new Date(expiresAt) - 5 * 60 * 1000 < Date.now()) {
    // Token is about to expire, refresh it
    try {
      const response = await axios.post('https://api.astroshield.com/api/v1/token/refresh', {}, {
        headers: { Authorization: `Bearer ${token}` }
      });
      
      // Save new token
      localStorage.setItem('astroshield_token', response.data.access_token);
      localStorage.setItem('astroshield_token_expires', response.data.expires_at);
      
      // Use new token for this request
      config.headers.Authorization = `Bearer ${response.data.access_token}`;
    } catch (error) {
      // Refresh failed, redirect to login
      localStorage.removeItem('astroshield_token');
      localStorage.removeItem('astroshield_token_expires');
      window.location.href = '/login';
    }
  } else if (token) {
    // Use existing token
    config.headers.Authorization = `Bearer ${token}`;
  }
  
  return config;
});

// Add a response interceptor to handle 401 errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response && error.response.status === 401) {
      // Clear tokens and redirect to login
      localStorage.removeItem('astroshield_token');
      localStorage.removeItem('astroshield_token_expires');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);
```

## API Keys (Alternative Authentication)

For some integration scenarios, you can use API keys instead of JWT tokens. API keys provide a simpler authentication method but do not expire automatically.

### Obtaining an API Key

API keys can be generated in the AstroShield Dashboard:

1. Log in to the [AstroShield Dashboard](https://dashboard.astroshield.com)
2. Navigate to API > API Keys
3. Click "Generate New API Key"
4. Name your key and select appropriate permissions
5. Copy and securely store your API key

### Using an API Key

Include the API key in the `X-API-Key` header:

```http
GET /api/v1/satellites
X-API-Key: as_api_key_12345abcdef67890
```

```python
import requests

def get_satellites_with_api_key(api_key):
    headers = {"X-API-Key": api_key}
    response = requests.get(
        "https://api.astroshield.com/api/v1/satellites",
        headers=headers
    )
    response.raise_for_status()
    return response.json()["data"]
```

## Security Considerations

### Token Security

- Never expose tokens in client-side code or URLs
- Set appropriate token expiration times
- Implement proper token revocation for user logout
- Use HTTPS for all API communications

### API Key Security

- Treat API keys like passwords
- Do not commit API keys to version control
- Use environment variables to store keys
- Implement key rotation policies

### Permission Scopes

The AstroShield API supports fine-grained permission scopes:

- `read:satellites` - Read satellite data
- `write:satellites` - Create and update satellites
- `read:maneuvers` - Read maneuver data
- `write:maneuvers` - Create and update maneuvers
- `execute:maneuvers` - Execute maneuvers
- `read:analytics` - Access analytics data
- `admin` - Full administrative access

Request only the scopes your application needs to follow the principle of least privilege.

## Troubleshooting

### Common Authentication Issues

1. **"Invalid credentials" error**
   - Verify username and password
   - Check for account lockout due to failed attempts
   - Ensure account is active

2. **"Token expired" error**
   - Implement automatic token refresh
   - Check client-server time synchronization
   - Verify token expiration handling

3. **"Invalid token" error**
   - Check token format and signature
   - Ensure token is included in the correct header format

4. **Missing Authorization header**
   - Verify header name and format (`Authorization: Bearer TOKEN`)
   - Check for header stripping by proxies or middleware

## Support

If you encounter authentication issues, please contact our support team:

- Email: support@astroshield.com
- Support Portal: https://support.astroshield.com
- Documentation: https://docs.astroshield.com/authentication 