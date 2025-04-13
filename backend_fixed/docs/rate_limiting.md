# AstroShield API Rate Limiting Guide

The AstroShield API implements rate limiting to ensure fair usage, maintain system stability, and prevent abuse. This guide explains how rate limiting works and provides best practices for handling rate limits in your applications.

## Rate Limit Policy

The API enforces different rate limits based on authentication status and subscription tier:

| Client Type | Requests per Minute | Requests per Day |
|-------------|---------------------|------------------|
| Unauthenticated | 20 | 1,000 |
| Basic Tier | 100 | 10,000 |
| Standard Tier | 300 | 50,000 |
| Premium Tier | 1,000 | 200,000 |
| Enterprise | Custom | Custom |

Rate limits are applied on a per-IP basis for unauthenticated requests and per-API key for authenticated requests.

## Rate Limit Headers

The API includes the following headers in all responses to help you track your rate limit status:

```
X-RateLimit-Limit: 100          # The maximum number of requests allowed in the current period
X-RateLimit-Remaining: 95       # The number of remaining requests in the current period
X-RateLimit-Reset: 1623456789   # The time at which the current rate limit window resets (Unix timestamp)
X-RateLimit-Used: 5             # The number of requests used in the current period
```

When you exceed your rate limit, the API will respond with a `429 Too Many Requests` status code and include a `Retry-After` header indicating how many seconds to wait before retrying.

## Rate Limit Response Example

```http
HTTP/1.1 429 Too Many Requests
Content-Type: application/json
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1623456789
X-RateLimit-Used: 100
Retry-After: 30

{
  "detail": "Rate limit exceeded. Please try again in 30 seconds.",
  "status_code": 429,
  "error_code": "rate_limit_exceeded"
}
```

## Rate Limiting Strategy

The AstroShield API uses a sliding window rate limiting strategy. This means:

1. Each request consumes one rate limit token
2. Tokens are replenished gradually over time rather than all at once
3. The rate limit window "slides" forward with time rather than resetting at fixed intervals

This approach provides a smoother experience by avoiding sudden traffic spikes when limits reset.

## Endpoint-Specific Rate Limits

Some endpoints have specialized rate limits due to their resource intensity or criticality:

| Endpoint | Rate Limit (requests/minute) |
|----------|------------------------------|
| `/api/v1/simulate` | 10 (all tiers) |
| `/api/v1/analytics/*` | 30 (Basic), 60 (Standard), 120 (Premium) |
| `/api/v1/satellites/bulk` | 5 (all tiers) |
| `/api/v1/token` | 5 (all tiers) |

These specialized limits apply independently of the global rate limits.

## Best Practices for Handling Rate Limits

### 1. Implement Exponential Backoff

When encountering rate limit errors, use exponential backoff to retry requests:

```python
import time
import requests
import random

def make_request_with_backoff(url, headers, max_retries=5):
    for attempt in range(max_retries):
        response = requests.get(url, headers=headers)
        
        if response.status_code != 429:  # Not rate limited
            return response
        
        # Get retry-after time from header or use exponential backoff
        retry_after = int(response.headers.get('Retry-After', 2 ** attempt))
        
        # Add jitter to prevent thundering herd problem
        jitter = random.uniform(0, 0.1 * retry_after)
        sleep_time = retry_after + jitter
        
        print(f"Rate limited. Retrying in {sleep_time:.2f} seconds...")
        time.sleep(sleep_time)
    
    # If we get here, we've exceeded max retries
    response.raise_for_status()
```

### 2. Track Rate Limit Headers

Monitor the rate limit headers to understand your usage patterns:

```javascript
function handleRateLimits(response) {
  const rateLimit = {
    limit: parseInt(response.headers.get('X-RateLimit-Limit') || '0'),
    remaining: parseInt(response.headers.get('X-RateLimit-Remaining') || '0'),
    reset: parseInt(response.headers.get('X-RateLimit-Reset') || '0'),
    used: parseInt(response.headers.get('X-RateLimit-Used') || '0')
  };
  
  // If we're close to the limit (e.g., less than 10% remaining)
  if (rateLimit.remaining < rateLimit.limit * 0.1) {
    console.warn(`Rate limit warning: ${rateLimit.remaining}/${rateLimit.limit} requests remaining`);
    // You might want to slow down your request rate or pause non-critical requests
  }
  
  return rateLimit;
}
```

### 3. Cache Responses

Reduce the number of API calls by caching responses for appropriate endpoints:

```python
import requests
from cachetools import TTLCache

# Create a cache with a maximum of 100 items that expire after 60 seconds
cache = TTLCache(maxsize=100, ttl=60)

def get_satellite_with_cache(token, satellite_id):
    cache_key = f"satellite:{satellite_id}"
    
    # Check if the data is in the cache
    if cache_key in cache:
        return cache[cache_key]
    
    # Not in cache, make the API request
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"https://api.astroshield.com/api/v1/satellites/{satellite_id}", headers=headers)
    response.raise_for_status()
    
    # Store in cache and return
    data = response.json()["data"]
    cache[cache_key] = data
    return data
```

### 4. Batch Requests

Where possible, use bulk endpoints to reduce the number of API calls:

```python
# Instead of:
for satellite_id in satellite_ids:
    get_satellite(token, satellite_id)

# Use:
response = requests.post(
    "https://api.astroshield.com/api/v1/satellites/batch",
    headers={"Authorization": f"Bearer {token}"},
    json={"ids": satellite_ids}
)
satellites = response.json()["data"]
```

### 5. Implement Rate Limiting on Your Side

To avoid hitting API rate limits, implement your own rate limiting:

```javascript
class RateLimiter {
  constructor(requestsPerMinute) {
    this.requestsPerMinute = requestsPerMinute;
    this.queue = [];
    this.running = false;
  }
  
  async request(fn) {
    return new Promise((resolve, reject) => {
      this.queue.push({ fn, resolve, reject });
      this.processQueue();
    });
  }
  
  async processQueue() {
    if (this.running) return;
    this.running = true;
    
    while (this.queue.length > 0) {
      const { fn, resolve, reject } = this.queue.shift();
      try {
        // Execute the API call
        const result = await fn();
        resolve(result);
      } catch (error) {
        reject(error);
      }
      
      // Wait to respect the rate limit
      await new Promise(resolve => setTimeout(resolve, 60000 / this.requestsPerMinute));
    }
    
    this.running = false;
  }
}

// Usage:
const limiter = new RateLimiter(90);  // 90 requests per minute (leaving some buffer)

async function getSatellite(id) {
  return limiter.request(() => fetch(`https://api.astroshield.com/api/v1/satellites/${id}`));
}
```

### 6. Monitor Long-Term Usage

For applications with predictable traffic patterns, monitor your usage over longer periods:

```python
class RateUsageTracker:
    def __init__(self):
        self.daily_usage = 0
        self.reset_date = datetime.now().date() + timedelta(days=1)
    
    def track_request(self):
        # Check if we need to reset the counter for a new day
        if datetime.now().date() >= self.reset_date:
            self.daily_usage = 0
            self.reset_date = datetime.now().date() + timedelta(days=1)
        
        self.daily_usage += 1
        
        # You can alert if approaching daily limits
        if self.daily_usage > 9000 and self.daily_usage % 100 == 0:  # For 10K daily limit
            print(f"Warning: Daily usage at {self.daily_usage}/10000")
```

## Rate Limit Increases

If your application requires higher rate limits:

1. **Optimize First**: Ensure you're following best practices to minimize unnecessary requests
2. **Upgrade Tier**: Consider upgrading to a higher subscription tier
3. **Contact Support**: For enterprise needs, contact support@astroshield.com to discuss custom rate limits

## Monitoring Rate Limit Status

The AstroShield Dashboard provides real-time monitoring of your API usage and rate limit status:

1. Log in to the [AstroShield Dashboard](https://dashboard.astroshield.com)
2. Navigate to API > Usage
3. View current usage, rate limit status, and historical patterns

## Frequently Asked Questions

### How do I know which tier I'm on?

You can check your current tier and rate limits by making a GET request to `/api/v1/account/limits` with your API key.

### Do unused requests roll over?

No, rate limits do not roll over from one period to the next. Unused requests within a rate limit window are not added to the next window.

### What happens during an API outage or maintenance?

Rate limits are not adjusted for API outages or maintenance. Plan your application to handle such scenarios gracefully.

### Are webhook notifications counted against rate limits?

No, outbound webhook notifications from AstroShield to your systems don't count against your rate limits.

### Can I get temporary rate limit increases?

For special events or temporary needs, contact support@astroshield.com at least 5 business days in advance to request temporary rate limit adjustments.

## Rate Limiting Error Codes

When you exceed rate limits, the API returns a `429 Too Many Requests` status code with one of the following error codes:

| Error Code | Description |
|------------|-------------|
| `rate_limit_exceeded` | Standard rate limit exceeded |
| `concurrent_requests_exceeded` | Too many concurrent requests |
| `daily_limit_exceeded` | Daily request allocation exceeded |
| `endpoint_limit_exceeded` | Endpoint-specific rate limit exceeded |

Understanding these error codes can help you implement more specific handling strategies.

## Conclusion

Proper handling of rate limits is essential for building reliable applications with the AstroShield API. By following these best practices, you can ensure your application operates within limits while providing a smooth experience for your users. 