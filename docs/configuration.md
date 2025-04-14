# CCDM Service Configuration Guide

This document describes the configuration options for the CCDM service.

## Configuration Files

Configuration is managed through environment-specific YAML files located in the `config/` directory:

```
config/
  ├── config.development.yaml
  ├── config.testing.yaml
  ├── config.staging.yaml
  └── config.production.yaml
```

The appropriate configuration file is loaded based on the `DEPLOYMENT_ENV` environment variable.

## Environment Variables

The following environment variables are used to configure the service:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DEPLOYMENT_ENV` | Yes | `development` | Environment (development, testing, staging, production) |
| `CONFIG_PATH` | No | `./config` | Path to configuration directory |
| `LOG_LEVEL` | No | `info` | Logging level (debug, info, warn, error) |
| `DATABASE_URL` | Yes | - | Database connection string |
| `JWT_SECRET` | Yes | - | Secret for JWT token generation |
| `PORT` | No | `8080` | Port to run the service on |
| `ENABLE_SWAGGER` | No | `true` | Enable Swagger documentation |
| `CORS_ALLOWED_ORIGINS` | No | `*` | Comma-separated list of allowed origins for CORS |
| `CACHE_TTL_SECONDS` | No | `300` | Cache time-to-live in seconds |
| `RATE_LIMIT_WINDOW_MS` | No | `60000` | Rate limiting window in milliseconds |
| `RATE_LIMIT_MAX_REQUESTS` | No | `100` | Maximum number of requests per window |

## Configuration File Structure

The configuration files follow this structure:

```yaml
service:
  name: ccdm-service
  version: 1.2.3
  
server:
  port: 8080
  timeout: 30
  cors:
    enabled: true
    allowed_origins:
      - https://astroshield.space
      - https://staging.astroshield.space
  rate_limiting:
    enabled: true
    window_ms: 60000
    max_requests: 100
    skip_trusted_clients: true
    trusted_ips:
      - 10.0.0.0/8
      - 192.168.0.0/16

database:
  url: postgresql://user:password@localhost:5432/ccdm
  pool:
    min_connections: 5
    max_connections: 20
    idle_timeout_ms: 10000
  migrations:
    auto_run: true
    directory: ./migrations

auth:
  jwt:
    secret: your-secret-key
    expiration_hours: 24
  password:
    min_length: 8
    require_special_chars: true
    require_numbers: true
  
logging:
  level: info
  format: json
  output: stdout
  file_output:
    enabled: false
    path: /var/log/ccdm/service.log
  include_request_body: false
  
metrics:
  enabled: true
  prometheus:
    enabled: true
    endpoint: /metrics
  
cache:
  enabled: true
  type: redis  # or memory
  redis:
    url: redis://localhost:6379
    password: ""
    database: 0
  ttl_seconds: 300
  
alerts:
  enabled: true
  endpoints:
    email:
      enabled: true
      recipients:
        - alerts@astroshield.space
        - oncall@astroshield.space
    slack:
      enabled: true
      webhook_url: https://hooks.slack.com/services/...

data_sources:
  space_track:
    enabled: true
    username: ${SPACE_TRACK_USERNAME}
    password: ${SPACE_TRACK_PASSWORD}
    api_url: https://www.space-track.org
    request_timeout_seconds: 30
    rate_limit:
      requests_per_day: 1000
      requests_per_hour: 100
  
  jspoc:
    enabled: true
    api_key: ${JSPOC_API_KEY}
    api_url: https://api.jspoc.com
    request_timeout_seconds: 30

analysis:
  auto_update_interval_minutes: 60
  prediction_window_days: 7
  use_machine_learning: true
  ml_models:
    prediction:
      version: v2
      threshold: 0.85
    classification:
      version: v1
      threshold: 0.75
```

## Configuration Precedence

Configuration values are loaded with the following precedence (highest to lowest):

1. Environment variables
2. Command-line arguments
3. Environment-specific configuration file
4. Default values

## Environment-Specific Configurations

### Development

Development configuration (`config.development.yaml`) is optimized for local development:

- Debug logging enabled
- In-memory cache
- Automatic database migrations
- Swagger documentation enabled
- CORS allows all origins
- Metrics collection disabled

### Testing

Testing configuration (`config.testing.yaml`) is designed for automated testing:

- No caching
- In-memory database or test database
- Detailed error responses
- No rate limiting

### Staging

Staging configuration (`config.staging.yaml`) mimics production with some adjustments:

- Lower rate limits
- Separate database instance
- Warning-level logging
- Test data sources with limited quotas

### Production

Production configuration (`config.production.yaml`) is optimized for reliability and security:

- Info-level logging
- No detailed error messages exposed
- Redis caching enabled
- Strict rate limiting
- CORS restricted to specific origins
- Prometheus metrics enabled
- Alerts configured

## Secrets Management

Sensitive configuration values like passwords and API keys should be provided through environment variables or a secure secrets management solution.

The configuration files support environment variable interpolation using the `${VARIABLE_NAME}` syntax:

```yaml
database:
  url: postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}
```

## Database Configuration

The database connection is configured via the `DATABASE_URL` environment variable or the `database.url` configuration property. The URL format depends on the database type:

### PostgreSQL

```
postgresql://username:password@hostname:port/database
```

### SQLite

```
sqlite:///path/to/database.sqlite
```

## Advanced Configuration

### Connection Pooling

Database connection pooling is configured in the `database.pool` section:

```yaml
database:
  pool:
    min_connections: 5
    max_connections: 20
    idle_timeout_ms: 10000
    max_lifetime_ms: 300000
```

### Rate Limiting

Rate limiting is configured in the `server.rate_limiting` section:

```yaml
server:
  rate_limiting:
    enabled: true
    window_ms: 60000
    max_requests: 100
    skip_trusted_clients: true
    trusted_ips:
      - 10.0.0.0/8
```

### Caching

Response caching is configured in the `cache` section:

```yaml
cache:
  enabled: true
  type: redis
  redis:
    url: redis://localhost:6379
  ttl_seconds: 300
```

### CORS

Cross-Origin Resource Sharing is configured in the `server.cors` section:

```yaml
server:
  cors:
    enabled: true
    allowed_origins:
      - https://astroshield.space
    allowed_methods:
      - GET
      - POST
    allowed_headers:
      - Content-Type
      - Authorization
    expose_headers:
      - X-RateLimit-Limit
      - X-RateLimit-Remaining
    max_age: 86400
```

## Monitoring Configuration

### Logging

Logging is configured in the `logging` section:

```yaml
logging:
  level: info
  format: json
  output: stdout
  file_output:
    enabled: true
    path: /var/log/ccdm/service.log
    max_size_mb: 100
    max_files: 5
  include_request_body: false
```

### Metrics

Metrics collection is configured in the `metrics` section:

```yaml
metrics:
  enabled: true
  prometheus:
    enabled: true
    endpoint: /metrics
  statsd:
    enabled: false
    host: localhost
    port: 8125
    prefix: ccdm
```

## Applying Configuration Changes

Most configuration changes require a service restart to take effect. Some components support runtime configuration reloading through the admin API endpoints. 