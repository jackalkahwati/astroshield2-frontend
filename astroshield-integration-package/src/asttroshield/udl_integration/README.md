# UDL Integration Package for AstroShield

This package provides integration with the Unified Data Library (UDL) for space domain awareness data within the AstroShield platform. It allows for seamless access to UDL data sources through both REST API and Secure Messaging interfaces.

## Features

- **Comprehensive UDL API Access**: Full access to UDL REST APIs for retrieving space object data
- **Secure Messaging Integration**: Real-time data streaming through UDL's Secure Messaging API
- **Flexible Configuration**: Support for configuration via files, environment variables, and code
- **Resilient API Handling**: Circuit breaker pattern, rate limiting, and intelligent retries
- **High Performance**: Asynchronous processing, connection pooling, and efficient resource usage
- **Comprehensive Monitoring**: Detailed metrics, health checks, and operational visibility
- **Data Transformation**: Built-in transformers for converting UDL data to AstroShield formats
- **Kafka Integration**: Seamless publishing of UDL data to Kafka topics

## Installation

```bash
pip install astroshield-udl-integration
```

Or install from source:

```bash
git clone https://github.com/yourusername/astroshield-integration-package.git
cd astroshield-integration-package
pip install .
```

## Quick Start

### Basic Usage

```python
from asttroshield.udl_integration import UDLClient, UDLIntegration

# Initialize client
client = UDLClient(
    base_url="https://unifieddatalibrary.com",
    username="your_username",  # Or use UDL_USERNAME env var
    password="your_password"   # Or use UDL_PASSWORD env var
)

# Get state vector data
state_vectors = client.get_state_vectors(epoch="now", maxResults=10)
print(f"Retrieved {len(state_vectors)} state vectors")

# Initialize integration with Kafka
integration = UDLIntegration(
    udl_username="your_username",
    udl_password="your_password",
    kafka_bootstrap_servers="localhost:9092",
    config_file="path/to/udl_config.yaml"  # Optional configuration file
)

# Register transformers for specific data types
integration.register_topic(
    "state_vector",
    integration.transform_state_vector,
    "astroshield.state_vectors"
)

# Process and publish state vectors to Kafka
integration.process_state_vectors(epoch="now")

# Start continuous polling of UDL data
integration.start_polling()
```

### Using Secure Messaging for Real-time Data

```python
from asttroshield.udl_integration import UDLMessagingClient

# Initialize Secure Messaging client
messaging = UDLMessagingClient(
    base_url="https://unifieddatalibrary.com",
    username="your_username",
    password="your_password"
)

# Define callback for processing messages
def process_message(message):
    print(f"Received message: {message.get('id')}")
    # Process message...

# List available topics
topics = messaging.list_topics()
print(f"Available topics: {[t['name'] for t in topics]}")

# Start consuming messages from a topic
messaging.start_consumer(
    topic="statevector",
    callback_fn=process_message,
    start_from_latest=True
)

# Keep the application running
try:
    import time
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    messaging.stop_all_consumers()
```

## Configuration

The package supports multiple configuration methods:

1. **Configuration File**: YAML or JSON format
2. **Environment Variables**: For sensitive information like credentials
3. **Constructor Parameters**: Direct configuration in code

### Configuration File

Example `udl_config.yaml`:

```yaml
# UDL API Configuration
udl:
  base_url: "https://unifieddatalibrary.com"
  timeout: 30
  max_retries: 3
  use_secure_messaging: true

# Kafka configuration
kafka:
  bootstrap_servers: "localhost:9092"
  security_protocol: "PLAINTEXT"
  
# Topic configuration  
topics:
  state_vector:
    udl_params:
      maxResults: 100
      epoch: "now"
    transform_func: "transform_state_vector"
    kafka_topic: "udl.state_vectors"
    polling_interval: 60
```

### Environment Variables

```
# UDL credentials
UDL_USERNAME=your_username
UDL_PASSWORD=your_password
UDL_API_KEY=your_api_key

# UDL configuration
UDL_BASE_URL=https://unifieddatalibrary.com
UDL_TIMEOUT=30
UDL_MAX_RETRIES=3

# Kafka configuration
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_SECURITY_PROTOCOL=SASL_SSL
KAFKA_SASL_MECHANISM=PLAIN
KAFKA_SASL_USERNAME=kafka_username
KAFKA_SASL_PASSWORD=kafka_password
```

## Enhanced Resilience Features

### Rate Limiting

The integration respects UDL rate limits (3 requests per second) and implements adaptive rate limiting:

```python
# Configure rate limiting
client = UDLClient(
    base_url="https://unifieddatalibrary.com",
    username="your_username",
    password="your_password",
    rate_limit_requests=3,
    rate_limit_period=1.0
)
```

### Circuit Breaker

Automatically detects API failures and prevents cascading failures:

```python
# Configure circuit breaker
client = UDLClient(
    base_url="https://unifieddatalibrary.com",
    username="your_username",
    password="your_password",
    circuit_breaker_threshold=5,   # Trips after 5 failures
    circuit_breaker_timeout=60     # Attempts recovery after 60 seconds
)
```

### Caching

Efficiently caches responses to reduce API calls:

```python
# Get state vectors with caching
vectors1 = client.get_state_vectors(epoch="now", use_cache=True)

# Force fresh data
vectors2 = client.get_state_vectors(epoch="now", use_cache=False)

# Clear cache manually
client.clear_cache()
```

## Monitoring

### Health Checks

Periodic health checks ensure the UDL API is available:

```python
# Check UDL health
health = client.get_health_status()
print(f"UDL API status: {health['status']}")

# Get metrics for a specific consumer
metrics = messaging.get_consumer_metrics(topic="statevector")
print(f"Consumer metrics: {metrics}")

# Get overall metrics
overall = integration.get_metrics()
print(f"UDL integration metrics: {overall}")
```

### Prometheus Integration

Enable Prometheus metrics for operational monitoring:

```python
# Enable Prometheus monitoring
integration = UDLIntegration(
    prometheus_port=8000,
    monitoring_enabled=True
)
```

## Error Handling

The integration implements sophisticated error handling:

```python
# Configure error handling
client = UDLClient(
    error_handlers={
        "default": lambda e: logger.error(f"UDL error: {str(e)}"),
        "401": lambda e: send_alert("Authentication failed")
    }
)

# Custom error handler for consumers
def handle_consumer_error(exception):
    logger.error(f"Consumer error: {str(exception)}")
    send_alert("Consumer failed", exception)

messaging.start_consumer(
    topic="statevector",
    callback_fn=process_message,
    error_handler=handle_consumer_error
)
```

## API Reference

See the complete [API Reference Documentation](https://astroshield.readthedocs.io/en/latest/api/udl_integration.html) for detailed information on all classes and methods.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 