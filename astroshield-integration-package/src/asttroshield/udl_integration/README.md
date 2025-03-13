# UDL Integration for AstroShield

This package provides integration between the Unified Data Library (UDL) APIs and AstroShield Kafka topics. It retrieves data from UDL, transforms it to AstroShield format, and publishes it to the appropriate Kafka topics.

## Features

- Retrieves a wide range of space situational awareness data from UDL:
  - State vectors
  - Conjunctions
  - Launch events
  - Tracks
  - Ephemeris
  - Maneuvers
  - Observations
  - Orbit determination
  - Sensor data
  - ELSETs (Element Sets / TLEs)
  - Weather data
  - Sensor tasking
  - Site information
  - RF data
  - Earth orientation parameters
  - Solar and geomagnetic data
  - Star catalog information
  - Cyber threats
  - Link status information
  - Communications data
  - Mission operations data
  - Vessel tracking data
  - Aircraft tracking data
  - Ground imagery
  - Sky imagery
  - Video streaming information
- Transforms UDL data to AstroShield format
- Publishes transformed data to AstroShield Kafka topics
- Supports two modes of operation:
  - **REST API Polling**: Periodically queries UDL REST APIs for data
  - **Secure Messaging Streaming**: Real-time streaming of data using UDL Secure Messaging API (requires special authorization)
- Supports continuous integration with configurable intervals
- Provides Docker containerization for easy deployment
- Allows selective processing of specific data types

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/your-organization/astroshield-integration-package.git
cd astroshield-integration-package

# Install the package
pip install -e .
```

### Using Docker

```bash
# Build the Docker image
docker build -t astroshield/udl-integration -f src/asttroshield/udl_integration/Dockerfile .

# Run the Docker container
docker run -d --name udl-integration \
  -e UDL_API_KEY=your-api-key \
  -e KAFKA_BOOTSTRAP_SERVERS=kafka:9092 \
  -e KAFKA_SASL_USERNAME=your-username \
  -e KAFKA_SASL_PASSWORD=your-password \
  astroshield/udl-integration
```

## Configuration

The UDL integration can be configured using environment variables or command-line arguments:

### UDL Configuration

- `UDL_BASE_URL`: Base URL for the UDL API (default: `https://unifieddatalibrary.com`)
- `UDL_API_KEY`: API key for UDL authentication
- `UDL_USERNAME`: Username for UDL authentication
- `UDL_PASSWORD`: Password for UDL authentication

### Kafka Configuration

- `KAFKA_BOOTSTRAP_SERVERS`: Comma-separated list of Kafka broker addresses (default: `localhost:9092`)
- `KAFKA_SECURITY_PROTOCOL`: Security protocol for Kafka (default: `SASL_SSL`)
- `KAFKA_SASL_MECHANISM`: SASL mechanism for Kafka (default: `PLAIN`)
- `KAFKA_SASL_USERNAME`: SASL username for Kafka
- `KAFKA_SASL_PASSWORD`: SASL password for Kafka
- `KAFKA_SSL_CA_LOCATION`: Path to CA certificate file for Kafka

### Integration Configuration

- `INTEGRATION_INTERVAL`: Interval between integration runs in seconds (default: `60`)
- `LOG_LEVEL`: Logging level (default: `INFO`)
- `UDL_INTEGRATION_DATA_TYPES`: Comma-separated list of data types to process (default: all)

## Usage

### Command-Line Interface

#### REST API Polling Mode

```bash
# Run the integration with default settings
python -m asttroshield.udl_integration.integration

# Run the integration with custom settings
python -m asttroshield.udl_integration.integration \
  --udl-api-key your-api-key \
  --kafka-bootstrap-servers kafka:9092 \
  --kafka-sasl-username your-username \
  --kafka-sasl-password your-password \
  --interval 300

# Run the integration with specific data types
python -m asttroshield.udl_integration.integration \
  --data-types state_vectors conjunctions launch_events

# Run a single integration iteration with specific data types
python -m asttroshield.udl_integration.integration \
  --one-shot \
  --data-types tracks ephemeris maneuvers
```

#### Secure Messaging Streaming Mode

```bash
# Run the integration with Secure Messaging enabled (requires UDL authorization)
python -m asttroshield.udl_integration.integration \
  --use-secure-messaging \
  --udl-username your-username \
  --udl-password your-password \
  --streaming-topics statevector,conjunction,elset \
  --sample-period 0.5 \
  --kafka-bootstrap-servers kafka:9092
```

> **Note**: Access to UDL Secure Messaging API requires special authorization from the UDL team. The Secure Messaging API provides real-time data streaming that is far superior to polling the REST API. To request access, contact the UDL Support Team.

Available data types:
- `state_vectors`: State vectors for orbital objects
- `conjunctions`: Conjunction events between orbital objects
- `launch_events`: Launch events and related data
- `tracks`: Tracking data for orbital objects
- `ephemeris`: Predicted positions of objects over time
- `maneuvers`: Spacecraft maneuver detections
- `observations`: Raw sensor observations
- `sensor_data`: Sensor information and capabilities
- `orbit_determinations`: Detailed orbit determination results
- `elsets`: Two-line element sets (TLEs)
- `weather_data`: Weather data affecting space operations
- `sensor_tasking`: Sensor tasking requests and status
- `cyber_threats`: Cyber threat notifications and alerts
- `link_status`: Communication link status information
- `comm_data`: Communications data and transmissions
- `mission_ops_data`: Mission operations information
- `vessel_data`: Maritime vessel tracking information
- `aircraft_data`: Aircraft tracking information
- `ground_imagery`: Ground-based imagery data
- `sky_imagery`: Sky/space imagery data
- `video_streaming`: Video streaming information for space monitoring

### Python API

```python
from asttroshield.udl_integration.integration import UDLIntegration

# Initialize the integration with REST API
integration = UDLIntegration(
    udl_api_key="your-api-key",
    kafka_bootstrap_servers="kafka:9092",
    kafka_sasl_username="your-username",
    kafka_sasl_password="your-password"
)

# Initialize with Secure Messaging enabled
integration_streaming = UDLIntegration(
    udl_username="your-username",
    udl_password="your-password",
    kafka_bootstrap_servers="kafka:9092",
    kafka_sasl_username="your-username",
    kafka_sasl_password="your-password",
    use_secure_messaging=True,
    sample_period=0.5  # Controls rate limiting (minimum: 0.34 seconds)
)

# Start streaming from specific topics
integration_streaming.start_streaming("statevector")
integration_streaming.start_streaming("conjunction")

# Start streaming from multiple topics at once
topics = ["statevector", "track", "elset"]
results = integration_streaming.start_streaming_multiple(topics)

# Stop streaming from a topic
integration_streaming.stop_streaming("statevector")

# Stop all active streams
integration_streaming.stop_all_streaming()

# Get a list of all available topics
available_topics = integration_streaming.get_available_topics()

# Get a list of currently active streams
active_streams = integration_streaming.get_active_streams()

# Process state vectors
integration.process_state_vectors()

# Process conjunctions
integration.process_conjunctions()

# Process launch events
integration.process_launch_events()

# Process tracks
integration.process_tracks()

# Process ephemeris
integration.process_ephemeris()

# Process maneuvers
integration.process_maneuvers()

# Process orbit determinations
integration.process_orbit_determinations()

# Run continuous integration with all data types
integration.run_continuous_integration(interval_seconds=300)

# Run continuous integration with specific data types
integration.run_continuous_integration(
    interval_seconds=300,
    data_types=["state_vectors", "conjunctions", "ephemeris"]
)
```

## Data Flow

1. The UDL client retrieves data from the UDL APIs (REST API or Secure Messaging)
2. The transformers convert UDL data to AstroShield format
3. The Kafka producer publishes the transformed data to AstroShield Kafka topics

### Secure Messaging vs REST API

| Feature | Secure Messaging API | REST API |
|---------|---------------------|----------|
| Real-time data | ✅ Continuous streaming | ❌ Periodic polling only |
| Authorization | Requires special access | Standard UDL access |
| Data freshness | Latest data as it arrives | May have delays between polls |
| Rate limiting | 3 requests per second (0.34s) | Varies by endpoint |
| Offset tracking | Maintains position in data stream | N/A |
| Implementation | `UDLMessagingClient` class | `UDLClient` class |
| Authentication | Username/password only | API key or username/password |

### Supported Data Types

- **State Vectors**: Position and velocity information for space objects
- **Conjunctions**: Close approach events between space objects
- **Launch Events**: Information about detected launch events
- **Tracks**: Tracking data for orbital objects over time
- **Ephemeris**: Predicted positions of objects over specified time periods
- **Maneuvers**: Spacecraft maneuver detection and characterization
- **Observations**: Raw sensor observations of space objects
- **Orbit Determination**: Detailed orbit determination results
- **Sensor Data**: Information about sensor capabilities and status
- **ELSETs/TLEs**: Two-line element sets for orbital objects
- **Weather Data**: Weather information affecting space operations
- **Sensor Tasking**: Sensor tasking requests and status
- **Cyber Threats**: Cyber security threats and notifications
- **Link Status**: Communication link status information
- **Communications Data**: Data about communications between ground and space assets
- **Mission Operations**: Information about mission operations and activities
- **Vessel Tracking**: Maritime vessel tracking information
- **Aircraft Tracking**: Aircraft tracking information
- **Ground Imagery**: Ground-based imagery data
- **Sky Imagery**: Sky/space imagery data
- **Video Streaming**: Video streaming information for space monitoring

## Docker Deployment

The UDL integration can be deployed as a Docker container:

```bash
# Build the Docker image
docker build -t astroshield/udl-integration -f src/asttroshield/udl_integration/Dockerfile .

# Run the Docker container
docker run -d --name udl-integration \
  -e UDL_API_KEY=your-api-key \
  -e KAFKA_BOOTSTRAP_SERVERS=kafka:9092 \
  -e KAFKA_SASL_USERNAME=your-username \
  -e KAFKA_SASL_PASSWORD=your-password \
  astroshield/udl-integration
```

## Kubernetes Deployment

For production deployments, you can use Kubernetes:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: udl-integration
spec:
  replicas: 1
  selector:
    matchLabels:
      app: udl-integration
  template:
    metadata:
      labels:
        app: udl-integration
    spec:
      containers:
      - name: udl-integration
        image: astroshield/udl-integration:latest
        env:
        - name: UDL_API_KEY
          valueFrom:
            secretKeyRef:
              name: udl-credentials
              key: api-key
        - name: KAFKA_BOOTSTRAP_SERVERS
          value: "kafka:9092"
        - name: KAFKA_SASL_USERNAME
          valueFrom:
            secretKeyRef:
              name: kafka-credentials
              key: username
        - name: KAFKA_SASL_PASSWORD
          valueFrom:
            secretKeyRef:
              name: kafka-credentials
              key: password
        - name: INTEGRATION_INTERVAL
          value: "300"
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Testing

The UDL integration includes comprehensive test coverage for all components. The test suite uses unittest and pytest frameworks with mocking to isolate components for testing.

### Test Structure

- **test_client.py**: Tests for the UDL API client
  - Tests for initialization, authentication, and request handling
  - Mocks HTTP responses to simulate UDL API behavior
  
- **test_transformers.py**: Tests for the data transformation functions
  - Tests for each transformation function
  - Verifies correct mapping between UDL and AstroShield formats

- **test_kafka_producer.py**: Tests for the Kafka message producer
  - Tests for initialization, serialization, and message sending
  - Mocks Kafka client interactions

- **test_integration.py**: Tests for the main integration module
  - Tests for the integration between client, transformers, and producer
  - Tests for the continuous integration functionality

### Running Tests Locally

You can run the tests locally using the provided shell script:

```bash
# Make the script executable
chmod +x src/asttroshield/udl_integration/run_tests.sh

# Run all tests
./src/asttroshield/udl_integration/run_tests.sh
```

### Running Tests with Docker

A dedicated Dockerfile for testing is provided:

```bash
# Build the test Docker image
docker build -t astroshield/udl-integration-test -f src/asttroshield/udl_integration/Dockerfile.test .

# Run tests in Docker
docker run --rm astroshield/udl-integration-test
```

### Test Coverage

The test suite includes coverage for:

- UDL client methods for all supported APIs
- Data transformers for all data types
- Kafka producer methods
- Integration module functionality

Tests mock external dependencies (UDL API calls, Kafka) to ensure isolation and repeatability.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 