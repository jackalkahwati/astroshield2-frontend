# AstroShield Integration Package

This integration package provides all the necessary resources for integrating with AstroShield's APIs and Kafka streams. It includes API specifications, message schemas, example code, and configuration templates to help you quickly connect to and utilize AstroShield's services.

## Table of Contents
- [Package Contents](#package-contents)
- [Getting Started](#getting-started)
- [Authentication](#authentication)
- [Available Kafka Topics](#available-kafka-topics)
- [System Architecture](#system-architecture)
- [Message Traceability](#message-traceability)
- [Integration Best Practices](#integration-best-practices)
- [Troubleshooting](#troubleshooting)
- [Support](#support)
- [Version Information](#version-information)
- [License](#license)
- [UDL Integration](#udl-integration)
- [Docker Deployment](#docker-deployment)

## Package Contents

- **API Documentation**
  - OpenAPI 3.0 specification (`api/openapi.yaml`)
  - Postman collection (`postman/AstroShield_API.postman_collection.json`)
  - Postman environment (`postman/AstroShield_API.postman_environment.json`)

- **Kafka Message Schemas**
  - JSON Schema definitions for all Kafka topics (`schemas/`)
  - Schema for Subsystem 4 CCDM Detection (`schemas/ss4.ccdm.detection.schema.json`)
  - Schema for Subsystem 6 Threat Assessment (`schemas/ss6.threat.assessment.schema.json`)
  - Example messages for each topic

- **Example Code**
  - Python examples for API integration (`examples/python/`)
  - Java examples for Kafka consumer integration (`examples/java/`)
  - JavaScript examples for Kafka producer integration (`examples/javascript/`)
  - CCDM detection consumer example (`examples/python/ccdm_consumer.py`)
  - Message tracing example showing cross-subsystem data flow (`examples/python/message_tracing_example.py`)

- **Configuration Templates**
  - Kafka client configuration (`config/kafka-client.properties`)

## Getting Started

### Prerequisites
- API credentials (contact jack@lattis.io to obtain)
- Kafka cluster access (provided with your subscription)
- One of the following development environments:
  - Python 3.8+
  - Java 11+
  - Node.js 14+

### API Integration

1. Review the OpenAPI specification in `api/openapi.yaml` to understand the available endpoints, request/response formats, and authentication requirements.

2. Import the Postman collection and environment from the `postman/` directory to quickly test the API endpoints.

3. Use the Python example in `examples/python/api_client.py` as a reference for implementing your own API client.

### Kafka Integration

1. Review the JSON Schema definitions in the `schemas/` directory to understand the structure of messages for each Kafka topic.

2. Configure your Kafka client using the template in `config/kafka-client.properties`. Replace the placeholder values with your actual credentials provided by AstroShield.

3. Use the example code in `examples/java/` and `examples/javascript/` as a reference for implementing your own Kafka consumers and producers.

## Authentication

### API Authentication

The AstroShield API supports two authentication methods:

1. **JWT Authentication**: Obtain a JWT token by sending a POST request to `/auth/token` with your username and password. Include the token in the `Authorization` header of subsequent requests as `Bearer <token>`.

2. **API Key Authentication**: Include your API key in the `X-API-Key` header of each request.

### Kafka Authentication

Kafka connections use SASL/PLAIN authentication over SSL. Configure your Kafka client with the following:

- Security protocol: `SASL_SSL`
- SASL mechanism: `PLAIN`
- SASL JAAS config: Username and password provided by AstroShield

Example configuration:
```properties
bootstrap.servers=kafka.astroshield.com:9093
security.protocol=SASL_SSL
sasl.mechanism=PLAIN
sasl.jaas.config=org.apache.kafka.common.security.plain.PlainLoginModule required username="your-username" password="your-password";
```

## Available Kafka Topics

| Topic | Description | Schema | Subsystem |
|-------|-------------|--------|-----------|
| `ss0.sensor.heartbeat` | Sensor health monitoring data | [Schema](schemas/ss0.sensor.heartbeat.schema.json) | Subsystem 0 (Data Ingestion) |
| `ss2.data.state-vector` | Spacecraft state vectors | [Schema](schemas/ss2.data.state-vector.schema.json) | Subsystem 2 (State Estimation) |
| `ss4.ccdm.detection` | CCDM behavior detection | [Schema](schemas/ss4.ccdm.detection.schema.json) | Subsystem 4 (CCDM) |
| `ss5.launch.prediction` | Predictions of upcoming launches | [Schema](schemas/ss5.launch.prediction.schema.json) | Subsystem 5 (Hostility Monitoring) |
| `ss5.telemetry.data` | Telemetry data from spacecraft | [Schema](schemas/ss5.telemetry.data.schema.json) | Subsystem 5 (Hostility Monitoring) |
| `ss5.conjunction.events` | Conjunction events between spacecraft | [Schema](schemas/ss5.conjunction.events.schema.json) | Subsystem 5 (Hostility Monitoring) |
| `ss5.cyber.threats` | Cyber threat notifications | [Schema](schemas/ss5.cyber.threats.schema.json) | Subsystem 5 (Hostility Monitoring) |
| `ss6.threat.assessment` | Comprehensive threat assessments and recommendations | [Schema](schemas/ss6.threat.assessment.schema.json) | Subsystem 6 (Threat Assessment) |
| `dmd-od-update` | DMD orbit determination updates | [Schema](schemas/dmd.od.schema.json) | Event Processor |
| `weather-data` | Weather condition updates | [Schema](schemas/weather.schema.json) | Event Processor |
| `maneuvers-detected` | Detected satellite maneuvers | [Schema](schemas/maneuvers.detected.schema.json) | Event Processor |
| `observation-windows` | Recommended observation windows | [Schema](schemas/observation.windows.schema.json) | Event Processor |

## System Architecture

AstroShield is organized into a modular subsystem architecture that enables distributed processing and scalability. Each subsystem focuses on specific aspects of space domain awareness:

### Subsystem 0: Data Ingestion
Responsible for ingesting data from various sensors and external sources. This subsystem handles the initial processing of raw data and makes it available to other subsystems through Kafka topics.

### Subsystem 1: Target Modeling
Maintains a database of known space objects, their capabilities, and characteristics. This subsystem provides reference data for object identification and threat assessment.

### Subsystem 2: State Estimation
Performs orbit determination, correlation, maneuver detection, and propagation. This subsystem is responsible for maintaining accurate state vectors for tracked objects and predicting their future positions.

### Subsystem 3: Command and Control (C2)
Handles sensor tasking and orchestration to optimize observation collection. This subsystem prioritizes objects for tracking based on inputs from other subsystems.

### Subsystem 4: CCDM Detection
Focuses on detecting Camouflage, Concealment, Deception, and Maneuvering behaviors. This subsystem analyzes pattern-of-life violations and anomalous behaviors that might indicate hostile intent.

### Subsystem 5: Hostility Monitoring
Monitors for potential threats, including conjunction events, cyber threats, and launch predictions. This subsystem provides early warning of potential hostile activities.

### Subsystem 6: Threat Assessment
Integrates data from all other subsystems to provide comprehensive threat assessments and recommended actions.

### Message Flow Between Subsystems

The AstroShield platform relies on a data flow architecture where information passes through subsystems in a logical progression:

1. **Raw Data Ingestion** (Subsystem 0)
   - Sensor data from ground stations, satellites, and external sources is ingested
   - Raw data is processed and converted to standardized formats
   - Output: Observation messages (ss0.sensor.observation)

2. **Object Identification** (Subsystem 1)
   - Correlates observations with known objects
   - Identifies and catalogs new objects
   - Output: Object identification messages (ss1.object.identification)

3. **State Determination** (Subsystem 2)
   - Processes observation data to generate state vectors
   - Performs orbit determination and propagation
   - Output: State vector messages (ss2.data.state-vector)

4. **Anomaly Detection** (Subsystem 4)
   - Analyzes state vectors and behavior patterns
   - Detects CCDM activities and anomalous behaviors
   - Output: CCDM detection messages (ss4.ccdm.detection)

5. **Threat Monitoring** (Subsystem 5)
   - Monitors for specific threat types (conjunctions, cyber, etc.)
   - Correlates multiple data sources for comprehensive monitoring
   - Output: Various threat messages (ss5.*.*)

6. **Assessment and Response** (Subsystem 6)
   - Integrates detection and monitoring data
   - Generates comprehensive threat assessments
   - Provides recommended actions
   - Output: Threat assessment messages (ss6.threat.assessment)

#### Example: Tracking a Suspicious Maneuver

Here's how information flows through the system when tracking a suspicious spacecraft maneuver:

1. **Subsystem 0**: A ground-based radar observes a spacecraft and sends the raw observation data.
2. **Subsystem 1**: The spacecraft is identified as "COSMOS-1234" based on observation correlation.
3. **Subsystem 2**: State vectors are generated showing the spacecraft's position and velocity.
4. **Subsystem 2**: Subsequent state vectors reveal an unexpected change in velocity.
5. **Subsystem 4**: The CCDM detector identifies this as a potential deceptive maneuver.
6. **Subsystem 5**: Hostility monitoring correlates this with recent cyber activity targeting ground systems.
7. **Subsystem 6**: A comprehensive threat assessment is generated, indicating medium threat level with recommended actions.

Throughout this process, the message tracing system maintains the lineage of information, allowing users to trace back from the final assessment to the original observation data.

## Message Traceability

AstroShield implements message traceability to track the lineage of information as it flows through the system. This enables users to understand how detections, alerts, and assessments were derived from raw data.

### Traceability Headers

All Kafka messages include traceability information in their headers:

- `traceId`: A unique identifier that follows the processing chain across multiple messages
- `parentMessageIds`: References to previous messages that contributed to the current message

### Implementing Traceability

When consuming messages and producing new ones based on that data:

1. Extract the `traceId` from the incoming message
2. Use the same `traceId` in your outgoing message
3. Add the `messageId` of the incoming message to the `parentMessageIds` array in your outgoing message

Example in Python:
```python
def process_message(incoming_message):
    # Process the message data
    result = analyze_data(incoming_message['payload'])
    
    # Create a new message with traceability
    outgoing_message = {
        'header': {
            'messageId': str(uuid.uuid4()),
            'timestamp': datetime.utcnow().isoformat(),
            'source': 'my-application',
            'messageType': 'my-message-type',
            'traceId': incoming_message['header']['traceId'],
            'parentMessageIds': [incoming_message['header']['messageId']]
        },
        'payload': result
    }
    
    return outgoing_message
```

## Integration Best Practices

### API Integration
- Implement proper error handling and retry logic
- Cache responses when appropriate to reduce API calls
- Use connection pooling for better performance
- Set reasonable timeouts for API requests
- Implement rate limiting in your client to avoid hitting API limits

### Kafka Integration
- Use consumer groups for scalable message processing
- Implement proper error handling for message processing
- Consider using a dead-letter queue for failed messages
- Monitor consumer lag to ensure timely processing
- Implement idempotent processing to handle duplicate messages

### Security Best Practices
- Store API keys and credentials securely (use environment variables or a secrets manager)
- Rotate API keys regularly (recommended every 90 days)
- Use TLS for all communications
- Implement proper access controls for your integration
- Audit and log all API calls and Kafka message processing

## Troubleshooting

### Common API Issues

| Issue | Possible Cause | Solution |
|-------|----------------|----------|
| 401 Unauthorized | Invalid or expired token | Refresh your JWT token or check API key |
| 403 Forbidden | Insufficient permissions | Contact jack@lattis.io to update permissions |
| 429 Too Many Requests | Rate limit exceeded | Implement backoff and retry logic |
| 5xx Server Error | Server-side issue | Retry with exponential backoff |

### Common Kafka Issues

| Issue | Possible Cause | Solution |
|-------|----------------|----------|
| Connection failures | Network or authentication issues | Check credentials and network connectivity |
| Consumer lag | Slow processing or high message volume | Scale consumers or optimize processing |
| Deserialization errors | Schema mismatch | Verify message format against schema |
| Offset commit failures | Broker connectivity issues | Implement proper error handling |

## Support

If you encounter any issues or have questions about integrating with AstroShield, please contact:

- Email: jack@lattis.io
- Response Time: Within 24 hours
- For urgent issues: Include "URGENT" in the email subject

## Version Information

- Integration Package Version: 1.0.0
- API Version: v1
- Last Updated: 2024-03-12

## License

This software is proprietary and confidential. See the [LICENSE](LICENSE) file for details.

Copyright (c) 2024 Stardrive Inc. All Rights Reserved.

## UDL Integration

The AstroShield Integration Package includes a client for the Unified Data Library (UDL) API, which provides access to space situational awareness data. The integration supports both the REST API and Secure Messaging API.

### Authentication

The UDL client supports three authentication methods:

1. **API Key Authentication**: Set the `UDL_API_KEY` environment variable or pass the `api_key` parameter to the client constructor.
2. **Basic Authentication**: Set the `UDL_USERNAME` and `UDL_PASSWORD` environment variables or pass the `username` and `password` parameters to the client constructor.
3. **Token Authentication**: This method is currently not working with the UDL API.

For most users, Basic Authentication is the recommended method.

### Secure Messaging API

The Secure Messaging API provides real-time streaming access to UDL data. To use this feature:

1. **Request Access**: Special authorization is required for the Secure Messaging API. Contact the UDL team to request access by filling out the Secure Messaging Access Form.
2. **Enable in Configuration**: Set `use_secure_messaging=True` when initializing the `UDLIntegration` class.

Without proper authorization, attempts to access the Secure Messaging API will result in 403 Forbidden errors.

### Environment Variables

The following environment variables can be set to configure the UDL integration:

- `UDL_API_KEY`: API key for API Key Authentication
- `UDL_USERNAME`: Username for Basic Authentication
- `UDL_PASSWORD`: Password for Basic Authentication
- `UDL_BASE_URL`: Base URL for the UDL API (default: "https://unifieddatalibrary.com")

### API Limitations and Troubleshooting

When using the UDL API, be aware of the following:

1. **Required Parameters**: Many UDL API endpoints require specific parameters:
   - State Vectors: Requires an `epoch` parameter (e.g., "now" or an ISO-8601 timestamp)
   - Conjunctions: Requires an `epoch` parameter and possibly other filtering parameters
   - Launch Events: Requires a `msgCreateDate` parameter (ISO-8601 timestamp)

2. **Rate Limiting**: The UDL API may have rate limits. The Secure Messaging client is configured with a default sample period of 0.34 seconds (approximately 3 requests per second) to respect these limits.

3. **Error Handling**: Common errors include:
   - 400 Bad Request: Missing or invalid parameters
   - 401 Unauthorized: Invalid credentials
   - 403 Forbidden: Insufficient permissions (common for Secure Messaging API)
   - 500 Internal Server Error: Server-side issues

4. **Troubleshooting Steps**:
   - Verify your credentials are correct
   - Check that you're providing all required parameters for each endpoint
   - Ensure you have the necessary permissions for the resources you're trying to access
   - For Secure Messaging API issues, confirm you have been granted access

### UDL Integration Components

- **UDL Client**: A client for interacting with the UDL APIs
- **Data Transformers**: Functions for transforming UDL data to AstroShield format
- **Kafka Producer**: A producer for publishing transformed data to AstroShield Kafka topics
- **Integration Script**: A script for running continuous integration between UDL and AstroShield

### Using the UDL Integration

To use the UDL integration, you need to set up the required environment variables:

```bash
# UDL Authentication
export UDL_BASE_URL=https://unifieddatalibrary.com
export UDL_API_KEY=your-api-key
# or
export UDL_USERNAME=your-username
export UDL_PASSWORD=your-password

# Kafka Configuration
export KAFKA_BOOTSTRAP_SERVERS=kafka:9092
export KAFKA_SASL_USERNAME=your-username
export KAFKA_SASL_PASSWORD=your-password
```

Then you can run the integration:

```bash
python -m asttroshield.udl_integration.integration
```

### Testing the UDL Integration

The UDL integration includes comprehensive unit tests for all components:

- **Client Tests**: Tests for the UDL API client
- **Transformer Tests**: Tests for the data transformation functions
- **Kafka Producer Tests**: Tests for the Kafka message producer
- **Integration Tests**: Tests for the main integration module

To run the tests, use:

```bash
# Run all tests
pytest src/asttroshield/udl_integration/tests/

# Run specific test file
pytest src/asttroshield/udl_integration/tests/test_client.py
pytest src/asttroshield/udl_integration/tests/test_transformers.py
pytest src/asttroshield/udl_integration/tests/test_kafka_producer.py
pytest src/asttroshield/udl_integration/tests/test_integration.py
```

For more information, see the [UDL Integration README](src/asttroshield/udl_integration/README.md).

## Docker Deployment

The AstroShield integration package includes Docker deployment examples for a complete event-driven architecture:

1. **Build and start services:**
   ```bash
   docker-compose -f config/docker-compose.event-processor.yml up -d
   ```

2. **Monitor Kafka topics:**
   ```bash
   # View Kafka UI at http://localhost:8080
   ```

3. **Test with demo data:**
   ```bash
   docker-compose -f config/docker-compose.event-processor.yml exec event-processor python simple_demo.py 10 5
   ```

This Docker Compose setup includes:
- Event processor service for handling events from various sources
- Kafka for message streaming
- Zookeeper for Kafka coordination
- Kafka UI for monitoring and administration

For custom deployments, modify the environment variables in the docker-compose file to match your configuration needs. 