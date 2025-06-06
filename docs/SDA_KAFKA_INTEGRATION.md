# AstroShield SDA Kafka Message Bus Integration

## Overview

This document describes the integration between AstroShield and the SDA (Space Development Agency) Kafka message bus, enabling real-time event-driven architecture for orbital intelligence and space domain awareness.

## Features

- **AWS MSK Connectivity**: Direct connection to SDA's managed Kafka infrastructure
- **Secure Authentication**: SASL_SSL with SCRAM-SHA-512 mechanism
- **IP Whitelisting Support**: Designed for SDA's secure network environment
- **Test Environment**: Comprehensive test topic support for development
- **Message Size Optimization**: Configurable limits for efficient data transfer
- **Orbital Intelligence Integration**: Native support for AstroShield's 15 AI models

## Quick Start

### 1. Prerequisites

```bash
# Install required dependencies
pip install confluent-kafka pydantic

# Ensure you have SDA Kafka credentials
# Contact Fernando or subsystem administrators for:
# - Username and password
# - IP whitelisting approval
# - Topic access permissions
```

### 2. Environment Setup

```bash
# Create environment template
python test_sda_kafka_integration.py template

# Edit .env.sda_kafka with your credentials
SDA_KAFKA_USERNAME=your-username-here
SDA_KAFKA_PASSWORD=your-password-here
```

### 3. Validate Setup

```bash
# Validate configuration and connectivity
python test_sda_kafka_integration.py validate
```

### 4. Run Demo

```bash
# Run interactive demonstration
python test_sda_kafka_integration.py demo
```

## Architecture

### SDA Topic Structure

The SDA Kafka message bus follows a hierarchical topic naming convention:

```
{subsystem}.{category}.{subcategory}
```

#### Production Topics

| Category | Topic | Description |
|----------|-------|-------------|
| **Launch Detection** |
| launch_detection | SS5.launch.detection | Real-time launch event detection |
| launch_prediction | SS5.launch.prediction | Launch window predictions |
| launch_analysis | SS5.launch.analysis | Post-launch analysis results |
| **Maneuver Detection** |
| maneuver_detection | SS2.maneuver.detection | Satellite maneuver events |
| maneuver_classification | SS2.maneuver.classification | Maneuver type classification |
| intent_assessment | SS5.intent.assessment | Intent analysis results |
| **Orbital Intelligence** |
| tle_update | SS1.tle.update | TLE data updates |
| orbital_analysis | SS1.orbital.analysis | Orbital analysis results |
| conjunction_warning | SS4.conjunction.warning | Collision warnings |
| **Threat Assessment** |
| threat_assessment | SS5.threat.assessment | Threat analysis results |
| hostility_score | SS5.hostility.score | Hostility scoring |
| threat_response | SS6.threat.response | Response coordination |
| **CCDM Events** |
| ccdm_detection | SS4.ccdm.detection | CCDM detection events |
| ccdm_analysis | SS4.ccdm.analysis | CCDM analysis results |
| ccdm_correlation | SS4.ccdm.correlation | Event correlation |

#### Test Topics

For development and testing, use test topic variants:

- `test.launch.detection`
- `test.maneuver.detection`
- `test.tle.update`
- `test.general.message`

### Message Schema

#### Official SDA Schemas

AstroShield now supports official SDA schemas from the Welders Arc GitLab repository:

##### 1. SDA SS4 Maneuvers-Detected Schema

```json
{
  "source": "astroshield",
  "satNo": "12345",
  "createdAt": "2024-01-01T12:00:00Z",
  "eventStartTime": "2024-01-01T11:55:00Z",
  "eventStopTime": "2024-01-01T12:00:00Z",
  "preCov": [[100.0, 0.0, ...], [0.0, 100.0, ...], ...],
  "prePosX": 6800.0,
  "prePosY": 0.0,
  "prePosZ": 0.0,
  "preVelX": 0.0,
  "preVelY": 7.5,
  "preVelZ": 0.0,
  "postCov": [[120.0, 0.0, ...], [0.0, 120.0, ...], ...],
  "postPosX": 6810.0,
  "postPosY": 0.0,
  "postPosZ": 0.0,
  "postVelX": 0.0,
  "postVelY": 7.52,
  "postVelZ": 0.0
}
```

**Required Fields**: `source`, `satNo`
**Optional Fields**: All position, velocity, covariance, and timestamp fields

##### 2. SDA SS5 Launch-Detected Schema

```json
{
  "source": "astroshield",
  "launchSite": "KSC",
  "vehicleId": "FALCON9-123",
  "payloadId": "STARLINK-BATCH-45",
  "launchTime": "2024-01-01T12:00:00Z",
  "confidence": 0.95
}
```

##### 3. SDA SS1 TLE-Update Schema

```json
{
  "source": "astroshield",
  "satelliteId": "STARLINK-1234",
  "catalogNumber": "44713",
  "line1": "1 44713U 19074A ...",
  "line2": "2 44713  53.0000 ...",
  "epoch": "2024-01-01T12:00:00Z",
  "inclination": 53.0,
  "eccentricity": 0.001234,
  "meanMotion": 15.12345678,
  "accuracy": 0.856,
  "confidence": 0.92
}
```

#### Generic AstroShield Schema

For non-schema-specific messages, AstroShield uses a generic wrapper:

```json
{
  "message_id": "uuid-string",
  "timestamp": "2024-01-01T12:00:00Z",
  "source_system": "astroshield",
  "subsystem": "SS1",
  "message_type": "tle_analysis",
  "priority": "normal",
  "correlation_id": "optional-correlation-id",
  "data": {
    // Message-specific payload
  },
  "metadata": {
    // Optional metadata
  }
}
```

#### Schema Factory Usage

```python
from asttroshield.sda_kafka.sda_schemas import SDASchemaFactory

# Create maneuver detection message
maneuver_msg = SDASchemaFactory.create_maneuver_detected(
    satellite_id="STARLINK-1234",
    source="astroshield",
    pre_position=[6800.0, 0.0, 0.0],
    pre_velocity=[0.0, 7.5, 0.0],
    post_position=[6810.0, 0.0, 0.0],
    post_velocity=[0.0, 7.52, 0.0]
)

# Create launch detection message
launch_msg = SDASchemaFactory.create_launch_detected(
    source="astroshield",
    launch_site="KSC",
    vehicle_id="FALCON9-123",
    confidence=0.95
)

# Create TLE update message
tle_msg = SDASchemaFactory.create_tle_update(
    satellite_id="STARLINK-1234",
    source="astroshield",
    catalog_number="44713",
    accuracy=0.856,
    confidence=0.92
)
```

## Usage Examples

### Basic Integration

```python
import asyncio
from asttroshield.sda_kafka import (
    SDAKafkaCredentials,
    AstroShieldSDAIntegration
)

async def main():
    # Initialize integration
    credentials = SDAKafkaCredentials.from_environment()
    sda_integration = AstroShieldSDAIntegration(credentials)
    await sda_integration.initialize()
    
    # Start consuming SDA events
    await sda_integration.start()

asyncio.run(main())
```

### Publishing TLE Analysis

```python
# Publish TLE analysis with orbital intelligence
tle_data = {
    "satellite_id": "STARLINK-1234",
    "line1": "1 44713U 19074A   21001.00000000 ...",
    "line2": "2 44713  53.0000 123.4567 ..."
}

analysis_results = {
    "accuracy_score": 0.856,      # OrbitAnalyzer-2.0
    "risk_level": "medium",       # ThreatScorer-1.0
    "satellite_type": "communication",  # ManeuverClassifier-1.5
    "maneuver_likelihood": 0.12,
    "threat_assessment": "low"
}

success = await sda_integration.publish_tle_analysis(
    tle_data, 
    analysis_results
)
```

### Publishing Maneuver Detection (Official SDA SS4 Schema)

```python
from datetime import datetime, timezone, timedelta
from asttroshield.sda_kafka.sda_schemas import SDASchemaFactory

# Official SDA maneuvers-detected schema format
maneuver_data = {
    "event_start_time": datetime.now(timezone.utc) - timedelta(minutes=5),
    "event_stop_time": datetime.now(timezone.utc),
    
    # Pre-maneuver state (position in km, velocity in km/s)
    "pre_position": [6800.0, 0.0, 0.0],  # km, ECI frame
    "pre_velocity": [0.0, 7.5, 0.0],     # km/s
    "pre_covariance": [
        [100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 100.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 100.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.01, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.01, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.01]
    ],
    
    # Post-maneuver state
    "post_position": [6810.0, 0.0, 0.0],  # km (altitude raised)
    "post_velocity": [0.0, 7.52, 0.0],     # km/s (delta-v applied)
    "post_covariance": [
        [120.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 120.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 120.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.012, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.012, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.012]
    ]
}

# Publish using official SDA schema
success = await sda_integration.publish_maneuver_detection(
    "STARLINK-1234",
    maneuver_data
)

# Alternative: Create schema directly
sda_message = SDASchemaFactory.create_maneuver_detected(
    satellite_id="STARLINK-1234",
    source="astroshield",
    pre_position=maneuver_data["pre_position"],
    pre_velocity=maneuver_data["pre_velocity"],
    post_position=maneuver_data["post_position"],
    post_velocity=maneuver_data["post_velocity"],
    event_start=maneuver_data["event_start_time"],
    event_stop=maneuver_data["event_stop_time"],
    pre_covariance=maneuver_data["pre_covariance"],
    post_covariance=maneuver_data["post_covariance"]
)
```

### Publishing Threat Assessment

```python
threat_data = {
    "target_id": "UNKNOWN-OBJECT-456",
    "threat_level": "high",
    "hostility_score": 0.78,
    "threat_indicators": [
        "proximity_to_critical_asset",
        "unusual_maneuver_pattern",
        "radio_emissions_detected"
    ],
    "assessment_confidence": 0.92,
    "recommended_response": "active_monitoring"
}

success = await sda_integration.publish_threat_assessment(threat_data)
```

### Consuming SDA Messages

```python
# Subscribe to SDA topics
def handle_tle_update(message):
    print(f"Received TLE update: {message.data}")

def handle_launch_detection(message):
    print(f"Launch detected: {message.data}")

# Setup subscriptions
sda_integration.kafka_client.subscribe("tle_update", handle_tle_update)
sda_integration.kafka_client.subscribe("launch_detection", handle_launch_detection)

# Start consuming
await sda_integration.start()
```

## Configuration

### Environment Variables

```bash
# === CONNECTION SETTINGS ===
SDA_KAFKA_BOOTSTRAP_SERVERS=kafka.sda.mil:9092
SDA_KAFKA_USERNAME=your-username-here
SDA_KAFKA_PASSWORD=your-password-here

# === SECURITY SETTINGS ===
SDA_KAFKA_SECURITY_PROTOCOL=SASL_SSL
SDA_KAFKA_SASL_MECHANISM=SCRAM-SHA-512
SDA_KAFKA_SSL_CA_LOCATION=/path/to/ca-cert.pem

# === CLIENT SETTINGS ===
SDA_KAFKA_CLIENT_ID=astroshield
SDA_KAFKA_GROUP_ID_PREFIX=astroshield

# === PERFORMANCE SETTINGS ===
SDA_KAFKA_MAX_MESSAGE_SIZE=1000000
SDA_KAFKA_BATCH_SIZE=100

# === TEST MODE ===
SDA_KAFKA_TEST_MODE=false
SDA_KAFKA_TEST_BOOTSTRAP_SERVERS=localhost:9092
```

### Message Size Optimization

Per SDA guidelines, optimize message sizes to avoid performance issues:

```python
# Configure maximum message size (default: 1MB)
os.environ['SDA_KAFKA_MAX_MESSAGE_SIZE'] = '1000000'

# Use compression
# Snappy compression is enabled by default for better performance
```

## Best Practices

### 1. Start with Test Environment

Always begin development with test topics:

```python
# Enable test mode
os.environ['SDA_KAFKA_TEST_MODE'] = 'true'

# This will automatically route to test.* topics
```

### 2. Follow Schema Validation

Ensure all messages conform to the SDA message schema:

```python
from asttroshield.sda_kafka import SDAMessageSchema, SDASubsystem, MessagePriority

message = SDAMessageSchema(
    source_system="astroshield",
    subsystem=SDASubsystem.SS1_TARGET_MODELING,
    message_type="tle_analysis",
    priority=MessagePriority.NORMAL,
    data=your_data
)
```

### 3. Handle Message Size Limits

If messages exceed size limits, consider:

```python
# Increase threshold for data publishing
# Split large datasets into smaller messages
# Use data compression where possible
```

### 4. Implement Proper Error Handling

```python
try:
    success = await sda_integration.publish_tle_analysis(tle_data, results)
    if not success:
        logger.error("Failed to publish to SDA")
except Exception as e:
    logger.error(f"SDA publishing error: {e}")
```

### 5. Monitor Consumer Lag

```python
# Implement consumer lag monitoring
# Set appropriate consumer group configurations
# Handle backpressure gracefully
```

## Integration with AstroShield Features

### TLE Chat Integration

The SDA Kafka integration works seamlessly with AstroShield's TLE chat feature:

```python
# TLE analysis from chat interface automatically publishes to SDA
# Results from 15 orbital intelligence models are included
# Real-time bidirectional communication with SDA systems
```

### Orbital Intelligence Models

Results from AstroShield's 15 specialized models are automatically included:

- **OrbitAnalyzer-2.0**: Fine-tuned TLE analysis (85.6% accuracy)
- **ManeuverClassifier-1.5**: Satellite maneuver detection
- **ThreatScorer-1.0**: Hostility likelihood assessment
- **IntelligenceKernel-1.0**: Unified model fusion

### Benchmarking Integration

Scientific benchmarking results are included in SDA messages:

```python
orbital_intelligence = {
    "orbital_accuracy": 0.856,
    "risk_assessment": 0.442,
    "satellite_recognition": 0.575,
    "model_confidence": 0.923,
    "benchmarking_source": "model_benchmark_results_20250605_110438.json"
}
```

## Troubleshooting

### Common Issues

#### 1. Connection Refused

**Issue**: Cannot connect to SDA Kafka bus

**Solutions**:
- Verify IP whitelisting with SDA team
- Check credentials (username/password)
- Confirm network connectivity to kafka.sda.mil:9092

#### 2. Authentication Failed

**Issue**: SASL authentication failures

**Solutions**:
- Verify username and password are correct
- Check SASL mechanism (SCRAM-SHA-512)
- Ensure credentials haven't expired

#### 3. Topic Access Denied

**Issue**: Cannot access specific topics

**Solutions**:
- Request topic access permissions from SDA administrators
- Verify topic names match SDA conventions
- Check ACL permissions for your user

#### 4. Message Size Exceeded

**Issue**: Messages rejected due to size

**Solutions**:
- Reduce message payload size
- Increase SDA_KAFKA_MAX_MESSAGE_SIZE if permitted
- Split large datasets into multiple messages

### Diagnostic Commands

```bash
# Validate complete setup
python test_sda_kafka_integration.py validate

# Check prerequisites
python -c "from src.asttroshield.sda_kafka.config import SDAKafkaSetup; SDAKafkaSetup.check_prerequisites()"

# Test connection
python test_sda_kafka_integration.py test

# View topic structure
python -c "from src.asttroshield.sda_kafka import SDATopicManager; print(SDATopicManager.list_topics())"
```

## Security Considerations

### 1. Credential Management

- Store credentials in environment variables, never in code
- Use secure credential management systems in production
- Rotate credentials regularly per SDA policy

### 2. Network Security

- Ensure proper firewall configurations
- Use VPN connections when required
- Verify SSL/TLS configurations

### 3. Message Security

- All messages are encrypted in transit (SASL_SSL)
- Sensitive data should be encrypted at the application level
- Follow SDA data classification guidelines

## Support and Resources

### Getting Help

1. **SDA Team Contacts**:
   - Fernando: Topic access and schemas
   - Subsystem administrators: IP whitelisting and permissions

2. **Documentation**:
   - SDA Welders Arc GitLab repository
   - Schema examples and publish/subscribe scripts

3. **AstroShield Integration**:
   - Use the test suite for validation
   - Check logs for detailed error information
   - Monitor Kafka consumer lag and performance

### Additional Resources

- [SDA Welders Arc Documentation](internal-link)
- [Kafka Best Practices](https://kafka.apache.org/documentation/)
- [AWS MSK Documentation](https://docs.aws.amazon.com/msk/)
- [AstroShield TLE Chat Integration](./TLE_CHAT_INTEGRATION.md)

## Changelog

### Version 1.0.0
- Initial SDA Kafka message bus integration
- Support for 15 orbital intelligence models
- Test environment and production topic routing
- Comprehensive error handling and logging
- Integration with AstroShield TLE chat features 