# Kafka Standards Compliance for AstroShield

## Overview

AstroShield has been updated to comply with the Space Domain Awareness (SDA) Welders Arc standard Kafka topic naming convention and access control matrix. This ensures interoperability with other SDA ecosystem partners.

## Topic Naming Convention

All Kafka topics follow the standard format:
```
ss[0-6].<category>.<subcategory>
```

Where:
- **ss0** - Data Ingestion (sensors, weather, synthetic data)
- **ss1** - Target Modeling (TMDB updates, object attributes)
- **ss2** - State Estimation (tracks, orbits, state vectors)
- **ss3** - Command & Control (access windows, detection probability)
- **ss4** - CCDM (indicators, attribution, anomalies)
- **ss5** - Hostility Monitoring (launch, PEZ/WEZ, RPO)
- **ss6** - Threat Assessment (response recommendations, risk mitigation)

## AstroShield Access Permissions

Based on the official access matrix, AstroShield has the following topic permissions:

### Read Access (📖)
- `ss0.data.launch-detection`
- `ss0.launch-prediction.launch-window`
- `ss0.sensor.heartbeat`
- `ss1.indicators.capabilities-updated`
- `ss1.tmdb.object-inserted`
- `ss1.tmdb.object-updated`
- `ss2.data.state-vector.best-state`
- `ss3.data.accesswindow`
- `ss3.data.detectionprobability`
- `ss4.attributes.orbital-attribution`
- `ss4.ccdm.ccdm-db`
- `ss4.ccdm.ooi`
- `ss5.launch.detection`
- `ss5.launch.fused`
- `ss5.launch.prediction`
- `ss5.launch.trajectory`

### Write Access (📝)
- `ss5.launch.asat-assessment`
- `ss5.launch.trajectory`
- `ss5.polygon-closures`

### Both Read/Write Access (🅱)
- `ss2.analysis.association-message`
- `ss2.data.elset.uct-candidate`
- `ss2.data.observation-track`
- `ss2.data.observation-track.true-uct`
- `ss2.data.orbit-determination`
- `ss2.data.state-vector`
- `ss2.data.state-vector.uct-candidate`
- `ss4.indicators.imaging-maneauvers-pol-violations`
- `ss4.indicators.maneuvers-detected`
- `ss5.launch.asat-assessment`
- `ss5.launch.coplanar-assessment`
- `ss5.reentry.prediction`
- `ss6.risk-mitigation.optimal-maneuver`
- `ui.event`

## Implementation Details

### 1. **Topic Definitions**
All standard topics are defined in:
```
backend_fixed/app/sda_integration/kafka/standard_topics.py
```

### 2. **Access Control**
The Kafka client enforces access permissions at the application level:
- Read permissions are checked when subscribing to topics
- Write permissions are checked when publishing messages
- Unauthorized operations log warnings/errors but don't crash the system

### 3. **Backward Compatibility**
The `KafkaTopics` class provides aliases mapping old topic names to new standard names:
```python
# Old name -> New standard name
SENSOR_OBSERVATIONS = ss2.data.observation-track
STATE_VECTORS = ss2.data.state-vector
EVENT_LAUNCH_DETECTION = ss5.launch.detection
```

### 4. **Topic Creation**
The `create_kafka_topics.sh` script creates all standard topics with appropriate:
- Partition counts (1-5 based on expected throughput)
- Replication factors (1 for development, increase for production)
- Retention policies (7 days default)

## Usage Examples

### Publishing to a Topic
```python
from app.sda_integration.kafka import KafkaTopics, WeldersArcMessage

# Publishing launch detection (AstroShield has write access)
message = WeldersArcMessage(
    message_id="launch-123",
    timestamp=datetime.utcnow(),
    subsystem="ss5_hostility",
    event_type="launch",
    data={"launch_site": "Cape Canaveral", "azimuth": 45.0}
)
await kafka_client.publish(KafkaTopics.SS5_LAUNCH_TRAJECTORY, message)
```

### Subscribing to a Topic
```python
# Subscribing to state vectors (AstroShield has read access)
def handle_state_vector(message: WeldersArcMessage):
    print(f"Received state vector: {message.data}")

kafka_client.subscribe(KafkaTopics.SS2_DATA_STATE_VECTOR, handle_state_vector)
```

## Integration with Other Systems

AstroShield integrates with the broader SDA ecosystem through:

1. **UDL Integration**: Publishes transformed UDL data to standard topics
2. **Subsystem Communication**: Each subsystem publishes/consumes from appropriate topics
3. **Event Processing**: Standard event types trigger workflows across subsystems

## Monitoring and Compliance

### Topic Health Monitoring
- Monitor message throughput per topic
- Track consumer lag for subscribed topics
- Alert on access permission violations

### Compliance Verification
1. Run `./create_kafka_topics.sh` to ensure all topics exist
2. Check application logs for permission errors
3. Verify message flow through integration tests

## Migration Notes

For systems upgrading from previous versions:

1. **Update Topic References**: Replace old topic names with standard names
2. **Test Access Permissions**: Verify read/write operations work as expected
3. **Monitor Legacy Topics**: Keep old topics temporarily for rollback capability
4. **Update Documentation**: Ensure all integration guides use new topic names

## Security Considerations

- **Authentication**: Uses SASL/SSL for secure broker connections
- **Authorization**: Application-level enforcement of topic access matrix
- **Encryption**: TLS encryption for data in transit
- **Audit Logging**: All access violations are logged for security review

## Contact Information

For questions about Kafka standards compliance:
- **SDA Integration Team**: sda-integration@astroshield.com
- **Kafka Admin**: kafka-admin@astroshield.com
- **Security Team**: security@astroshield.com 