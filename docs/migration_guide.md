# Migration Guide: Moving to the New AstroShield Architecture

This guide explains how to migrate existing AstroShield code to the new subsystem-based architecture that provides enhanced message traceability and standardized Kafka integration.

## Table of Contents

1. [Overview of Changes](#overview-of-changes)
2. [Message Format Migration](#message-format-migration)
3. [Moving Code to Subsystems](#moving-code-to-subsystems)
4. [Kafka Client Migration](#kafka-client-migration)
5. [Testing your Migration](#testing-your-migration)
6. [Gradual Migration Strategy](#gradual-migration-strategy)

## Overview of Changes

The new AstroShield architecture introduces several significant changes:

1. **Subsystem Organization**: Code is now organized into numbered subsystems (SS0-SS6) that handle specific responsibilities
2. **Standardized Message Format**: All messages follow a consistent structure with headers and payloads
3. **Message Traceability**: Messages now include tracing information to track message flow
4. **Kafka Utilities**: Standardized utilities for Kafka interactions
5. **Base Classes**: Common base classes for subsystem implementation

## Message Format Migration

### Old Format

```json
{
  "timestamp": 1678901234,
  "message": "Sensor reading",
  "data": {
    "value": 42
  }
}
```

### New Format

```json
{
  "header": {
    "messageId": "550e8400-e29b-41d4-a716-446655440000",
    "timestamp": "2023-10-25T12:34:56.789Z",
    "source": "sensor_123",
    "messageType": "ss0.sensor.reading",
    "traceId": "550e8400-e29b-41d4-a716-446655440000",
    "parentMessageIds": []
  },
  "payload": {
    "timestamp": "2023-10-25T12:34:56.789Z",
    "message": "Sensor reading",
    "data": {
      "value": 42
    }
  }
}
```

### Migration Steps

1. **Update Message Creation**:
   
   Old code:
   ```python
   message = {
       "timestamp": time.time(),
       "message": "Sensor reading",
       "data": {
           "value": sensor_value
       }
   }
   ```

   New code:
   ```python
   from src.asttroshield.common.message_headers import MessageFactory

   payload = {
       "timestamp": datetime.utcnow().isoformat(),
       "message": "Sensor reading",
       "data": {
           "value": sensor_value
       }
   }
   
   message = MessageFactory.create_message(
       message_type="ss0.sensor.reading",
       source="sensor_123",
       payload=payload
   )
   ```

2. **Update Message Processing**:
   
   Old code:
   ```python
   def process_message(message):
       timestamp = message.get('timestamp')
       data = message.get('data', {})
       # Process data...
   ```

   New code:
   ```python
   def process_message(message):
       header = message.get('header', {})
       payload = message.get('payload', {})
       
       message_id = header.get('messageId')
       trace_id = header.get('traceId')
       message_type = header.get('messageType')
       
       # Process payload data
       data = payload.get('data', {})
       # Process data...
       
       # Optionally create a derived message
       derived_message = MessageFactory.create_derived_message(
           parent_message=message,
           message_type="ss1.target.update",
           source="target_processor",
           payload={"processed": True, "result": processed_data}
       )
   ```

## Moving Code to Subsystems

1. **Identify the appropriate subsystem** for your code based on its functionality:
   - SS0: Data ingestion from sensors and external sources
   - SS1: Target modeling from ingested data
   - SS2: State estimation and tracking
   - SS3: Command and control
   - SS4: CCDM detection
   - SS5: Hostility monitoring
   - SS6: Threat assessment

2. **Create a subsystem class** that extends the appropriate base class:

   ```python
   from src.asttroshield.common.subsystem import SubsystemBase
   
   class MySensorSubsystem(SubsystemBase):
       def __init__(self, config):
           super().__init__(subsystem_id=0, name="Sensor Subsystem")
           self.config = config
           
           # Register input and output topics
           self.register_input_topic("external.sensor.data")
           self.register_output_topic("ss0.sensor.processed")
       
       def process_message(self, message):
           # Process the message
           header = message.get('header', {})
           payload = message.get('payload', {})
           
           # Your processing logic here
           processed_data = self._process_sensor_data(payload)
           
           # Create a derived message
           result_message = self.derive_message(
               message, 
               "ss0.sensor.processed",
               {"processed": True, "data": processed_data}
           )
           
           # Publish the derived message
           self.publish_message("ss0.sensor.processed", result_message)
           
       def _process_sensor_data(self, payload):
           # Your existing processing logic
           # ...
           return processed_result
   ```

## Kafka Client Migration

### Producer Migration

Old code:
```python
producer = KafkaProducer(
    bootstrap_servers=bootstrap_servers,
    security_protocol=security_protocol,
    sasl_mechanism=sasl_mechanism,
    sasl_plain_username=username,
    sasl_plain_password=password,
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

message = {"timestamp": time.time(), "data": sensor_data}
producer.send(topic, value=message)
```

New code:
```python
from src.asttroshield.common.kafka_utils import KafkaConfig, AstroShieldProducer
from src.asttroshield.common.message_headers import MessageFactory

# Create Kafka configuration
kafka_config = KafkaConfig(
    bootstrap_servers=bootstrap_servers,
    security_protocol=security_protocol,
    sasl_mechanism=sasl_mechanism,
    producer_username=username,
    producer_password=password
)

# Create producer
producer = AstroShieldProducer(kafka_config)

# Create standardized message
message = MessageFactory.create_message(
    message_type="ss0.sensor.data",
    source="sensor_123",
    payload={"timestamp": datetime.utcnow().isoformat(), "data": sensor_data}
)

# Send message
producer.publish(topic, message)
```

### Consumer Migration

Old code:
```python
consumer = KafkaConsumer(
    topic,
    bootstrap_servers=bootstrap_servers,
    security_protocol=security_protocol,
    sasl_mechanism=sasl_mechanism,
    sasl_plain_username=username,
    sasl_plain_password=password,
    group_id=group_id,
    auto_offset_reset='earliest',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

for message in consumer:
    value = message.value
    # Process the message
    process_data(value)
```

New code:
```python
from src.asttroshield.common.kafka_utils import KafkaConfig, AstroShieldConsumer

# Create Kafka configuration
kafka_config = KafkaConfig(
    bootstrap_servers=bootstrap_servers,
    security_protocol=security_protocol,
    sasl_mechanism=sasl_mechanism,
    consumer_username=username,
    consumer_password=password
)

# Create consumer with message processor
consumer = AstroShieldConsumer(
    kafka_config,
    topics=[topic],
    group_id=group_id,
    process_message=process_data
)

# Start consuming (this will run in a loop until stopped)
consumer.start()

# When you want to stop
consumer.stop()
```

## Testing your Migration

1. **Unit Testing**:
   - Update your tests to use the new message structure
   - Use `MessageFactory.create_message()` in your test setup
   - Verify that message headers are correctly propagated

2. **Integration Testing**:
   - Use the example application as a reference
   - Test message flow between subsystems
   - Verify that traceIDs are properly maintained across messages

3. **Compatibility Testing**:
   - If you're migrating gradually, test that new code can still process old message formats
   - Ensure backward compatibility with existing systems

## Gradual Migration Strategy

For large systems, we recommend a phased migration approach:

1. **Phase 1: Message Format Adaptation**
   - Update your code to use the new message format with headers and payloads
   - Implement the MessageFactory for creating messages
   - Don't move code to subsystems yet

2. **Phase 2: Kafka Utilities Migration**
   - Replace direct Kafka client usage with the AstroShield Kafka utilities
   - Test thoroughly to ensure no message loss

3. **Phase 3: Subsystem Organization**
   - Gradually move code into appropriate subsystem classes
   - Start with one subsystem at a time, beginning with either the ingestion (SS0) or the final output (SS6)

4. **Phase 4: Full Migration**
   - Complete the migration of all code to the appropriate subsystems
   - Implement end-to-end message tracing
   - Deploy the fully migrated system

## Need Help?

If you encounter issues during migration, please contact the AstroShield platform team for assistance. 