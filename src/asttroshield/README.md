# AstroShield Core Architecture

This directory contains the core components of the AstroShield platform, organized according to the subsystem architecture defined in the SDA transcripts.

## Subsystem Architecture

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

## Message Flow

Messages flow through the system in a logical progression from raw data ingestion to final assessment:

1. **Raw Data Ingestion** (Subsystem 0)
   - Sensor data is ingested from ground stations, satellites, and external sources
   - Raw data is processed and converted to standardized formats
   - Output: `ss0.sensor.observation`

2. **Object Identification** (Subsystem 1)
   - Correlates observations with known objects
   - Identifies and catalogs new objects
   - Output: `ss1.object.identification`

3. **State Determination** (Subsystem 2)
   - Processes observation data to generate state vectors
   - Performs orbit determination and propagation
   - Output: `ss2.data.state-vector`

4. **Anomaly Detection** (Subsystem 4)
   - Analyzes state vectors and behavior patterns
   - Detects CCDM activities and anomalous behaviors
   - Output: `ss4.ccdm.detection`

5. **Threat Monitoring** (Subsystem 5)
   - Monitors for specific threat types (conjunctions, cyber, etc.)
   - Correlates multiple data sources for comprehensive monitoring
   - Output: Various topics (`ss5.*.*`)

6. **Assessment and Response** (Subsystem 6)
   - Integrates detection and monitoring data
   - Generates comprehensive threat assessments
   - Provides recommended actions
   - Output: `ss6.threat.assessment`

## Message Structure

All messages in the system follow a standardized structure with header and payload:

```json
{
  "header": {
    "messageId": "unique-uuid",
    "timestamp": "2023-03-15T12:34:56.789Z",
    "messageType": "state.vector",
    "source": "ss2_state_estimation",
    "priority": "NORMAL",
    "traceId": "trace-uuid",
    "parentMessageIds": ["parent-message-uuid"]
  },
  "payload": {
    // Message-specific content
  }
}
```

## Traceability

AstroShield implements message traceability to track data lineage through the system:

- Each message has a unique `traceId` that follows the processing chain
- Messages reference their parent messages via `parentMessageIds`
- This enables tracking of data from raw observation to final assessment

## Common Components

### Message Headers Module

The `common/message_headers.py` module provides classes for handling standardized message headers:

- `MessageHeader`: Handles message IDs, timestamps, and traceability
- `MessageFactory`: Creates standardized messages with proper headers

Example:

```python
from src.asttroshield.common.message_headers import MessageFactory

# Create a new message
message = MessageFactory.create_message(
    message_type="state.vector",
    source="my_component",
    payload={"objectId": "123", "position": {...}}
)

# Create a derived message (maintains traceability)
derived_message = MessageFactory.derive_message(
    parent_message=message,
    message_type="ccdm.detection",
    source="my_component",
    payload={"detectionId": "456", "objectId": "123"}
)
```

### Logging Utilities

The `common/logging_utils.py` module provides enhanced logging with trace context:

- `configure_logging()`: Sets up logging with trace ID support
- `get_logger()`: Gets a logger that includes trace IDs in logs
- `trace_context()`: Context manager for setting the current trace ID
- `trace_method()`: Decorator for adding trace context to methods

Example:

```python
from src.asttroshield.common.logging_utils import configure_logging, get_logger, trace_context

# Configure logging
configure_logging()
logger = get_logger(__name__)

# Log with trace context
with trace_context("my-trace-id"):
    logger.info("Processing message")  # Includes trace ID in log
```

### Kafka Utilities

The `common/kafka_utils.py` module provides standardized Kafka integration:

- `KafkaConfig`: Configuration helper for Kafka connections
- `AstroShieldProducer`: Producer for publishing standardized messages
- `AstroShieldConsumer`: Consumer for processing standardized messages

Example:

```python
from src.asttroshield.common.kafka_utils import KafkaConfig, AstroShieldProducer

# Create Kafka configuration
config = KafkaConfig(
    bootstrap_servers="localhost:9092",
    security_protocol="PLAINTEXT"
)

# Create producer
producer = AstroShieldProducer(config, "my_component")

# Publish a message
producer.publish(
    topic="my-topic",
    message_type="example.message",
    payload={"key": "value"}
)
```

### Subsystem Base Classes

The `common/subsystem.py` module provides base classes for implementing subsystems:

- `SubsystemBase`: Abstract base class for all subsystems
- `Subsystem0` through `Subsystem6`: Specific implementations for each subsystem

Example of creating a custom subsystem:

```python
from src.asttroshield.common.subsystem import Subsystem4
from src.asttroshield.common.logging_utils import trace_context

class MyCCDMSubsystem(Subsystem4):
    @trace_context
    def process_message(self, message):
        # Process incoming message
        header = message.get("header", {})
        payload = message.get("payload", {})
        
        # Generate a detection
        detection_payload = {...}
        
        # Publish derived message (maintains traceability)
        self.derive_message(
            parent_message=message,
            message_type="ccdm.detection",
            payload=detection_payload,
            topic="ss4.ccdm.detection"
        )
```

## Example Application

An example application demonstrating the subsystem architecture is provided in `src/example_app.py`. This application:

1. Sets up multiple subsystems
2. Demonstrates message flow through the system
3. Shows how traceability is maintained
4. Provides examples of implementing custom processing logic

To run the example:

```bash
# Run all subsystems
python src/example_app.py

# Run a specific subsystem
python src/example_app.py --subsystem 0  # Data Ingestion
python src/example_app.py --subsystem 1  # Target Modeling
python src/example_app.py --subsystem 2  # State Estimation
python src/example_app.py --subsystem 4  # CCDM Detection
python src/example_app.py --subsystem 6  # Threat Assessment
```

## Integration with Existing Code

When integrating existing code with this architecture:

1. Wrap your existing code in the appropriate subsystem class
2. Ensure all messages use the standardized header format
3. Maintain traceability by using `derive_message()` for derived messages
4. Use the logging utilities to include trace IDs in logs 