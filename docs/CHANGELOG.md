# Changelog

## Version 2.0.0 - Message Traceability and Subsystem Architecture

### Major Changes

#### Core Architecture
- Implemented a new subsystem-based architecture following SDA specifications
- Created base classes for all subsystems with standardized interfaces
- Added message traceability throughout the entire system
- Developed a standardized message format with headers and payloads

#### New Components
- **Message Headers**: Added `MessageFactory` for creating and deriving messages with traceability
- **Kafka Utilities**: Created `KafkaConfig`, `AstroShieldProducer`, and `AstroShieldConsumer` classes
- **Subsystem Base**: Implemented abstract base classes for all subsystems
- **Example Application**: Added a comprehensive example showing subsystem coordination

#### Documentation
- Added detailed README files explaining the new architecture
- Created a migration guide for updating existing code
- Added comprehensive code examples for each subsystem
- Updated main README with architecture overview

### File Changes

#### New Files
- `src/asttroshield/common/message_headers.py` - Message header utilities
- `src/asttroshield/common/kafka_utils.py` - Kafka integration utilities
- `src/asttroshield/common/subsystem.py` - Subsystem base classes
- `src/asttroshield/README.md` - Core architecture documentation
- `src/example_app.py` - Example application demonstrating subsystems
- `examples/traceability_demo.py` - End-to-end demo of message tracing
- `docs/migration_guide.md` - Guide for migrating to the new architecture
- `astroshield-integration-package/schemas/ss4.ccdm.detection.schema.json` - CCDM detection schema

#### Updated Files
- `src/kafka_client/kafka_publish.py` - Updated to use new message structure
- `src/kafka_client/kafka_consume.py` - Updated to handle message tracing
- `tests/test_ccdm_service.py` - Added tests for message tracing
- `README.md` - Updated with new architecture information
- `astroshield-integration-package/README.md` - Updated with subsystem information

### Integration Package Updates
- Added CCDM detection schema based on SDA transcripts
- Updated README with subsystem structure information
- Added Python example for consuming CCDM detection messages
- Enhanced documentation with traceability information

## Version 1.0.0 - Initial Release

- Initial AstroShield platform release
- Basic Kafka integration
- Simple message structure
- Preliminary SDA capabilities 