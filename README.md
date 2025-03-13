# AstroShield Platform

The AstroShield Platform is a comprehensive solution for Space Domain Awareness (SDA) that integrates sensors, analytics, and threat assessment to protect space assets. This platform follows a microservices architecture organized around core subsystems that handle different aspects of the SDA mission.

## Repository Structure

The repository is organized into the following main components:

- **src/asttroshield/**: Core AstroShield subsystem architecture and components
- **src/kafka_client/**: Kafka producer and consumer examples
- **astroshield-integration-package/**: Partner integration package with schemas and examples
- **examples/**: Usage examples for various platform capabilities
- **ui/**: User interface components for visualization and interaction
- **tools/**: Utility scripts and development tools

## Core Architecture

AstroShield implements a subsystem-based architecture as defined in the SDA specifications. The system consists of the following subsystems:

### Subsystems

1. **Subsystem 0 (SS0) - Data Ingestion**: Handles raw data ingestion from sensors and external sources
2. **Subsystem 1 (SS1) - Target Modeling**: Processes ingested data to create and update target models
3. **Subsystem 2 (SS2) - State Estimation**: Tracks objects and estimates their current and future states
4. **Subsystem 3 (SS3) - Command & Control (C2)**: Manages mission control and decision support
5. **Subsystem 4 (SS4) - CCDM Detection**: Detects Camouflage, Concealment, Deception, and Maneuvering
6. **Subsystem 5 (SS5) - Hostility Monitoring**: Monitors for potentially hostile actions
7. **Subsystem 6 (SS6) - Threat Assessment**: Analyzes detected activities and provides threat assessments

## Message-Driven Architecture

The AstroShield platform is built on a message-driven architecture using Kafka as the messaging backbone. All subsystems communicate through standardized message formats that include:

- **Headers**: Metadata about the message, including trace information
- **Payloads**: The actual content of the message (schema-validated)

### Message Traceability

A key feature of the AstroShield platform is message traceability. Every message includes:

- `messageId`: A unique identifier for each message
- `traceId`: An identifier that remains consistent through an entire message chain
- `parentMessageIds`: References to messages that triggered the current message

This traceability provides complete visibility into message flow throughout the system, making it easier to debug, audit, and validate system behavior.

## Getting Started

### Prerequisites

- Python 3.8+
- Kafka cluster (we use Confluent Cloud in production)
- Environment variables for Kafka configuration

### Setting Up the Environment

1. Clone this repository
   ```
   git clone https://github.com/your-org/astroshield.git
   ```

2. Install dependencies
   ```
   pip install -r requirements.txt
   ```

3. Copy the `.env.example` file to `.env` and fill in your Kafka credentials

### Running the Example Application

The example application demonstrates how multiple subsystems can work together:

```
python src/example_app.py --all
```

Or run a specific subsystem:

```
python src/example_app.py --subsystem 0
```

## Integration Package

For partners looking to integrate with AstroShield, refer to the `astroshield-integration-package` directory, which contains:

- Message schemas for all subsystems
- Example code for consuming and producing messages
- Documentation on integration patterns

## Recent Updates

- **Standardized Message Structure**: All messages now use a consistent structure with headers for traceability
- **Subsystem Architecture**: Implemented the complete subsystem-based architecture according to SDA specifications
- **Message Tracing**: Added comprehensive message tracing across all subsystems
- **Kafka Utilities**: Created utility classes for standardized Kafka interaction
- **Example Application**: Added a complete example application demonstrating subsystem coordination

## License

This software is proprietary and confidential. Refer to the LICENSE file for details.

## Contact

For questions or support, contact the AstroShield development team at support@example.com.


Updated: Mon Jan 13 16:20:52 PST 2025
