# AstroShield Vantiq Integration

This module provides integration between the AstroShield platform and Vantiq, enabling seamless communication and data exchange between the two systems.

## Features

- Message transformation between AstroShield and Vantiq formats
- Kafka-based communication for reliable messaging
- Support for various message types (maneuvers, observations, object details)
- Robust error handling and logging

## Installation

```bash
cd src/asttroshield/vantiq_integration
npm install
```

## Configuration

Create a `.env` file in the root directory with the following variables:

```
# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_CLIENT_ID=astroshield-vantiq-integration
KAFKA_GROUP_ID=astroshield-vantiq-group

# Topics
ASTROSHIELD_OUTBOUND_TOPIC=astroshield-outbound
VANTIQ_INBOUND_TOPIC=vantiq-inbound
VANTIQ_OUTBOUND_TOPIC=vantiq-outbound
ASTROSHIELD_INBOUND_TOPIC=astroshield-inbound

# Logging
LOG_LEVEL=info
```

## Running the Service

```bash
# Development mode with auto-reload
npm run dev

# Production mode
npm start
```

## Testing

This package includes both unit tests and integration tests.

### Unit Tests

The unit tests test individual components in isolation, using mocks and stubs where necessary.

```bash
# Run unit tests
npm test
```

### Integration Tests

The integration tests test the interaction between components and systems, including Kafka connectivity.

**Prerequisites for Integration Tests:**
- Kafka server running and accessible
- Appropriate test topics created in Kafka

```bash
# Run integration tests
npm run test:integration

# Run all tests (unit + integration)
npm run test:all
```

## Testing Environment Setup

For integration tests, you need a Kafka server. You can use Docker to quickly set up a test environment:

```bash
# Start Kafka and Zookeeper
docker-compose up -d

# Create test topics
docker exec -it kafka kafka-topics.sh --create --topic astroshield-test-outbound --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
docker exec -it kafka kafka-topics.sh --create --topic vantiq-test-inbound --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
docker exec -it kafka kafka-topics.sh --create --topic vantiq-test-outbound --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
docker exec -it kafka kafka-topics.sh --create --topic astroshield-test-inbound --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
```

## Test File Structure

- `tests/unit/` - Unit tests for each component
  - `test_transformers.js` - Tests for message transformers
  - `test_validation.js` - Tests for message validation
  - ...

- `tests/integration/` - Integration tests
  - `test_kafka_integration.js` - Tests for Kafka connectivity and message flow
  - ...

- `tests/test_config.js` - Shared test configuration and helpers

## Development Guidelines

1. Write unit tests for all new functionality
2. Ensure integration tests cover key user flows
3. Run linting (`npm run lint`) before committing code
4. Follow the existing code style and patterns 