# Astroshield Event-Driven Architecture

This repository contains the implementation of an event-driven architecture for Astroshield, designed to respond to real-time events such as satellite maneuvers and changing weather conditions that affect observation windows.

## Key Features

- **Real-time maneuver detection** using DMD orbit determination data
- **Weather data integration** for observation planning
- **Event-driven processing** with Kafka for improved responsiveness
- **Retry mechanisms** for resilient messaging
- **Standardized message formats** for Kafka topics

## Components

- **DMD Orbit Determination Client**: Interfaces with the DMD API to detect maneuvers
- **Weather Data Service**: Analyzes weather conditions for optimal observation windows
- **Kafka Consumer**: Listens for events and routes them to appropriate handlers
- **Kafka Producer**: Publishes detected events with retry capability
- **Event Handlers**: Process specific types of events (maneuvers, weather conditions)

## Running the Demo

The demo script `demo_event_processor.py` simulates real-time processing of events without requiring a Kafka cluster. It demonstrates how the system responds to both DMD orbit determination updates and weather data updates.

```bash
# Run with default settings (6 events with 2 second delay)
python demo_event_processor.py

# Run with custom number of events and delay
python demo_event_processor.py 10 1  # 10 events with 1 second delay
```

The demo will alternate between DMD object updates and weather data updates, showing how the system processes them and publishes resulting events (maneuver detections and observation window recommendations).

## Running Tests

The tests verify the functionality of the event processing system:

```bash
# Run the test suite
python tests/test_event_processor.py
```

This runs a series of unit tests that verify:
- DMD maneuver detection logic
- Weather data integration and observation window recommendations
- End-to-end event flow

## Environment Variables

The actual production deployment uses the following environment variables:

- `KAFKA_BOOTSTRAP_SERVERS`: Comma-separated list of Kafka bootstrap servers
- `KAFKA_TOPIC_PREFIXES`: Comma-separated list of topic prefixes to subscribe to
- `KAFKA_CONSUMER_GROUP`: Consumer group ID
- `KAFKA_PRODUCER_CLIENT_ID`: Producer client ID
- `KAFKA_SECURITY_PROTOCOL`: Security protocol (PLAINTEXT, SSL, SASL_PLAINTEXT, SASL_SSL)
- `KAFKA_SASL_MECHANISM`: SASL mechanism if applicable
- `KAFKA_CONSUMER_USERNAME`: Username for consumer
- `KAFKA_CONSUMER_PASSWORD`: Password for consumer
- `KAFKA_PRODUCER_USERNAME`: Username for producer
- `KAFKA_PRODUCER_PASSWORD`: Password for producer
- `KAFKA_MAX_RETRIES`: Maximum retry attempts for publishing (default: 3)
- `KAFKA_RETRY_BACKOFF_MS`: Initial backoff time in ms (default: 100)
- `KAFKA_MAX_BACKOFF_MS`: Maximum backoff time in ms (default: 5000)
- `UDL_BASE_URL`: Base URL for UDL API
- `UDL_API_KEY`: API key for UDL authentication

## Kafka Topics

The system uses the following Kafka topics:

- Input:
  - `dmd-od-update`: DMD orbit determination updates
  - `weather-data`: Weather data updates

- Output:
  - `maneuvers-detected`: Detected satellite maneuvers
  - `observation-windows`: Recommended observation windows based on weather

## Message Formats

### Maneuver Detection Event

```json
{
  "header": {
    "messageType": "maneuver-detected",
    "source": "dmd-od-integration",
    "timestamp": "2023-06-01T12:34:56.789Z"
  },
  "payload": {
    "catalogId": "DMD-12345",
    "deltaV": 0.15,
    "confidence": 0.85,
    "maneuverType": "ORBIT_ADJUSTMENT", 
    "detectionTime": "2023-06-01T12:30:45.123Z"
  }
}
```

### Observation Window Event

```json
{
  "header": {
    "messageType": "observation-window-recommended",
    "source": "weather-integration",
    "timestamp": "2023-06-01T12:34:56.789Z"
  },
  "payload": {
    "location": {
      "latitude": 40.7128,
      "longitude": -74.0060
    },
    "qualityScore": 0.82,
    "qualityCategory": "EXCELLENT",
    "recommendation": "GO",
    "observationWindow": {
      "start_time": "2023-06-01T13:00:00.000Z",
      "end_time": "2023-06-01T14:00:00.000Z",
      "duration_minutes": 60
    },
    "targetObject": {
      "catalog_id": "SAT-5678",
      "altitude_km": 650.0
    }
  }
}
```

## Notes

- This implementation was developed in response to the technical meeting recommendations to transition to an event-driven architecture.
- The system is designed to be extensible, allowing new event types and handlers to be added in the future.
- The mock implementations simulate realistic behavior for testing and demonstration purposes.