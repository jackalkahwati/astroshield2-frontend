# Kafka Topic Specifications

## Overview

This document provides detailed specifications for all Kafka topics used in the AstroShield platform. It includes information about topic naming conventions, message formats, schema versions, and usage patterns.

## Topic Naming Convention

Topics follow a hierarchical naming convention:
```
<subsystem>.<category>.<subcategory>[.<detail>]
```

- **subsystem**: The subsystem identifier (e.g., ss0, ss1, ss2)
- **category**: The general category (e.g., data, service, analysis)
- **subcategory**: More specific classification (e.g., weather, heartbeat)
- **detail**: Optional additional detail (e.g., contrails, turbulence)

## Message Structure

All messages follow a standard structure with:

1. **Header**: Contains metadata about the message
   - `messageId`: UUID to uniquely identify the message
   - `messageTime`: ISO 8601 timestamp of when the message occurred
   - `messageVersion`: Message header schema version
   - `subsystem`: Subsystem that produced the message
   - `dataProvider`: Provider or component within the subsystem
   - `dataType`: Payload type or topic name
   - `dataVersion`: Payload schema version
   - `customProperties`: Optional additional properties

2. **Payload**: Contains the actual data, structured according to the schema for that topic

## Topic List and Specifications

| Topic | Description | Schema Version | Implemented | Parent Topic |
|-------|-------------|----------------|------------|--------------|
| **Test Environment** |
| test-environment.ss0 | Test environment topic | 0.1.0 | Yes | No |
| test-environment.ss1 | Test environment topic | 0.1.0 | Yes | No |
| test-environment.ss2 | Test environment topic | 0.1.0 | Yes | No |
| test-environment.ss3 | Test environment topic | 0.1.0 | Yes | No |
| test-environment.ss4 | Test environment topic | 0.1.0 | Yes | No |
| test-environment.ss5 | Test environment topic | 0.1.0 | Yes | No |
| test-environment.ss6 | Test environment topic | 0.1.0 | Yes | No |
| **Subsystem 0 (SS0) - Sensor Data** |
| ss0.data.weather.contrails | Weather contrails data | 0.1.0 | Yes | No |
| ss0.data.weather.reflectivity | Weather reflectivity data | 0.1.0 | Yes | No |
| ss0.data.weather.turbulence | Weather turbulence data | 0.1.0 | Yes | No |
| ss0.data.weather.vtec | Weather VTEC data | 0.1.0 | Yes | No |
| ss0.data.weather.windshear | Weather windshear data | 0.1.0 | Yes | No |
| ss0.data.launch-detection | Launch detection messages (seismic, infrasound, overhead imagery, etc.) | 0.1.0 | Yes | No |
| ss0.launch-prediction.launch-window | Social media initial launch windows, refined launch windows by image/SAR analysis, weather go/no-go assessment | 0.1.0 | Yes | No |
| ss0.sensor.heartbeat | Sensor heartbeat and status message | 0.2.0 | Yes | No |
| **Subsystem 2 (SS2) - Analysis** |
| ss2.analysis.association-message | A proposed link between two data objects, with some probability associated | 0.1.0 | Yes | Yes |
| ss2.analysis.score-message | A proposed quality metric for a data object, particularly state vectors and elsets | 0.1.0 | Yes | Yes |
| ss2.data.candidate-object | Proposed catalog objects based on UCT or other data | 0.1.0 | Yes | Yes |
| ss2.data.elset.sgp4 | Two-line SGP4 elementsets associated to some RSO | 0.1.0 | Yes | Yes |
| ss2.data.elset.sgp4-xp | Two-line SGP4-XP element sets associated to some RSO | 0.1.0 | Yes | Yes |
| ss2.data.ephemeris | Collection of state vectors associated to a single RSO | 0.1.0 | Yes | Yes |
| ss2.data.observation-track | Collection of observations associated to a single RSO | 0.1.0 | Yes | Yes |
| ss2.data.orbit-determination | Descriptive data for how a linked state vector or elset was generated | 0.1.0 | Yes | Yes |
| ss2.data.resident-space-object | A single orbiting object with associated parameters (mass, area, etc) | 0.1.0 | Yes | Yes |
| ss2.data.state-vector | Position, velocity, time structure for the location of an RSO at some time | 0.1.0 | Yes | Yes |
| ss2.requests.generic-request | Topic to request specific toolchains or tools be run | 0.1.0 | Yes | Yes |
| ss2.responses.generic-response | Topic for responses to requests, tied directly to the original request message | 0.1.0 | Yes | Yes |
| ss2.service.events | Generic event class for logging service processes | 0.1.0 | Yes | TBD |
| ss2.service.heartbeat | Service heartbeat and status message | 0.1.0 | Yes | No |
| **Subsystem 3 (SS3) - Command** |
| ss3.command.change-request | C2 to task sensors for search and acquisition | 0.1.0 | TBD | Yes |
| **Subsystem 5 (SS5) - Launch** |
| ss5.launch-prediction.coplaner-assessment | Coplaner assessment of potential launch vehicle windows against blue assets | 0.1.0 | Yes | Yes |
| ss5.launch.prediction | Space launch prediction alerts | 0.1.0 | Yes | No |
| ss5.launch.detection | Space launch detection alerts | 0.1.0 | Yes | No |
| ss5.launch.nominal | Space launch nominals | 0.1.0 | Yes | No |
| ss5.launch.trajectory | Space launch vehicle trajectory prediction | 0.1.0 | Yes | No |
| ss5.service.heartbeat | SS5 services heartbeats | 0.1.0 | Yes | No |
| ss5.launch.asat-assessment | ASAT assessment on predicted and detected space launches | 0.1.0 | Yes | No |
| ss5.launch.weather-check | Go/No-go assessments for space launches based on weather | 0.1.0 | Yes | No |
| ss5.reentry.prediction | Object reentry predictions | 0.1.0 | Yes | No |
| **Subsystem 6 (SS6) - Response** |
| ss6.response-recommendation.launch | Response recommendations related to launch events | 0.1.0 | In-Progress | No |
| ss6.response-recommendation.on-orbit | Response recommendations related to on-orbit threats/concerns | 0.1.0 | In-Progress | No |

## Schema Versioning

Schemas are versioned using semantic versioning (MAJOR.MINOR.PATCH):
- MAJOR: Incompatible changes
- MINOR: Backwards-compatible additions
- PATCH: Backwards-compatible bug fixes

Schema files are stored in the `src/welders_arc_schemas/schemas/` directory with the following structure:
```
schemas/
  ├── <subsystem>/
  │   ├── <category>/
  │   │   ├── <subcategory>/
  │   │   │   ├── <version>.json
  │   │   │   └── examples/
  │   │   │       └── example.json
```

## Producer and Consumer Guidelines

### Producers
1. Always validate messages against the schema before publishing
2. Include all required fields in the message header
3. Use the latest schema version when possible
4. Set appropriate retention policies based on data criticality

### Consumers
1. Validate incoming messages against the schema
2. Handle schema version changes gracefully
3. Implement proper error handling for malformed messages
4. Use consumer groups for load balancing when appropriate

## Security and Access Control

1. All Kafka connections use SASL_SSL with SCRAM-SHA-512 authentication
2. Credentials are stored in environment variables, never in code
3. Topic-level access control is implemented through ACLs
4. Regular credential rotation is enforced

## Monitoring and Management

1. Topic metrics are available through the monitoring dashboard
2. Failed message deliveries are logged and alerted
3. Consumer lag is monitored to ensure timely processing
4. Schema registry ensures compatibility between producers and consumers

## Example Messages

See the `examples` directory within each schema folder for sample messages. 