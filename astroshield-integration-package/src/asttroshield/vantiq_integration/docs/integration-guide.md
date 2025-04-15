# Astroshield-Vantiq Integration Guide

## Overview

This guide explains how to integrate Astroshield's space situational awareness data with Vantiq's event orchestration platform. The integration enables real-time processing of satellite maneuvers, observation opportunities, and other space events within Vantiq applications.

## Architecture

The integration uses Kafka as the primary message bus:

```ascii
┌───────────┐    ┌─────────┐    ┌────────────┐    ┌───────────┐
│ Astroshield│───▶│  Kafka  │───▶│ Vantiq Kafka│───▶│  Vantiq   │
│  Event    │    │  Topics │    │   Source   │    │  Platform │
│ Processor │    └─────────┘    └────────────┘    └───────────┘
└───────────┘                                          │
      │                                                │
      │        ┌───────────┐                           │
      └───────▶│Astroshield│◀──────────────────────────┘
               │   API     │     REST API Calls
               └───────────┘
```

## Setup Instructions

### 1. Configure Kafka Source in Vantiq

1. Navigate to the Vantiq IDE
2. Go to Sources → Create Source
3. Select KAFKA as the source type
4. Configure with the provided `vantiq-source-config.json`
5. Test the connection

### 2. Import Type Definitions

1. Import the ManeuverDetection and ObservationWindow types
2. These types map the Kafka events to Vantiq's type system

### 3. Create Processing Rules

1. Create rules using the templates in `rules/`
2. Customize the rules to match your business logic
3. Test with sample events

### 4. Configure REST API Access

1. Create a resource in Vantiq for API access
2. Configure authentication using your Astroshield API key
3. Test connectivity from Vantiq to Astroshield API

## Event Processing Patterns

### 1. Real-time Maneuver Detection

```vail
RULE ProcessManeuver
WHEN EVENT OCCURS FROM AstroshieldKafkaSource as message
WHERE message.topic == "maneuvers-detected"
// Processing logic
```

### 2. ML-Enhanced Analysis

```vail
RULE EnhancedManeuverAnalysis
WHEN EVENT OCCURS FROM TOPIC "/astroshield/maneuvers" as event
// Extract features for ML
var features = extractManeuverFeatures(event)
// Call ML service
var prediction = PROCEDURE CallMLService(features)
// Take action based on prediction
```

### 3. Multi-source Correlation

```vail
RULE CorrelateSpaceEvents
WHEN EVENT OCCURS FROM TOPIC "/astroshield/maneuvers" as maneuverEvent
JOIN (SELECT * FROM CyberEvent 
      WHERE targetSystem == maneuverEvent.catalogId
      AND timestamp BETWEEN 
          DateTime.subtract(maneuverEvent.detectionTime, "PT1H") AND
          maneuverEvent.detectionTime) as cyberEvent
// Correlated event processing
``` 