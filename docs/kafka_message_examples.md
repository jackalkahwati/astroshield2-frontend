# Kafka Message Schema Examples

This document provides examples of the message schemas used in the AstroShield Kafka topics. These examples can be used as references when producing or consuming messages.

## Common Message Header

All messages include a standard header structure:

```json
{
  "messageId": "550e8400-e29b-41d4-a716-446655440000",
  "messageTime": "2023-05-01T12:00:00Z",
  "messageVersion": "1.0.0",
  "subsystem": "ss0",
  "dataProvider": "sensor_network_alpha",
  "dataType": "ss0.sensor.heartbeat",
  "dataVersion": "0.2.0",
  "customProperties": {
    "priority": "high",
    "environment": "production"
  }
}
```

## Sensor Heartbeat (ss0.sensor.heartbeat)

```json
{
  "header": {
    "messageId": "550e8400-e29b-41d4-a716-446655440001",
    "messageTime": "2023-05-01T12:00:00Z",
    "messageVersion": "1.0.0",
    "subsystem": "ss0",
    "dataProvider": "ground_station_1",
    "dataType": "ss0.sensor.heartbeat",
    "dataVersion": "0.2.0"
  },
  "payload": {
    "timeHeartbeat": "2023-05-01T12:00:00Z",
    "idSensor": "ground_radar_001",
    "status": "OPERATIONAL",
    "eo": {
      "sunlit": true,
      "overcastRatio": 0.25
    },
    "description": "Ground radar station operating normally"
  }
}
```

## Weather Data (ss0.data.weather.turbulence)

```json
{
  "header": {
    "messageId": "550e8400-e29b-41d4-a716-446655440002",
    "messageTime": "2023-05-01T12:05:00Z",
    "messageVersion": "1.0.0",
    "subsystem": "ss0",
    "dataProvider": "weather_service",
    "dataType": "ss0.data.weather.turbulence",
    "dataVersion": "0.1.0"
  },
  "payload": {
    "timestamp": "2023-05-01T12:05:00Z",
    "location": {
      "latitude": 28.5,
      "longitude": -80.65,
      "altitude": 10000
    },
    "turbulenceLevel": "moderate",
    "eddy_dissipation_rate": 0.35,
    "confidence": 0.85,
    "source": "aircraft_report"
  }
}
```

## Launch Detection (ss0.data.launch-detection)

```json
{
  "header": {
    "messageId": "550e8400-e29b-41d4-a716-446655440003",
    "messageTime": "2023-05-01T14:30:00Z",
    "messageVersion": "1.0.0",
    "subsystem": "ss0",
    "dataProvider": "infrared_satellite_network",
    "dataType": "ss0.data.launch-detection",
    "dataVersion": "0.1.0"
  },
  "payload": {
    "detectionTime": "2023-05-01T14:30:00Z",
    "detectionId": "LD-20230501-001",
    "confidence": 0.95,
    "location": {
      "latitude": 40.75,
      "longitude": 64.55,
      "altitude": 0
    },
    "detectionMethod": "infrared",
    "heatSignature": {
      "intensity": 8.5,
      "spectralProfile": "solid_rocket_booster"
    },
    "estimatedTrajectory": {
      "azimuth": 95.5,
      "elevation": 85.2,
      "initialVelocity": 150
    },
    "potentialVehicleType": ["ICBM", "SLV"],
    "images": [
      {
        "url": "https://data.astroshield.com/detections/LD-20230501-001/image1.jpg",
        "timestamp": "2023-05-01T14:30:05Z",
        "type": "infrared"
      }
    ]
  }
}
```

## State Vector (ss2.data.state-vector)

```json
{
  "header": {
    "messageId": "550e8400-e29b-41d4-a716-446655440004",
    "messageTime": "2023-05-01T15:00:00Z",
    "messageVersion": "1.0.0",
    "subsystem": "ss2",
    "dataProvider": "orbit_determination_service",
    "dataType": "ss2.data.state-vector",
    "dataVersion": "0.1.0"
  },
  "payload": {
    "objectId": "2023-001A",
    "noradId": 54321,
    "epoch": "2023-05-01T15:00:00Z",
    "frame": "GCRF",
    "position": {
      "x": 5000.123,
      "y": 4000.456,
      "z": 1000.789,
      "units": "km"
    },
    "velocity": {
      "x": 5.123,
      "y": -4.456,
      "z": 2.789,
      "units": "km/s"
    },
    "covariance": [
      [1.0e-6, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 1.0e-6, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 1.0e-6, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 1.0e-9, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 1.0e-9, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 1.0e-9]
    ],
    "metadata": {
      "source": "radar_tracking",
      "quality": 0.95,
      "maneuver_detected": false
    }
  }
}
```

## Resident Space Object (ss2.data.resident-space-object)

```json
{
  "header": {
    "messageId": "550e8400-e29b-41d4-a716-446655440005",
    "messageTime": "2023-05-01T15:05:00Z",
    "messageVersion": "1.0.0",
    "subsystem": "ss2",
    "dataProvider": "catalog_service",
    "dataType": "ss2.data.resident-space-object",
    "dataVersion": "0.1.0"
  },
  "payload": {
    "objectId": "2023-001A",
    "noradId": 54321,
    "internationalDesignator": "2023-001A",
    "name": "Example Satellite",
    "type": "PAYLOAD",
    "launchDate": "2023-01-01T00:00:00Z",
    "country": "USA",
    "status": "ACTIVE",
    "physicalProperties": {
      "mass": 1200.5,
      "length": 5.2,
      "width": 2.4,
      "height": 2.1,
      "crossSectionalArea": 12.48,
      "dragCoefficient": 2.2,
      "reflectivity": 0.3
    },
    "orbitType": "LEO",
    "metadata": {
      "mission": "Earth Observation",
      "operator": "Example Space Agency",
      "lastUpdated": "2023-05-01T15:00:00Z"
    }
  }
}
```

## Conjunction Analysis (ss2.analysis.association-message)

```json
{
  "header": {
    "messageId": "550e8400-e29b-41d4-a716-446655440006",
    "messageTime": "2023-05-01T16:00:00Z",
    "messageVersion": "1.0.0",
    "subsystem": "ss2",
    "dataProvider": "conjunction_analysis_service",
    "dataType": "ss2.analysis.association-message",
    "dataVersion": "0.1.0"
  },
  "payload": {
    "associationId": "ASSOC-20230501-001",
    "primaryObjectId": "2023-001A",
    "primaryNoradId": 54321,
    "secondaryObjectId": "1999-025DEB",
    "secondaryNoradId": 25544,
    "associationType": "CONJUNCTION",
    "timeOfClosestApproach": "2023-05-02T10:15:00Z",
    "missDistance": {
      "value": 0.125,
      "units": "km"
    },
    "relativeVelocity": {
      "value": 12.5,
      "units": "km/s"
    },
    "collisionProbability": 0.0000125,
    "associationConfidence": 0.98,
    "analysisMethod": "MONTE_CARLO",
    "analysisTime": "2023-05-01T16:00:00Z",
    "metadata": {
      "analyst": "automated",
      "actionRequired": true,
      "severity": "HIGH"
    }
  }
}
```

## Launch Prediction (ss5.launch.prediction)

```json
{
  "header": {
    "messageId": "550e8400-e29b-41d4-a716-446655440007",
    "messageTime": "2023-05-01T08:00:00Z",
    "messageVersion": "1.0.0",
    "subsystem": "ss5",
    "dataProvider": "launch_intelligence",
    "dataType": "ss5.launch.prediction",
    "dataVersion": "0.1.0"
  },
  "payload": {
    "predictionId": "LP-20230501-001",
    "launchSite": {
      "name": "Baikonur Cosmodrome",
      "latitude": 45.9644,
      "longitude": 63.3053,
      "country": "Kazakhstan"
    },
    "predictedLaunchWindow": {
      "start": "2023-05-05T14:00:00Z",
      "end": "2023-05-05T18:00:00Z"
    },
    "confidence": 0.85,
    "vehicleType": "Soyuz-2.1b",
    "payload": {
      "type": "SATELLITE",
      "name": "Example Communication Satellite",
      "estimatedMass": 5500,
      "estimatedDimensions": {
        "length": 8.5,
        "diameter": 3.2
      }
    },
    "targetOrbit": {
      "type": "GEO",
      "inclination": 0.1,
      "altitude": 35786
    },
    "intelligenceSources": [
      "OPEN_SOURCE",
      "SATELLITE_IMAGERY",
      "NOTAMS"
    ],
    "threatAssessment": {
      "level": "LOW",
      "militaryPurpose": false,
      "asatCapability": false
    }
  }
}
```

## Response Recommendation (ss6.response-recommendation.launch)

```json
{
  "header": {
    "messageId": "550e8400-e29b-41d4-a716-446655440008",
    "messageTime": "2023-05-01T16:30:00Z",
    "messageVersion": "1.0.0",
    "subsystem": "ss6",
    "dataProvider": "response_recommendation_engine",
    "dataType": "ss6.response-recommendation.launch",
    "dataVersion": "0.1.0"
  },
  "payload": {
    "recommendationId": "REC-20230501-001",
    "triggerEvent": {
      "type": "LAUNCH_DETECTION",
      "id": "LD-20230501-001",
      "time": "2023-05-01T14:30:00Z"
    },
    "threatAssessment": {
      "level": "MEDIUM",
      "confidence": 0.75,
      "impactTimeframe": "HOURS",
      "potentialTargets": [
        {
          "assetId": "SAT-001",
          "assetName": "Military Communications Satellite",
          "riskLevel": "HIGH"
        }
      ]
    },
    "recommendations": [
      {
        "id": "REC-20230501-001-1",
        "action": "INCREASE_MONITORING",
        "description": "Increase monitoring frequency of potentially affected assets",
        "priority": "HIGH",
        "timeframe": "IMMEDIATE"
      },
      {
        "id": "REC-20230501-001-2",
        "action": "NOTIFY_COMMAND",
        "description": "Notify Space Command of potential threat",
        "priority": "HIGH",
        "timeframe": "IMMEDIATE"
      },
      {
        "id": "REC-20230501-001-3",
        "action": "PREPARE_MANEUVER",
        "description": "Prepare evasive maneuver plans for potentially affected assets",
        "priority": "MEDIUM",
        "timeframe": "WITHIN_1_HOUR"
      }
    ],
    "justification": "Launch trajectory analysis indicates potential close approach to critical assets",
    "automatedActions": [
      {
        "id": "ACTION-20230501-001",
        "type": "ALERT",
        "status": "COMPLETED",
        "timestamp": "2023-05-01T14:31:00Z"
      }
    ]
  }
}
```

## Generic Request (ss2.requests.generic-request)

```json
{
  "header": {
    "messageId": "550e8400-e29b-41d4-a716-446655440009",
    "messageTime": "2023-05-01T17:00:00Z",
    "messageVersion": "1.0.0",
    "subsystem": "ss2",
    "dataProvider": "mission_control",
    "dataType": "ss2.requests.generic-request",
    "dataVersion": "0.1.0"
  },
  "payload": {
    "requestId": "REQ-20230501-001",
    "requestType": "PROPAGATION",
    "priority": "NORMAL",
    "requester": "mission_planning_system",
    "parameters": {
      "objectId": "2023-001A",
      "noradId": 54321,
      "startTime": "2023-05-01T17:00:00Z",
      "endTime": "2023-05-02T17:00:00Z",
      "stepSize": 60,
      "propagator": "SGP4",
      "includeCovariance": true
    },
    "responseTopics": [
      "ss2.responses.generic-response"
    ],
    "metadata": {
      "purpose": "mission_planning",
      "urgency": "routine"
    }
  }
}
```

## Generic Response (ss2.responses.generic-response)

```json
{
  "header": {
    "messageId": "550e8400-e29b-41d4-a716-446655440010",
    "messageTime": "2023-05-01T17:01:00Z",
    "messageVersion": "1.0.0",
    "subsystem": "ss2",
    "dataProvider": "propagation_service",
    "dataType": "ss2.responses.generic-response",
    "dataVersion": "0.1.0"
  },
  "payload": {
    "responseId": "RESP-20230501-001",
    "requestId": "REQ-20230501-001",
    "status": "COMPLETED",
    "processingTime": 0.85,
    "results": {
      "propagatedStates": [
        {
          "epoch": "2023-05-01T17:00:00Z",
          "position": {
            "x": 5000.123,
            "y": 4000.456,
            "z": 1000.789,
            "units": "km"
          },
          "velocity": {
            "x": 5.123,
            "y": -4.456,
            "z": 2.789,
            "units": "km/s"
          }
        },
        {
          "epoch": "2023-05-01T18:00:00Z",
          "position": {
            "x": 4500.123,
            "y": 4500.456,
            "z": 1200.789,
            "units": "km"
          },
          "velocity": {
            "x": 5.223,
            "y": -4.356,
            "z": 2.689,
            "units": "km/s"
          }
        }
      ],
      "metadata": {
        "propagator": "SGP4",
        "stepSize": 60,
        "totalSteps": 24
      }
    },
    "errors": [],
    "warnings": []
  }
}
```

## Schema Validation

To validate messages against their schemas, you can use the JSON Schema validation tools:

```python
from jsonschema import validate
import json

# Load the schema
with open('schemas/ss0/sensor/heartbeat/0.2.0.json', 'r') as f:
    schema = json.load(f)

# Load the message
with open('message.json', 'r') as f:
    message = json.load(f)

# Validate
try:
    validate(instance=message['payload'], schema=schema)
    print("Message validation: SUCCESS")
except Exception as e:
    print(f"Message validation: FAILED - {e}")
``` 