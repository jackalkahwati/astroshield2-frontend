# Astroshield to Vantiq Data Mapping

This document details how Astroshield event data maps to Vantiq data types.

## Maneuver Detection Events

| Astroshield Field | Vantiq Field | Data Type | Description |
|-------------------|--------------|-----------|-------------|
| `payload.catalogId` | `catalogId` | String | Space object identifier |
| `payload.deltaV` | `deltaV` | Real | Change in velocity (km/s) |
| `payload.confidence` | `confidence` | Real | Detection confidence (0-1) |
| `payload.maneuverType` | `maneuverType` | String | Type of maneuver detected |
| `payload.detectionTime` | `detectionTime` | DateTime | Time of detection |
| `header.source` | `source` | String | Source system identifier |
| `header.messageId` | `messageId` | String | Unique message identifier |
| `header.traceId` | `traceId` | String | Trace ID for event correlation |

## Observation Window Events

| Astroshield Field | Vantiq Field | Data Type | Description |
|-------------------|--------------|-----------|-------------|
| `payload.location` | `location` | Object | Observation location |
| `payload.qualityScore` | `qualityScore` | Real | Quality score (0-1) |
| `payload.qualityCategory` | `qualityCategory` | String | Quality category |
| `payload.recommendation` | `recommendation` | String | GO/NO-GO recommendation |
| `payload.observationWindow.start_time` | `observationWindow.startTime` | DateTime | Window start time |
| `payload.observationWindow.end_time` | `observationWindow.endTime` | DateTime | Window end time |
| `payload.observationWindow.duration_minutes` | `observationWindow.durationMinutes` | Integer | Duration in minutes |
| `payload.targetObject.catalog_id` | `targetObject.catalogId` | String | Target object identifier |
| `payload.targetObject.altitude_km` | `targetObject.altitudeKm` | Real | Object altitude in km |

## Data Transformations

For some fields, transformations are required:

### Time Format Conversion
Astroshield uses ISO-8601 strings, which need to be converted to Vantiq DateTime objects:

```javascript
// In Vantiq VAIL code
var vantiqDateTime = DateTime.parse(astroshieldIsoString)
```

### Nested Object Transformation
Vantiq prefers dot notation for deeply nested objects:

```javascript
// Transform from Astroshield format
var vantiqFormat = {
    "location.latitude": astroshieldEvent.payload.location.latitude,
    "location.longitude": astroshieldEvent.payload.location.longitude
}
``` 