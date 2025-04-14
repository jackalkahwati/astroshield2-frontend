# CCDM Service User Guide

## Overview

The Conjunction and Collision Data Management (CCDM) service provides advanced tools for monitoring, analyzing, and mitigating potential satellite conjunctions and collisions. This guide will help you understand how to interpret the data and use the service effectively.

## Historical Analysis

### Understanding Historical Analysis Results

The historical analysis endpoint (`/api/v1/ccdm/history/{object_id}`) provides a time-series view of threat levels and details for a specific space object. When interpreting these results, keep in mind:

#### Threat Levels

- **None**: No significant threat detected. Regular monitoring is sufficient.
- **Low**: Minor conjunction possibilities exist. Awareness is advised, but immediate action is typically not required.
- **Medium**: Significant conjunction possibilities detected. Closer monitoring and preparation for potential action is recommended.
- **High**: Serious conjunction threat detected. Action planning should begin, and heightened monitoring is necessary.

#### Data Point Structure

Each data point in the analysis contains:

```json
{
  "date": "2023-06-15T10:00:00",
  "threat_level": "medium",
  "details": {
    "closest_approach": 25000,   // meters
    "relative_velocity": 12500,  // m/s
    "collision_probability": 0.00025,
    "time_to_closest_approach": 86400,  // seconds
    "secondary_objects": [
      {"norad_id": "12345", "miss_distance": 26000},
      {"norad_id": "23456", "miss_distance": 35000}
    ],
    "recommended_actions": ["monitor", "prepare_maneuver"]
  }
}
```

#### Key Metrics Explained

- **Closest Approach**: Minimum distance (in meters) between your object and another space object. Values under 1000m typically fall into higher threat categories.
- **Relative Velocity**: Speed at which the objects approach each other in meters per second. Higher values can indicate more severe impact potential but less conjunction duration.
- **Collision Probability**: Statistical likelihood of collision. Typical thresholds:
  - Above 0.0001 (1 in 10,000): High concern
  - 0.00001 to 0.0001: Medium concern
  - Below 0.00001: Low concern
- **Time to Closest Approach**: Seconds until the closest approach occurs. This helps prioritize immediate versus future concerns.

#### Trend Analysis

The `trend_summary` field provides an interpretation of how threat levels are changing over time:

- **Improving**: Threat levels are generally decreasing, suggesting natural orbital dynamics are reducing conjunction risk.
- **Stable**: Threat levels remain consistent with no significant change.
- **Deteriorating**: Threat levels are increasing, suggesting closer monitoring or intervention may be necessary.
- **Fluctuating**: Inconsistent pattern that requires ongoing monitoring.

### Using Pagination

For large date ranges, the historical analysis results are paginated:

```json
{
  "pagination": {
    "page": 1,
    "page_size": 100,
    "total_items": 365,
    "total_pages": 4
  }
}
```

Use the `page` and `page_size` parameters to navigate through the results.

## Shape Change Detection

The shape change detection endpoint (`/api/v1/ccdm/detect_shape_changes`) helps identify when a space object's observable shape has changed, which could indicate deployment of solar panels, antennas, or other structural changes.

### Interpreting Shape Change Results

```json
{
  "object_id": "25544",
  "start_time": "2023-05-01T00:00:00",
  "end_time": "2023-06-01T00:00:00",
  "changes_detected": true,
  "confidence": 0.85,
  "change_points": [
    {
      "timestamp": "2023-05-15T14:30:00",
      "before_shape": "compact",
      "after_shape": "extended",
      "confidence": 0.92,
      "likely_cause": "solar_panel_deployment"
    }
  ]
}
```

- **changes_detected**: Boolean indicating if any changes were detected in the time period.
- **confidence**: Overall confidence in the analysis.
- **change_points**: Specific moments when shapes changed, including:
  - Before and after shape classifications
  - Confidence in this specific change
  - Likely cause based on known spacecraft behaviors

## Thermal Signature Analysis

The thermal signature endpoint (`/api/v1/ccdm/assess_thermal_signature`) provides information about the thermal characteristics of a space object.

### Interpreting Thermal Signature Results

```json
{
  "object_id": "25544",
  "timestamp": "2023-06-15T10:00:00",
  "signature_type": "variable",
  "temperature_range": {
    "min": -150,
    "max": 120
  },
  "anomalies_detected": false,
  "analysis": {
    "consistent_with_type": true,
    "suggests_activity": true,
    "activity_type": "normal_operations"
  }
}
```

- **signature_type**: Classification of the thermal pattern (stable, variable, cycling, etc.)
- **temperature_range**: Estimated temperature range in Celsius
- **anomalies_detected**: Whether unusual thermal patterns were detected
- **analysis**: Interpretation of the thermal data, including:
  - Whether the signature matches the expected pattern for this object type
  - Whether the signature suggests active systems
  - Classification of the activity type if detected

## Propulsive Capability Evaluation

The propulsive capability endpoint (`/api/v1/ccdm/evaluate_propulsive_capabilities`) assesses an object's ability to maneuver based on observed behavior.

### Interpreting Propulsion Results

```json
{
  "object_id": "25544",
  "analysis_period": 30,
  "capabilities_detected": true,
  "maneuver_types": ["station_keeping", "collision_avoidance"],
  "estimated_delta_v_capability": 120,
  "confidence": 0.89,
  "recent_maneuvers": [
    {
      "timestamp": "2023-06-10T08:15:00",
      "type": "station_keeping",
      "delta_v": 0.5,
      "confidence": 0.95
    }
  ]
}
```

- **capabilities_detected**: Whether propulsive capabilities were detected
- **maneuver_types**: Types of maneuvers the object appears capable of performing
- **estimated_delta_v_capability**: Estimated total delta-v capability in m/s
- **recent_maneuvers**: List of detected maneuvers including:
  - Type of maneuver
  - Delta-v used (in m/s)
  - Confidence in the detection

## CCDM Assessment

The comprehensive assessment endpoint (`/api/v1/ccdm/assessment/{object_id}`) provides an overall analysis of an object's behavior, capabilities, and risks.

### Interpreting Assessment Results

```json
{
  "object_id": "25544",
  "classification": {
    "type": "active_satellite",
    "confidence": 0.98,
    "capabilities": ["maneuver", "communication", "observation"]
  },
  "behavior_assessment": {
    "pattern": "predictable",
    "anomalies_detected": false,
    "last_maneuver": "2023-06-10T08:15:00"
  },
  "threat_assessment": {
    "current_level": "low",
    "collision_risk": 0.00005,
    "conjunction_events": 2
  },
  "recommendations": [
    "continue_regular_monitoring",
    "no_immediate_action_required"
  ]
}
```

The assessment combines multiple data sources to provide:

- **classification**: What type of object this appears to be and its capabilities
- **behavior_assessment**: Analysis of the object's orbital behavior patterns
- **threat_assessment**: Current risk levels and notable events
- **recommendations**: Suggested actions based on the assessment

## Anomaly Detection

The anomaly detection endpoint (`/api/v1/ccdm/anomalies/{object_id}`) identifies unusual behavior or characteristics that deviate from the object's historical patterns.

### Interpreting Anomaly Results

```json
{
  "object_id": "25544",
  "days": 30,
  "anomalies": [
    {
      "timestamp": "2023-06-12T15:30:00",
      "type": "unexpected_maneuver",
      "severity": "medium",
      "description": "Orbital adjustment outside normal maintenance pattern",
      "confidence": 0.85
    }
  ],
  "total_anomalies": 1,
  "analysis_period": {
    "start": "2023-05-13T00:00:00",
    "end": "2023-06-13T00:00:00"
  }
}
```

- **anomalies**: List of detected anomalies including:
  - Type of anomaly
  - Severity level
  - Description of what was detected
  - Confidence in the detection
- **total_anomalies**: Count of anomalies in the period

## Best Practices

1. **Regular Monitoring**: Check historical analysis at least daily for critical assets.
2. **Threat Level Escalation**: When threat levels increase from "none" to "low" or higher, increase monitoring frequency.
3. **Conjunction Events**: For any "medium" or higher threat, review the detailed conjunction data.
4. **Cross-Verification**: Use multiple endpoints to corroborate findings (e.g., verify a detected shape change against thermal signature changes).
5. **Time Windows**: When analyzing historical data:
   - Short-term (7 days): Identify immediate threats
   - Medium-term (30 days): Identify patterns and trends
   - Long-term (90+ days): Assess seasonal variations and long-period conjunctions

## Troubleshooting

### Common Issues

1. **Missing Data Points**: For periods with no data, check:
   - Object visibility during that period
   - Service outages (check `/api/v1/ccdm/status`)
   - Whether the object was actively tracked at that time

2. **Inconsistent Threat Levels**: If threat levels fluctuate rapidly:
   - Check for actual orbital changes
   - Verify data quality by comparing with other sources
   - Consider increasing the analysis window to see broader trends

3. **Performance Issues**: For slow responses:
   - Reduce date range requests to smaller chunks
   - Use pagination effectively
   - Schedule bulk analysis during off-peak hours

### Error Codes

| Code | Meaning | Action |
|------|---------|--------|
| INVALID_INPUT | Parameter validation failed | Check parameter formats and values |
| NOT_FOUND | Object ID not found | Verify the object ID exists in the catalog |
| PROCESSING_ERROR | Analysis engine error | Check inputs and try again, or contact support |
| TIMEOUT_ERROR | Request took too long | Reduce date range or try during off-peak hours |
| DATABASE_ERROR | Database access issue | Wait and retry, or contact support |

## Getting Support

For further assistance:
- Check the API documentation for detailed endpoint specifications
- Contact support with your object ID and request timestamp
- Join the user forum for community assistance

---

This documentation will be updated as new features and improvements are released to the CCDM service. 