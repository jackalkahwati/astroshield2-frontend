# CCDM API Examples

This document provides example requests and responses for each CCDM (Conjunction and Collision Detection and Mitigation) API endpoint.

## Authentication

### Login

**Request:**
```bash
curl -X POST "https://api.astroshield.com/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "operator@astroshield.com",
    "password": "your_secure_password"
  }'
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_at": "2023-06-15T15:30:45Z"
}
```

## Conjunction Data

### List Conjunctions

**Request:**
```bash
curl -X GET "https://api.astroshield.com/conjunctions?status=active&limit=2" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

**Response:**
```json
{
  "conjunctions": [
    {
      "id": "conj-2023-06-15-001",
      "time_of_closest_approach": "2023-06-16T08:45:30Z",
      "primary_object": {
        "norad_id": 25544,
        "name": "ISS (ZARYA)",
        "object_type": "PAYLOAD"
      },
      "secondary_object": {
        "norad_id": 47423,
        "name": "COSMOS 2251 DEB",
        "object_type": "DEBRIS"
      },
      "miss_distance": 0.873,
      "probability_of_collision": 0.00021,
      "status": "ACTIVE",
      "created_at": "2023-06-15T03:22:18Z"
    },
    {
      "id": "conj-2023-06-15-002",
      "time_of_closest_approach": "2023-06-16T14:12:05Z",
      "primary_object": {
        "norad_id": 43013,
        "name": "STARLINK-1234",
        "object_type": "PAYLOAD"
      },
      "secondary_object": {
        "norad_id": 38345,
        "name": "COSMOS 2251 DEB",
        "object_type": "DEBRIS"
      },
      "miss_distance": 1.245,
      "probability_of_collision": 0.00008,
      "status": "ACTIVE",
      "created_at": "2023-06-15T05:18:42Z"
    }
  ],
  "pagination": {
    "total": 24,
    "limit": 2,
    "offset": 0,
    "next_offset": 2
  }
}
```

### Get Conjunction Details

**Request:**
```bash
curl -X GET "https://api.astroshield.com/conjunctions/conj-2023-06-15-001" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

**Response:**
```json
{
  "id": "conj-2023-06-15-001",
  "time_of_closest_approach": "2023-06-16T08:45:30Z",
  "primary_object": {
    "norad_id": 25544,
    "name": "ISS (ZARYA)",
    "object_type": "PAYLOAD",
    "dimensions": {
      "length": 73.0,
      "width": 108.5,
      "height": 20.1
    },
    "mass": 420000,
    "orbit": {
      "apogee": 424.3,
      "perigee": 418.2,
      "inclination": 51.6,
      "period_minutes": 92.8
    }
  },
  "secondary_object": {
    "norad_id": 47423,
    "name": "COSMOS 2251 DEB",
    "object_type": "DEBRIS",
    "dimensions": {
      "length": 0.1,
      "width": 0.1,
      "height": 0.1
    },
    "mass": 0.05,
    "orbit": {
      "apogee": 815.2,
      "perigee": 785.5,
      "inclination": 71.2,
      "period_minutes": 100.4
    }
  },
  "miss_distance": 0.873,
  "probability_of_collision": 0.00021,
  "relative_velocity": 14.23,
  "status": "ACTIVE",
  "recommended_actions": [
    {
      "type": "MONITOR",
      "description": "Continue to monitor the conjunction event",
      "urgency": "LOW"
    }
  ],
  "historical_assessments": [
    {
      "timestamp": "2023-06-15T03:22:18Z",
      "miss_distance": 0.912,
      "probability_of_collision": 0.00019
    },
    {
      "timestamp": "2023-06-15T08:22:18Z",
      "miss_distance": 0.873,
      "probability_of_collision": 0.00021
    }
  ],
  "created_at": "2023-06-15T03:22:18Z",
  "updated_at": "2023-06-15T08:22:18Z"
}
```

## Space Objects

### List Objects

**Request:**
```bash
curl -X GET "https://api.astroshield.com/objects?object_type=PAYLOAD&limit=2" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

**Response:**
```json
{
  "objects": [
    {
      "norad_id": 25544,
      "name": "ISS (ZARYA)",
      "international_designator": "1998-067A",
      "object_type": "PAYLOAD",
      "owner": "ISS",
      "launch_date": "1998-11-20",
      "orbit_type": "LEO",
      "operational_status": "ACTIVE"
    },
    {
      "norad_id": 43013,
      "name": "STARLINK-1234",
      "international_designator": "2018-020A",
      "object_type": "PAYLOAD",
      "owner": "SPACEX",
      "launch_date": "2018-02-22",
      "orbit_type": "LEO",
      "operational_status": "ACTIVE"
    }
  ],
  "pagination": {
    "total": 7823,
    "limit": 2,
    "offset": 0,
    "next_offset": 2
  }
}
```

### Get Object Details

**Request:**
```bash
curl -X GET "https://api.astroshield.com/objects/25544" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

**Response:**
```json
{
  "norad_id": 25544,
  "name": "ISS (ZARYA)",
  "international_designator": "1998-067A",
  "object_type": "PAYLOAD",
  "owner": "ISS",
  "launch_date": "1998-11-20",
  "orbit": {
    "apogee": 424.3,
    "perigee": 418.2,
    "inclination": 51.6,
    "period_minutes": 92.8,
    "epoch": "2023-06-15T00:00:00Z"
  },
  "physical_properties": {
    "dimensions": {
      "length": 73.0,
      "width": 108.5,
      "height": 20.1
    },
    "mass": 420000,
    "cross_sectional_area": 2500
  },
  "operational_status": "ACTIVE",
  "maneuver_capability": true,
  "current_tle": {
    "line1": "1 25544U 98067A   23166.50164963  .00010403  00000+0  18759-3 0  9996",
    "line2": "2 25544  51.6423 339.8465 0005924  88.7504 354.9268 15.50582211350375"
  },
  "active_conjunctions": 3,
  "recent_maneuvers": [
    {
      "id": "mnv-2023-06-10-001",
      "execution_time": "2023-06-10T12:34:22Z",
      "delta_v": 0.21,
      "reason": "STATION_KEEPING"
    }
  ],
  "created_at": "2023-06-01T00:00:00Z",
  "updated_at": "2023-06-15T08:22:18Z"
}
```

## CCDM Analysis

### Analyze Object

**Request:**
```bash
curl -X POST "https://api.astroshield.com/ccdm/analyze" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..." \
  -H "Content-Type: application/json" \
  -d '{
    "norad_id": 25544,
    "analysis_options": ["SHAPE_CHANGE", "THERMAL", "PROPULSIVE"]
  }'
```

**Response:**
```json
{
  "norad_id": 25544,
  "timestamp": "2023-06-15T09:32:45Z",
  "analysis_results": [
    {
      "timestamp": "2023-06-15T09:32:45Z",
      "confidence": 0.87,
      "threat_level": "NONE",
      "details": {
        "component": "subsystem-1",
        "anomaly_score": 0.12
      }
    },
    {
      "timestamp": "2023-06-15T09:31:45Z",
      "confidence": 0.89,
      "threat_level": "NONE",
      "details": {
        "component": "subsystem-2",
        "anomaly_score": 0.08
      }
    },
    {
      "timestamp": "2023-06-15T09:30:45Z",
      "confidence": 0.92,
      "threat_level": "NONE",
      "details": {
        "component": "subsystem-3",
        "anomaly_score": 0.05
      }
    }
  ],
  "summary": "Analysis completed for object 25544",
  "metadata": {
    "norad_id": 25544,
    "name": "ISS (ZARYA)",
    "international_designator": "1998-067A",
    "orbit_type": "LEO",
    "apogee": 424.3,
    "perigee": 418.2,
    "period_minutes": 92.8,
    "launch_date": "1998-11-20",
    "country": "ISS",
    "status": "ACTIVE"
  }
}
```

### Assess Threat

**Request:**
```bash
curl -X POST "https://api.astroshield.com/ccdm/assess-threat" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..." \
  -H "Content-Type: application/json" \
  -d '{
    "norad_id": 43013,
    "assessment_factors": ["COLLISION", "MANEUVER", "DEBRIS"]
  }'
```

**Response:**
```json
{
  "norad_id": 43013,
  "timestamp": "2023-06-15T09:35:22Z",
  "overall_threat": "LOW",
  "confidence": 0.82,
  "threat_components": {
    "collision": "LOW",
    "maneuver": "NONE",
    "debris": "NONE"
  },
  "recommendations": [
    "Monitor the object regularly",
    "Verify telemetry data with secondary sources",
    "Update trajectory predictions"
  ],
  "metadata": {
    "norad_id": 43013,
    "name": "STARLINK-1234",
    "international_designator": "2018-020A",
    "orbit_type": "LEO",
    "apogee": 550.2,
    "perigee": 545.8,
    "period_minutes": 95.6,
    "launch_date": "2018-02-22",
    "country": "USA",
    "status": "ACTIVE"
  }
}
```

### Get Historical Analysis

**Request:**
```bash
curl -X GET "https://api.astroshield.com/ccdm/historical-analysis?norad_id=25544&start_date=2023-06-10&end_date=2023-06-12&page=1&page_size=2" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

**Response:**
```json
{
  "norad_id": 25544,
  "start_date": "2023-06-10",
  "end_date": "2023-06-12",
  "analysis_points": [
    {
      "timestamp": "2023-06-10T00:00:00Z",
      "threat_level": "NONE",
      "confidence": 0.92,
      "details": {
        "day": 0,
        "date": "2023-06-10T00:00:00Z",
        "metrics": {
          "position_uncertainty": 12.5,
          "velocity_uncertainty": 0.025,
          "signal_strength": -82.3,
          "maneuver_probability": 0.05
        }
      }
    },
    {
      "timestamp": "2023-06-11T00:00:00Z",
      "threat_level": "NONE",
      "confidence": 0.91,
      "details": {
        "day": 1,
        "date": "2023-06-11T00:00:00Z",
        "metrics": {
          "position_uncertainty": 13.2,
          "velocity_uncertainty": 0.028,
          "signal_strength": -83.1,
          "maneuver_probability": 0.04
        }
      }
    }
  ],
  "trend_summary": "Historical analysis for 3 days shows normal behavior",
  "metadata": {
    "norad_id": 25544,
    "name": "ISS (ZARYA)",
    "international_designator": "1998-067A",
    "orbit_type": "LEO",
    "apogee": 424.3,
    "perigee": 418.2,
    "period_minutes": 92.8,
    "launch_date": "1998-11-20",
    "country": "ISS",
    "status": "ACTIVE",
    "pagination": {
      "page": 1,
      "page_size": 2,
      "total_items": 3,
      "total_pages": 2
    }
  }
}
```

### Detect Shape Changes

**Request:**
```bash
curl -X POST "https://api.astroshield.com/ccdm/detect-shape-changes" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..." \
  -H "Content-Type: application/json" \
  -d '{
    "norad_id": 43013,
    "start_date": "2023-05-01",
    "end_date": "2023-06-15"
  }'
```

**Response:**
```json
{
  "norad_id": 43013,
  "detected_changes": [
    {
      "timestamp": "2023-05-15T08:23:45Z",
      "description": "Detected change in solar panel configuration",
      "confidence": 0.78,
      "before_shape": "standard_configuration",
      "after_shape": "modified_configuration",
      "significance": 0.35
    },
    {
      "timestamp": "2023-06-02T14:12:33Z",
      "description": "Detected change in antenna configuration",
      "confidence": 0.82,
      "before_shape": "standard_configuration",
      "after_shape": "modified_configuration",
      "significance": 0.42
    }
  ],
  "summary": "Detected 2 shape changes with average significance of 0.39",
  "metadata": {
    "norad_id": 43013,
    "name": "STARLINK-1234",
    "international_designator": "2018-020A",
    "orbit_type": "LEO",
    "apogee": 550.2,
    "perigee": 545.8,
    "period_minutes": 95.6,
    "launch_date": "2018-02-22",
    "country": "USA",
    "status": "ACTIVE"
  }
}
```

## Maneuver Planning

### Plan Maneuver

**Request:**
```bash
curl -X POST "https://api.astroshield.com/maneuvers/plan" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..." \
  -H "Content-Type: application/json" \
  -d '{
    "conjunction_id": "conj-2023-06-15-001",
    "execution_time": "2023-06-16T06:45:30Z",
    "strategy": "MAXIMIZE_MISS_DISTANCE",
    "constraints": {
      "max_delta_v": 0.5,
      "min_post_maneuver_poc_reduction": 0.8
    }
  }'
```

**Response:**
```json
{
  "maneuver_id": "mnv-2023-06-15-001",
  "conjunction_id": "conj-2023-06-15-001",
  "spacecraft_norad_id": 25544,
  "execution_time": "2023-06-16T06:45:30Z",
  "burn_duration_seconds": 12.3,
  "delta_v": 0.32,
  "delta_v_vector": {
    "x": 0.18,
    "y": 0.23,
    "z": -0.12
  },
  "pre_maneuver_metrics": {
    "miss_distance": 0.873,
    "probability_of_collision": 0.00021
  },
  "post_maneuver_metrics": {
    "miss_distance": 15.42,
    "probability_of_collision": 0.000002
  },
  "fuel_use_kg": 1.28,
  "post_maneuver_orbit": {
    "apogee": 424.8,
    "perigee": 418.2,
    "inclination": 51.6,
    "period_minutes": 92.9
  },
  "status": "PLANNED",
  "alternatives": [
    {
      "strategy": "MINIMIZE_FUEL",
      "delta_v": 0.22,
      "miss_distance": 9.38,
      "probability_of_collision": 0.000007,
      "fuel_use_kg": 0.84
    }
  ],
  "created_at": "2023-06-15T09:40:18Z",
  "created_by": "operator@astroshield.com"
}
```

## Analytics

### Get Statistics

**Request:**
```bash
curl -X GET "https://api.astroshield.com/analytics/conjunction-stats?start_date=2023-06-01&end_date=2023-06-15" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

**Response:**
```json
{
  "time_period": {
    "start_date": "2023-06-01",
    "end_date": "2023-06-15"
  },
  "total_conjunctions": 287,
  "by_status": {
    "ACTIVE": 42,
    "MITIGATED": 18,
    "RESOLVED": 227
  },
  "by_object_type": {
    "PAYLOAD_VS_DEBRIS": 186,
    "PAYLOAD_VS_PAYLOAD": 52,
    "DEBRIS_VS_DEBRIS": 49
  },
  "risk_distribution": {
    "HIGH": 8,
    "MEDIUM": 29,
    "LOW": 250
  },
  "miss_distance_stats": {
    "minimum": 0.128,
    "maximum": 24.82,
    "average": 5.67,
    "median": 4.28
  },
  "maneuvers_performed": 18,
  "total_delta_v_used": 4.83,
  "trend": [
    {
      "date": "2023-06-01",
      "new_conjunctions": 22,
      "resolved_conjunctions": 19
    },
    {
      "date": "2023-06-02",
      "new_conjunctions": 19,
      "resolved_conjunctions": 17
    }
  ]
}
```

## Notifications

### Get Notification Settings

**Request:**
```bash
curl -X GET "https://api.astroshield.com/notifications/settings" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

**Response:**
```json
{
  "user_id": "user-123",
  "email_notifications": {
    "enabled": true,
    "addresses": ["operator@astroshield.com", "team@astroshield.com"],
    "frequency": "IMMEDIATE"
  },
  "sms_notifications": {
    "enabled": true,
    "phone_numbers": ["+12025551234"],
    "frequency": "HIGH_PRIORITY_ONLY"
  },
  "webhook_notifications": {
    "enabled": false,
    "urls": [],
    "secret": null
  },
  "notification_rules": [
    {
      "id": "rule-001",
      "name": "High Risk Conjunctions",
      "condition": "probability_of_collision > 0.0001",
      "channels": ["EMAIL", "SMS"],
      "priority": "HIGH"
    },
    {
      "id": "rule-002",
      "name": "New Conjunctions",
      "condition": "status == 'ACTIVE'",
      "channels": ["EMAIL"],
      "priority": "MEDIUM"
    }
  ],
  "updated_at": "2023-06-01T14:22:30Z"
}
```

### Update Notification Settings

**Request:**
```bash
curl -X PUT "https://api.astroshield.com/notifications/settings" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..." \
  -H "Content-Type: application/json" \
  -d '{
    "email_notifications": {
      "enabled": true,
      "addresses": ["operator@astroshield.com", "team@astroshield.com", "manager@astroshield.com"],
      "frequency": "IMMEDIATE"
    },
    "sms_notifications": {
      "enabled": true,
      "phone_numbers": ["+12025551234"],
      "frequency": "HIGH_PRIORITY_ONLY"
    },
    "webhook_notifications": {
      "enabled": true,
      "urls": ["https://hooks.astroshield.com/conjunction-alerts"],
      "secret": "wh_sec_abcdefghijklmnopqrstuvwxyz"
    },
    "notification_rules": [
      {
        "id": "rule-001",
        "name": "High Risk Conjunctions",
        "condition": "probability_of_collision > 0.00005",
        "channels": ["EMAIL", "SMS", "WEBHOOK"],
        "priority": "HIGH"
      },
      {
        "id": "rule-002",
        "name": "New Conjunctions",
        "condition": "status == 'ACTIVE'",
        "channels": ["EMAIL"],
        "priority": "MEDIUM"
      },
      {
        "id": "rule-003",
        "name": "Resolved Conjunctions",
        "condition": "status_changed && status == 'RESOLVED'",
        "channels": ["EMAIL"],
        "priority": "LOW"
      }
    ]
  }'
```

**Response:**
```json
{
  "user_id": "user-123",
  "email_notifications": {
    "enabled": true,
    "addresses": ["operator@astroshield.com", "team@astroshield.com", "manager@astroshield.com"],
    "frequency": "IMMEDIATE"
  },
  "sms_notifications": {
    "enabled": true,
    "phone_numbers": ["+12025551234"],
    "frequency": "HIGH_PRIORITY_ONLY"
  },
  "webhook_notifications": {
    "enabled": true,
    "urls": ["https://hooks.astroshield.com/conjunction-alerts"],
    "secret": "wh_sec_abcdefghijklmnopqrstuvwxyz"
  },
  "notification_rules": [
    {
      "id": "rule-001",
      "name": "High Risk Conjunctions",
      "condition": "probability_of_collision > 0.00005",
      "channels": ["EMAIL", "SMS", "WEBHOOK"],
      "priority": "HIGH"
    },
    {
      "id": "rule-002",
      "name": "New Conjunctions",
      "condition": "status == 'ACTIVE'",
      "channels": ["EMAIL"],
      "priority": "MEDIUM"
    },
    {
      "id": "rule-003",
      "name": "Resolved Conjunctions",
      "condition": "status_changed && status == 'RESOLVED'",
      "channels": ["EMAIL"],
      "priority": "LOW"
    }
  ],
  "updated_at": "2023-06-15T10:15:22Z"
}
```

## Error Handling

### 400 Bad Request

**Request with Invalid Parameters:**
```bash
curl -X GET "https://api.astroshield.com/conjunctions?status=invalid_status" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

**Response:**
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid 'status' parameter. Allowed values are 'active', 'resolved', 'mitigated', or 'all'.",
    "details": {
      "field": "status",
      "value": "invalid_status",
      "allowed_values": ["active", "resolved", "mitigated", "all"]
    },
    "timestamp": "2023-06-15T10:18:30Z",
    "request_id": "req_abc123"
  }
}
```

### 401 Unauthorized

**Request with Invalid Token:**
```bash
curl -X GET "https://api.astroshield.com/conjunctions" \
  -H "Authorization: Bearer invalid_token"
```

**Response:**
```json
{
  "error": {
    "code": "UNAUTHORIZED",
    "message": "Invalid or expired authentication token",
    "timestamp": "2023-06-15T10:20:45Z",
    "request_id": "req_def456"
  }
}
```

### 404 Not Found

**Request for Non-Existent Resource:**
```bash
curl -X GET "https://api.astroshield.com/conjunctions/non-existent-id" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

**Response:**
```json
{
  "error": {
    "code": "RESOURCE_NOT_FOUND",
    "message": "Conjunction with ID 'non-existent-id' not found",
    "timestamp": "2023-06-15T10:22:15Z",
    "request_id": "req_ghi789"
  }
}
```

### 429 Too Many Requests

**Response when Rate Limited:**
```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Please try again later.",
    "details": {
      "rate_limit": 100,
      "retry_after": 45
    },
    "timestamp": "2023-06-15T10:25:30Z",
    "request_id": "req_jkl012"
  }
}
```

### 500 Internal Server Error

**Response on Server Error:**
```json
{
  "error": {
    "code": "SERVER_ERROR",
    "message": "An unexpected error occurred while processing your request",
    "timestamp": "2023-06-15T10:28:45Z",
    "request_id": "req_mno345"
  }
}
``` 