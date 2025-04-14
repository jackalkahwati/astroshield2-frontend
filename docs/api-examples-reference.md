# AstroShield API Examples Reference

This document provides example requests and responses for all major AstroShield API endpoints. Use these examples for integration testing, client development, and troubleshooting.

## Authentication

### Login

**Request:**
```bash
curl -X POST "https://api.astroshield.com/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "operator@example.com",
    "password": "secure_password_here"
  }'
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyLTEyMyIsImV4cCI6MTY5MzQ4NTYwMH0.abc123",
  "token_type": "bearer",
  "expires_at": "2023-06-15T15:30:45Z",
  "user": {
    "id": "user-123",
    "username": "operator@example.com",
    "roles": ["operator", "analyst"]
  }
}
```

### Refresh Token

**Request:**
```bash
curl -X POST "https://api.astroshield.com/auth/refresh" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyLTEyMyIsImV4cCI6MTY5MzQ4NTYwMH0.abc123"
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyLTEyMyIsImV4cCI6MTY5MzU3MjAwMH0.xyz456",
  "token_type": "bearer",
  "expires_at": "2023-06-15T18:30:45Z"
}
```

## Historical Analysis

### Get Historical Analysis

**Request:**
```bash
curl -X GET "https://api.astroshield.com/api/ccdm/historical-analysis?norad_id=25544&start_date=2023-06-01&end_date=2023-06-05&page=1&page_size=2" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyLTEyMyIsImV4cCI6MTY5MzQ4NTYwMH0.abc123"
```

**Response:**
```json
{
  "norad_id": 25544,
  "start_date": "2023-06-01",
  "end_date": "2023-06-05",
  "analysis_points": [
    {
      "timestamp": "2023-06-01T00:00:00Z",
      "threat_level": "NONE",
      "confidence": 0.92,
      "details": {
        "day": 0,
        "date": "2023-06-01T00:00:00Z",
        "metrics": {
          "position_uncertainty": 12.5,
          "velocity_uncertainty": 0.025,
          "signal_strength": -82.3,
          "maneuver_probability": 0.05
        }
      }
    },
    {
      "timestamp": "2023-06-02T00:00:00Z",
      "threat_level": "NONE",
      "confidence": 0.94,
      "details": {
        "day": 1,
        "date": "2023-06-02T00:00:00Z",
        "metrics": {
          "position_uncertainty": 11.8,
          "velocity_uncertainty": 0.022,
          "signal_strength": -81.7,
          "maneuver_probability": 0.04
        }
      }
    }
  ],
  "anomalies": [],
  "trend_summary": "Historical analysis shows stable behavior with no anomalies detected",
  "metadata": {
    "norad_id": 25544,
    "name": "ISS (ZARYA)",
    "international_designator": "1998-067A",
    "orbit_type": "LEO",
    "apogee": 424.3,
    "perigee": 418.2,
    "period_minutes": 92.8,
    "pagination": {
      "page": 1,
      "page_size": 2,
      "total_items": 5,
      "total_pages": 3
    }
  }
}
```

### Get Paginated Example (Page 2)

**Request:**
```bash
curl -X GET "https://api.astroshield.com/api/ccdm/historical-analysis?norad_id=25544&start_date=2023-06-01&end_date=2023-06-05&page=2&page_size=2" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyLTEyMyIsImV4cCI6MTY5MzQ4NTYwMH0.abc123"
```

**Response:**
```json
{
  "norad_id": 25544,
  "start_date": "2023-06-01",
  "end_date": "2023-06-05",
  "analysis_points": [
    {
      "timestamp": "2023-06-03T00:00:00Z",
      "threat_level": "NONE",
      "confidence": 0.91,
      "details": {
        "day": 2,
        "date": "2023-06-03T00:00:00Z",
        "metrics": {
          "position_uncertainty": 13.2,
          "velocity_uncertainty": 0.027,
          "signal_strength": -82.8,
          "maneuver_probability": 0.06
        }
      }
    },
    {
      "timestamp": "2023-06-04T00:00:00Z",
      "threat_level": "LOW",
      "confidence": 0.88,
      "details": {
        "day": 3,
        "date": "2023-06-04T00:00:00Z",
        "metrics": {
          "position_uncertainty": 18.5,
          "velocity_uncertainty": 0.035,
          "signal_strength": -83.4,
          "maneuver_probability": 0.22
        }
      }
    }
  ],
  "anomalies": [
    {
      "timestamp": "2023-06-04T12:45:12Z",
      "type": "ELEVATED_UNCERTAINTY",
      "severity": "LOW",
      "description": "Position uncertainty increased above nominal threshold"
    }
  ],
  "trend_summary": "Historical analysis shows generally stable behavior with minor fluctuations",
  "metadata": {
    "norad_id": 25544,
    "name": "ISS (ZARYA)",
    "international_designator": "1998-067A",
    "orbit_type": "LEO",
    "apogee": 424.3,
    "perigee": 418.2,
    "period_minutes": 92.8,
    "pagination": {
      "page": 2,
      "page_size": 2,
      "total_items": 5,
      "total_pages": 3
    }
  }
}
```

## Conjunction Management

### List Active Conjunctions

**Request:**
```bash
curl -X GET "https://api.astroshield.com/api/conjunctions?status=active&limit=2" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyLTEyMyIsImV4cCI6MTY5MzQ4NTYwMH0.abc123"
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
curl -X GET "https://api.astroshield.com/api/conjunctions/conj-2023-06-15-001" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyLTEyMyIsImV4cCI6MTY5MzQ4NTYwMH0.abc123"
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

## Maneuver Planning

### Plan Conjunction Avoidance Maneuver

**Request:**
```bash
curl -X POST "https://api.astroshield.com/api/maneuvers/plan" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyLTEyMyIsImV4cCI6MTY5MzQ4NTYwMH0.abc123" \
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
  "created_by": "operator@example.com"
}
```

### Get Maneuver Details

**Request:**
```bash
curl -X GET "https://api.astroshield.com/api/maneuvers/mnv-2023-06-15-001" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyLTEyMyIsImV4cCI6MTY5MzQ4NTYwMH0.abc123"
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
  "approval_status": "PENDING",
  "approval_needed_by": "2023-06-16T04:45:30Z",
  "created_at": "2023-06-15T09:40:18Z",
  "created_by": "operator@example.com",
  "updated_at": "2023-06-15T09:40:18Z"
}
```

### Approve Maneuver

**Request:**
```bash
curl -X POST "https://api.astroshield.com/api/maneuvers/mnv-2023-06-15-001/approve" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyLTEyMyIsImV4cCI6MTY5MzQ4NTYwMH0.abc123" \
  -H "Content-Type: application/json" \
  -d '{
    "approver_note": "Approved after review by Flight Director",
    "notification_emails": ["flight-team@example.com", "ground-control@example.com"]
  }'
```

**Response:**
```json
{
  "maneuver_id": "mnv-2023-06-15-001",
  "approval_status": "APPROVED",
  "approved_at": "2023-06-15T11:22:45Z",
  "approved_by": "operator@example.com",
  "approver_note": "Approved after review by Flight Director",
  "notifications_sent": true,
  "notification_recipients": ["flight-team@example.com", "ground-control@example.com"],
  "execution_scheduled": true,
  "execution_time": "2023-06-16T06:45:30Z"
}
```

## Space Object Catalog

### List Space Objects

**Request:**
```bash
curl -X GET "https://api.astroshield.com/api/objects?object_type=PAYLOAD&limit=2" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyLTEyMyIsImV4cCI6MTY5MzQ4NTYwMH0.abc123"
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
curl -X GET "https://api.astroshield.com/api/objects/25544" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyLTEyMyIsImV4cCI6MTY5MzQ4NTYwMH0.abc123"
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
    "type": "LEO",
    "apogee": 424.3,
    "perigee": 418.2,
    "inclination": 51.6,
    "period_minutes": 92.8,
    "eccentricity": 0.0004256,
    "semi_major_axis": 6783.5,
    "mean_anomaly": 347.2,
    "mean_motion": 15.49,
    "epoch": "2023-06-15T00:00:00Z"
  },
  "physical": {
    "mass": 420000,
    "dimensions": {
      "length": 73.0,
      "width": 108.5,
      "height": 20.1
    },
    "radar_cross_section": 428.5,
    "material": "MULTIPLE"
  },
  "operational_status": "ACTIVE",
  "maneuverability": "HIGH",
  "metadata": {
    "tracked_since": "1998-11-20",
    "last_updated": "2023-06-15T05:22:18Z",
    "data_source": "UDL",
    "tags": ["MANNED", "STATION"]
  }
}
```

## UDL Integration

### Get UDL Status

**Request:**
```bash
curl -X GET "https://api.astroshield.com/api/udl/status" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyLTEyMyIsImV4cCI6MTY5MzQ4NTYwMH0.abc123"
```

**Response:**
```json
{
  "status": "OPERATIONAL",
  "last_successful_sync": "2023-06-15T08:15:22Z",
  "last_sync_attempt": "2023-06-15T08:15:22Z",
  "sync_frequency_minutes": 15,
  "sync_statuses": {
    "catalog": "SUCCESS",
    "conjunctions": "SUCCESS",
    "maneuvers": "SUCCESS"
  },
  "metrics": {
    "objects_synced": 24876,
    "conjunctions_synced": 342,
    "maneuvers_synced": 18
  },
  "next_scheduled_sync": "2023-06-15T08:30:22Z"
}
```

### Manually Trigger UDL Sync

**Request:**
```bash
curl -X POST "https://api.astroshield.com/api/udl/sync" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyLTEyMyIsImV4cCI6MTY5MzQ4NTYwMH0.abc123" \
  -H "Content-Type: application/json" \
  -d '{
    "sync_types": ["catalog", "conjunctions"],
    "priority": "HIGH"
  }'
```

**Response:**
```json
{
  "sync_id": "sync-2023-06-15-123",
  "status": "QUEUED",
  "sync_types": ["catalog", "conjunctions"],
  "priority": "HIGH",
  "requested_at": "2023-06-15T09:45:12Z",
  "requested_by": "operator@example.com",
  "estimated_completion_time": "2023-06-15T09:50:12Z"
}
```

## System Health

### Get Health Status

**Request:**
```bash
curl -X GET "https://api.astroshield.com/api/health" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyLTEyMyIsImV4cCI6MTY5MzQ4NTYwMH0.abc123"
```

**Response:**
```json
{
  "status": "HEALTHY",
  "timestamp": "2023-06-15T10:12:34Z",
  "version": "1.5.2",
  "environment": "production",
  "uptime_seconds": 345600,
  "components": [
    {
      "name": "api",
      "status": "HEALTHY",
      "response_time_ms": 75
    },
    {
      "name": "database",
      "status": "HEALTHY",
      "details": {
        "connections": 12,
        "max_connections": 100,
        "disk_usage_percent": 42
      }
    },
    {
      "name": "redis",
      "status": "HEALTHY",
      "details": {
        "used_memory_mb": 124,
        "hit_rate": 0.92
      }
    },
    {
      "name": "udl_integration",
      "status": "HEALTHY",
      "details": {
        "last_successful_connection": "2023-06-15T10:10:12Z"
      }
    },
    {
      "name": "kafka",
      "status": "HEALTHY",
      "details": {
        "brokers_online": 3,
        "topics": 12
      }
    }
  ],
  "metrics": {
    "requests_per_minute": 122,
    "average_response_time_ms": 82,
    "error_rate": 0.002
  }
}
```

### Get Component Health

**Request:**
```bash
curl -X GET "https://api.astroshield.com/api/health/database" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyLTEyMyIsImV4cCI6MTY5MzQ4NTYwMH0.abc123"
```

**Response:**
```json
{
  "name": "database",
  "status": "HEALTHY",
  "timestamp": "2023-06-15T10:12:34Z",
  "details": {
    "connections": 12,
    "max_connections": 100,
    "connection_usage_percent": 12,
    "disk_usage_percent": 42,
    "read_iops": 245,
    "write_iops": 78,
    "query_stats": {
      "queries_per_second": 35,
      "slow_queries_last_hour": 2,
      "avg_query_time_ms": 12
    },
    "replication": {
      "status": "SYNCHRONIZED",
      "lag_seconds": 0.05
    }
  },
  "recent_issues": []
}
```

## Error Examples

### Authentication Error

**Request:**
```bash
curl -X POST "https://api.astroshield.com/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "wrong@example.com",
    "password": "incorrect_password"
  }'
```

**Response:**
```json
{
  "error": {
    "code": "AUTH001",
    "message": "Authentication failed",
    "details": "Invalid username or password",
    "timestamp": "2023-06-15T10:15:22Z",
    "request_id": "req-abc-123-xyz"
  }
}
```

### Resource Not Found

**Request:**
```bash
curl -X GET "https://api.astroshield.com/api/objects/99999" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyLTEyMyIsImV4cCI6MTY5MzQ4NTYwMH0.abc123"
```

**Response:**
```json
{
  "error": {
    "code": "DATA001",
    "message": "Object not found",
    "details": "No space object with NORAD ID 99999 exists in the database",
    "timestamp": "2023-06-15T10:18:42Z",
    "request_id": "req-def-456-uvw",
    "object_id": "99999"
  }
}
```

### Invalid Input

**Request:**
```bash
curl -X GET "https://api.astroshield.com/api/ccdm/historical-analysis?norad_id=25544&start_date=2023-06-15&end_date=2023-06-01" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyLTEyMyIsImV4cCI6MTY5MzQ4NTYwMH0.abc123"
```

**Response:**
```json
{
  "error": {
    "code": "DATA002",
    "message": "Invalid date range",
    "details": "End date (2023-06-01) must be after start date (2023-06-15)",
    "timestamp": "2023-06-15T10:20:15Z",
    "request_id": "req-ghi-789-rst",
    "validation_errors": [
      {
        "field": "end_date",
        "message": "End date must be after start date"
      }
    ]
  }
}
```

### Rate Limit Exceeded

**Request:**
```bash
curl -X GET "https://api.astroshield.com/api/conjunctions?status=active" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyLTEyMyIsImV4cCI6MTY5MzQ4NTYwMH0.abc123"
```

**Response:**
```json
{
  "error": {
    "code": "SYS001",
    "message": "Rate limit exceeded",
    "details": "You have exceeded the rate limit of 100 requests per minute for this endpoint",
    "timestamp": "2023-06-15T10:22:33Z",
    "request_id": "req-jkl-012-opq",
    "rate_limit": {
      "limit": 100,
      "interval": "minute",
      "retry_after_seconds": 15
    }
  }
}
``` 