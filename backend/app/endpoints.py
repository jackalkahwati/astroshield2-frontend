from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
import random

router = APIRouter()

# Health Check
@router.get("/health")
async def get_health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "database": "connected",
            "api": "operational",
            "telemetry": "active"
        }
    }

# Satellites
@router.get("/satellites")
async def get_satellites():
    return [
        {
            "id": "SAT-001",
            "name": "AstroShield-1",
            "status": "operational",
            "orbit": {
                "altitude": 500,
                "inclination": 51.6,
                "period": 92.7
            },
            "health": {
                "power": 98,
                "thermal": 95,
                "communication": 99
            }
        },
        {
            "id": "SAT-002",
            "name": "AstroShield-2",
            "status": "operational",
            "orbit": {
                "altitude": 520,
                "inclination": 51.6,
                "period": 93.1
            },
            "health": {
                "power": 97,
                "thermal": 96,
                "communication": 98
            }
        }
    ]

@router.get("/satellites/{satellite_id}")
async def get_satellite_by_id(satellite_id: str):
    # Mock data for a single satellite
    return {
        "id": satellite_id,
        "name": f"AstroShield-{satellite_id[-1]}",
        "status": "operational",
        "orbit": {
            "altitude": 500 + random.randint(0, 50),
            "inclination": 51.6,
            "period": 92.7 + random.random()
        },
        "health": {
            "power": random.randint(95, 100),
            "thermal": random.randint(95, 100),
            "communication": random.randint(95, 100)
        },
        "telemetry": {
            "lastUpdate": datetime.now().isoformat(),
            "signalStrength": random.randint(85, 100),
            "temperature": 20 + random.random() * 5,
            "batteryLevel": random.randint(85, 100)
        }
    }

@router.get("/telemetry/{satellite_id}")
async def get_telemetry_data(satellite_id: str):
    return {
        "satellite_id": satellite_id,
        "timestamp": datetime.now().isoformat(),
        "data": {
            "power": {
                "battery_level": random.randint(85, 100),
                "solar_panel_output": random.uniform(90, 100),
                "power_consumption": random.uniform(50, 80)
            },
            "thermal": {
                "internal_temp": 20 + random.random() * 5,
                "external_temp": -10 + random.random() * 5,
                "heating_power": random.uniform(20, 40)
            },
            "communication": {
                "signal_strength": random.randint(85, 100),
                "bit_error_rate": random.uniform(0, 0.001),
                "latency": random.uniform(0.1, 0.5)
            }
        }
    }

@router.get("/indicators")
async def get_indicators():
    return {
        "timestamp": datetime.now().isoformat(),
        "indicators": {
            "orbit_stability": {
                "value": random.uniform(95, 100),
                "trend": "stable",
                "alerts": []
            },
            "power_management": {
                "value": random.uniform(90, 100),
                "trend": "improving",
                "alerts": []
            },
            "thermal_control": {
                "value": random.uniform(92, 98),
                "trend": "stable",
                "alerts": []
            },
            "communication_quality": {
                "value": random.uniform(95, 100),
                "trend": "stable",
                "alerts": []
            }
        }
    }

@router.get("/stability/{satellite_id}")
async def get_stability_analysis(satellite_id: str):
    return {
        "satellite_id": satellite_id,
        "timestamp": datetime.now().isoformat(),
        "analysis": {
            "attitude_stability": {
                "value": random.uniform(95, 100),
                "confidence": random.uniform(90, 100),
                "trend": "stable"
            },
            "orbit_stability": {
                "value": random.uniform(95, 100),
                "confidence": random.uniform(90, 100),
                "trend": "stable"
            },
            "thermal_stability": {
                "value": random.uniform(92, 98),
                "confidence": random.uniform(90, 100),
                "trend": "stable"
            }
        },
        "recommendations": []
    }

@router.get("/comprehensive/data")
async def get_comprehensive_data():
    return {
        "metrics": {
            "attitude_stability": 95.5,
            "orbit_stability": 98.2,
            "thermal_stability": 87.3,
            "power_stability": 92.8,
            "communication_stability": 96.1
        },
        "status": "nominal",
        "alerts": [],
        "timestamp": datetime.now().isoformat()
    }

@router.get("/stability/metrics")
async def get_stability_metrics():
    return {
        "metrics": {
            "attitude_stability": 95.5,
            "orbit_stability": 98.2,
            "thermal_stability": 87.3,
            "power_stability": 92.8,
            "communication_stability": 96.1
        },
        "status": "nominal",
        "timestamp": "2024-01-21T12:00:00Z"
    }

@router.get("/analytics/data")
async def get_analytics_data():
    # Generate mock analytics data
    current_time = datetime.now()
    
    # Generate daily trends data
    daily_trends = []
    for i in range(24):  # Last 24 hours
        timestamp = current_time - timedelta(hours=23-i)
        daily_trends.append({
            "timestamp": timestamp.isoformat(),
            "conjunction_count": random.randint(0, 5),
            "threat_level": random.uniform(0, 100),
            "protection_coverage": random.uniform(85, 100),
            "response_time": random.uniform(0.5, 3.0),
            "mitigation_success": random.uniform(80, 100)
        })
    
    return {
        "summary": {
            "total_conjunctions_analyzed": random.randint(100, 500),
            "threats_detected": random.randint(10, 50),
            "threats_mitigated": random.randint(8, 45),
            "average_response_time": random.uniform(1.0, 2.5),
            "protection_coverage": random.uniform(90, 99)
        },
        "current_metrics": {
            "threat_analysis": {
                "value": random.uniform(90, 99),
                "trend": "stable",
                "status": "nominal"
            },
            "collision_avoidance": {
                "value": random.uniform(90, 99),
                "trend": "improving",
                "status": "nominal"
            },
            "debris_tracking": {
                "value": random.uniform(90, 99),
                "trend": "stable",
                "status": "nominal"
            },
            "protection_status": {
                "value": random.uniform(90, 99),
                "trend": "stable",
                "status": "nominal"
            }
        },
        "trends": {
            "daily": daily_trends,
            "weekly_summary": {
                "average_threat_level": random.uniform(10, 30),
                "total_conjunctions": random.randint(500, 1000),
                "mitigation_success_rate": random.uniform(90, 99),
                "average_response_time": random.uniform(1.0, 2.0)
            },
            "monthly_summary": {
                "average_threat_level": random.uniform(10, 30),
                "total_conjunctions": random.randint(2000, 4000),
                "mitigation_success_rate": random.uniform(90, 99),
                "average_response_time": random.uniform(1.0, 2.0)
            }
        }
    }

@router.get("/maneuvers")
async def get_maneuvers():
    current_time = datetime.now()
    
    # Generate mock maneuvers data
    maneuvers = []
    for i in range(5):  # Last 5 maneuvers
        scheduled_time = current_time - timedelta(hours=random.randint(1, 24))
        completed_time = scheduled_time + timedelta(minutes=random.randint(30, 120)) if random.random() > 0.3 else None
        
        maneuvers.append({
            "id": f"MNV-{random.randint(1000, 9999)}",
            "type": random.choice(["hohmann", "stationkeeping", "phasing", "collision"]),
            "status": random.choice(["scheduled", "completed", "failed", "executing"]),
            "scheduledTime": scheduled_time.isoformat(),
            "completedTime": completed_time.isoformat() if completed_time else None,
            "details": {
                "deltaV": random.uniform(0.1, 2.0),
                "duration": random.randint(1800, 7200),  # Duration in seconds
                "fuel_required": random.uniform(5, 20),
                "rotation_angle": random.uniform(0, 360),
                "fuel_used": random.uniform(4, 18) if completed_time else None
            }
        })
    
    return {
        "maneuvers": maneuvers,
        "resources": {
            "fuel_remaining": random.uniform(50, 100),
            "thrust_capacity": random.uniform(80, 100),
            "next_maintenance": (current_time + timedelta(days=random.randint(10, 30))).isoformat()
        },
        "lastUpdate": current_time.isoformat()
    }

@router.post("/maneuvers")
async def create_maneuver(data: dict):
    # Mock creating a new maneuver
    return {
        "id": f"MNV-{random.randint(1000, 9999)}",
        "type": data.get("type", "unknown"),
        "status": "scheduled",
        "scheduledTime": datetime.now().isoformat(),
        "details": data.get("details", {}),
        "created": datetime.now().isoformat()
    }
