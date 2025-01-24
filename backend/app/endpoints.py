from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
import random
from typing import List, Optional
from pydantic import BaseModel

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

# Models
class OrbitData(BaseModel):
    altitude: float
    inclination: float
    period: float

class HealthData(BaseModel):
    power: int
    thermal: int
    communication: int

class TelemetryData(BaseModel):
    lastUpdate: str
    signalStrength: int
    temperature: float
    batteryLevel: int

class Satellite(BaseModel):
    id: str
    name: str
    status: str
    orbit: OrbitData
    health: HealthData
    telemetry: Optional[TelemetryData]

class ManeuverDetails(BaseModel):
    deltaV: float
    duration: int
    fuel_required: float
    rotation_angle: float
    fuel_used: Optional[float]

class Maneuver(BaseModel):
    id: str
    type: str
    status: str
    scheduledTime: str
    completedTime: Optional[str]
    details: ManeuverDetails

class CreateManeuverRequest(BaseModel):
    type: str
    details: dict

# Satellites
@router.get("/satellites", response_model=List[Satellite])
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

@router.get("/satellites/{satellite_id}", response_model=Satellite)
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

@router.get("/maneuvers", response_model=List[Maneuver])
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
    
    return maneuvers

@router.post("/maneuvers", response_model=Maneuver)
async def create_maneuver(data: CreateManeuverRequest):
    # Mock creating a new maneuver
    return {
        "id": f"MNV-{random.randint(1000, 9999)}",
        "type": data.type,
        "status": "scheduled",
        "scheduledTime": datetime.now().isoformat(),
        "completedTime": None,
        "details": {
            "deltaV": random.uniform(0.1, 2.0),
            "duration": random.randint(1800, 7200),
            "fuel_required": random.uniform(5, 20),
            "rotation_angle": random.uniform(0, 360),
            "fuel_used": None
        }
    }
