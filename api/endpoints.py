from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
import random

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Create router without prefix - prefix will be added when mounting
router = APIRouter()

@app.get("/")
async def root():
    return {"message": "Welcome to AstroShield API"}

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
        "timestamp": datetime.now().isoformat()
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
            "protection_coverage": random.uniform(90, 100),
            "response_time": random.uniform(0.5, 2.0),
            "mitigation_success": random.uniform(85, 100)
        })
    
    return {
        "summary": {
            "total_conjunctions_analyzed": 1245,
            "threats_detected": 87,
            "threats_mitigated": 85,
            "average_response_time": 1.2,  # seconds
            "protection_coverage": 98.5  # percentage
        },
        "current_metrics": {
            "threat_analysis": {
                "value": 92.5,  # percentage of threats accurately analyzed
                "trend": "improving",
                "status": "nominal"
            },
            "collision_avoidance": {
                "value": 98.2,  # percentage of successful avoidance maneuvers
                "trend": "stable",
                "status": "nominal"
            },
            "debris_tracking": {
                "value": 95.8,  # percentage of debris objects tracked
                "trend": "stable",
                "status": "nominal"
            },
            "protection_status": {
                "value": 97.5,  # overall protection effectiveness
                "trend": "improving",
                "status": "nominal"
            }
        },
        "trends": {
            "daily": daily_trends,
            "weekly_summary": {
                "average_threat_level": 15.2,
                "total_conjunctions": 168,
                "mitigation_success_rate": 97.8,
                "average_response_time": 1.1
            },
            "monthly_summary": {
                "average_threat_level": 14.8,
                "total_conjunctions": 720,
                "mitigation_success_rate": 98.2,
                "average_response_time": 1.15
            }
        }
    }

@router.get("/maneuvers/data")
async def get_maneuvers_data():
    return {
        "maneuvers": [
            {
                "id": "MNV-001",
                "type": "hohmann",
                "status": "scheduled",
                "scheduledTime": (datetime.now() + timedelta(hours=2)).isoformat(),
                "details": {
                    "deltaV": 0.5,
                    "duration": 300,
                    "fuel_required": 0.8
                }
            },
            {
                "id": "MNV-002",
                "type": "stationkeeping",
                "status": "completed",
                "scheduledTime": (datetime.now() - timedelta(hours=2)).isoformat(),
                "completedTime": (datetime.now() - timedelta(hours=1)).isoformat(),
                "details": {
                    "deltaV": 0.2,
                    "duration": 180,
                    "fuel_used": 0.3
                }
            }
        ],
        "resources": {
            "fuel_remaining": 85.5,
            "thrust_capacity": 98.2,
            "next_maintenance": (datetime.now() + timedelta(days=7)).isoformat()
        },
        "lastUpdate": datetime.now().isoformat()
    }

@router.post("/maneuvers/plan")
async def plan_maneuver(maneuver: dict):
    # In a real implementation, this would validate and store the maneuver plan
    return {
        "id": f"MNV-{random.randint(100, 999)}",
        "type": maneuver["type"],
        "status": "scheduled",
        "scheduledTime": maneuver["executionTime"],
        "details": {
            "deltaV": maneuver["deltaV"],
            "duration": random.randint(100, 500),
            "fuel_required": maneuver["deltaV"] * 0.8
        }
    }

# Mount the router with /api prefix
app.include_router(router, prefix="/api")
