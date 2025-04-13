from fastapi import FastAPI, Request
import uvicorn
from datetime import datetime, timedelta
import random
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="AstroShield API", 
              description="AstroShield Space Domain Awareness Platform API",
              version="1.0.0")

# Environment variables
PRODUCTION = os.environ.get("PRODUCTION", "false").lower() == "true"
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 3001))

# Configure CORS properly for production
if PRODUCTION:
    # In production, only allow the actual domain
    origins = [
        "https://astroshield.sdataplab.com",
        "http://astroshield.sdataplab.com",
    ]
else:
    # In development, allow all origins
    origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Error handling middleware
@app.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "status": "error"}
        )

@app.get("/")
def read_root():
    return {"message": "AstroShield API running", "environment": "production" if PRODUCTION else "development"}

@app.get("/status")
def status():
    return {
        "status": "operational",
        "version": "1.0.0",
        "environment": "production" if PRODUCTION else "development",
        "services": {
            "api": "online",
            "database": "simulated",
            "kafka": "simulated"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/health")
def api_v1_health():
    """API v1 endpoint for health status."""
    return status()

@app.get("/maneuvers")
def maneuvers():
    """Return a list of simulated satellite maneuvers."""
    # Generate some sample maneuvers data
    current_time = datetime.now()
    
    sample_maneuvers = [
        {
            "id": f"MNV-{random.randint(1000, 9999)}",
            "type": "Orbital Correction",
            "status": "Completed",
            "scheduledTime": (current_time - timedelta(days=2)).isoformat(),
            "completedTime": (current_time - timedelta(days=2, hours=-1)).isoformat(),
            "details": {
                "delta_v": round(random.uniform(0.5, 2.0), 2),
                "duration": round(random.uniform(300, 1200), 1),
                "fuel_required": round(random.uniform(0.5, 2.0), 2),
                "fuel_used": round(random.uniform(0.1, 0.5), 2)
            }
        },
        {
            "id": f"MNV-{random.randint(1000, 9999)}",
            "type": "Collision Avoidance",
            "status": "Scheduled",
            "scheduledTime": (current_time + timedelta(hours=5)).isoformat(),
            "details": {
                "delta_v": round(random.uniform(1.0, 3.0), 2),
                "duration": round(random.uniform(60, 300), 1),
                "fuel_required": round(random.uniform(0.3, 1.5), 2)
            }
        },
        {
            "id": f"MNV-{random.randint(1000, 9999)}",
            "type": "Station Keeping",
            "status": "In Progress",
            "scheduledTime": (current_time - timedelta(hours=1)).isoformat(),
            "details": {
                "delta_v": round(random.uniform(0.2, 1.0), 2),
                "duration": round(random.uniform(600, 1800), 1),
                "fuel_required": round(random.uniform(0.2, 1.0), 2),
                "fuel_used": round(random.uniform(0.05, 0.3), 2)
            }
        }
    ]
    
    return {"maneuvers": sample_maneuvers}

@app.get("/satellites")
def satellites():
    """Return a list of simulated satellites."""
    current_time = datetime.now()
    
    sample_satellites = [
        {
            "id": "SAT-001",
            "name": "AstroShield-1",
            "type": "Surveillance",
            "status": "Active",
            "orbit": "LEO",
            "altitude": 520,
            "inclination": 51.6,
            "period": 95,
            "launch_date": (current_time - timedelta(days=365)).isoformat(),
            "last_contact": (current_time - timedelta(minutes=15)).isoformat(),
            "health": 98
        },
        {
            "id": "SAT-002",
            "name": "AstroShield-2",
            "type": "Communication",
            "status": "Active",
            "orbit": "MEO",
            "altitude": 12000,
            "inclination": 45.2,
            "period": 302,
            "launch_date": (current_time - timedelta(days=180)).isoformat(),
            "last_contact": (current_time - timedelta(minutes=5)).isoformat(),
            "health": 100
        },
        {
            "id": "SAT-003",
            "name": "AstroShield-3",
            "type": "Imaging",
            "status": "Maintenance",
            "orbit": "SSO",
            "altitude": 650,
            "inclination": 97.8,
            "period": 98,
            "launch_date": (current_time - timedelta(days=90)).isoformat(),
            "last_contact": (current_time - timedelta(hours=2)).isoformat(),
            "health": 85
        }
    ]
    
    return {"satellites": sample_satellites}

@app.get("/api/v1/maneuvers")
def api_v1_maneuvers():
    """API v1 endpoint for maneuvers, returns the data directly as expected by the frontend."""
    data = maneuvers()
    return data["maneuvers"]

@app.get("/api/v1/satellites")
def api_v1_satellites():
    """API v1 endpoint for satellites, returns the same data."""
    return satellites()

@app.get("/api/v1/events")
def api_v1_events():
    """API v1 endpoint for events, returns simulated event data."""
    current_time = datetime.now()
    
    sample_events = [
        {
            "id": f"EVT-{random.randint(1000, 9999)}",
            "type": "Conjunction Warning",
            "severity": "High",
            "satellite": "SAT-001",
            "timestamp": (current_time - timedelta(hours=2)).isoformat(),
            "details": {
                "object": "Debris-45678",
                "time_to_conjunction": 24.5,
                "distance": 0.3,
                "probability": 0.015
            }
        },
        {
            "id": f"EVT-{random.randint(1000, 9999)}",
            "type": "Maneuver Completed",
            "severity": "Info",
            "satellite": "SAT-002",
            "timestamp": (current_time - timedelta(hours=12)).isoformat(),
            "details": {
                "maneuver_id": "MNV-4567",
                "delta_v_actual": 1.2,
                "fuel_consumed": 0.4
            }
        },
        {
            "id": f"EVT-{random.randint(1000, 9999)}",
            "type": "System Alert",
            "severity": "Medium",
            "satellite": "SAT-003",
            "timestamp": (current_time - timedelta(minutes=45)).isoformat(),
            "details": {
                "subsystem": "Power Control",
                "message": "Battery charge below optimal threshold",
                "current_value": 68,
                "threshold": 70
            }
        }
    ]
    
    return sample_events

if __name__ == "__main__":
    print(f"Starting AstroShield API in {'production' if PRODUCTION else 'development'} mode")
    print(f"Server running on {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT)