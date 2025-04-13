from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from typing import List
from datetime import datetime

# Create the FastAPI app
app = FastAPI(
    title="AstroShield API",
    description="Backend API for the AstroShield satellite protection system",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add basic health check endpoint
@app.get("/api/v1/health")
def health_check():
    return {"status": "healthy", "version": "1.0.0"}

# Add a satellites endpoint with example data
@app.get("/api/v1/satellites")
def get_satellites():
    return [
        {
            "id": "sat-001",
            "name": "Starlink-1234",
            "type": "Communication",
            "orbit": "LEO",
            "status": "Active"
        },
        {
            "id": "sat-002",
            "name": "ISS",
            "type": "Space Station",
            "orbit": "LEO",
            "status": "Active"
        },
        {
            "id": "sat-003",
            "name": "GPS-IIF-10",
            "type": "Navigation",
            "orbit": "MEO",
            "status": "Active"
        }
    ]

# Add a simple events endpoint
@app.get("/api/v1/events")
def get_events():
    return [
        {
            "id": "evt-001",
            "type": "Proximity",
            "severity": "High",
            "timestamp": "2025-04-11T07:30:00Z",
            "description": "Close approach detected"
        },
        {
            "id": "evt-002",
            "type": "Maneuver",
            "severity": "Medium",
            "timestamp": "2025-04-10T14:45:00Z",
            "description": "Orbital adjustment"
        }
    ]

# Add a mock UDL status endpoint
@app.get("/api/v1/mock-udl-status")
def mock_udl_status():
    return {
        "status": "connected",
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": ["state-vectors", "elsets", "sensor-data", "conjunction-data"],
        "active_sensors": 7,
        "last_data_time": datetime.utcnow().isoformat()
    }

# Run the app
if __name__ == "__main__":
    port = int(os.getenv("PORT", 3001))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True) 