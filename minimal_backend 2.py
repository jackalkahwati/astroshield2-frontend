#!/usr/bin/env python3
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
from datetime import datetime, timedelta

# Configure logging directly, without importing from src
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="AstroShield API",
    description="Backend API for the AstroShield satellite protection system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/api/v1/health")
async def health_check():
    """
    Health check endpoint to verify the API is running
    """
    return {
        "status": "healthy",
        "uptime": "1d 5h 22m",
        "memory": {
            "used": 512,
            "total": 2048,
            "percent": 25.0
        },
        "cpu": {
            "usage": 12.5
        },
        "version": "1.0.0"
    }

# Maneuvers endpoint (mocked data)
@app.get("/api/v1/maneuvers")
async def get_maneuvers():
    """
    Get a list of maneuvers (mocked)
    """
    now = datetime.now()
    maneuvers = [
        {
            "id": "mnv-001",
            "satellite_id": "sat-001",
            "status": "completed",
            "type": "collision_avoidance",
            "scheduledTime": (now - timedelta(hours=12)).isoformat(),
            "completedTime": (now - timedelta(hours=11)).isoformat(),
            "details": {
                "delta_v": 0.02,
                "duration": 15.0,
                "fuel_required": 5.2
            },
            "created_by": "user@example.com",
            "created_at": (now - timedelta(hours=24)).isoformat()
        },
        {
            "id": "mnv-002",
            "satellite_id": "sat-001",
            "status": "scheduled",
            "type": "station_keeping",
            "scheduledTime": (now + timedelta(hours=5)).isoformat(),
            "completedTime": None,
            "details": {
                "delta_v": 0.01,
                "duration": 10.0,
                "fuel_required": 2.1
            },
            "created_by": "user@example.com",
            "created_at": (now - timedelta(hours=3)).isoformat()
        }
    ]
    return maneuvers

# Satellites endpoint (mocked data)
@app.get("/api/v1/satellites")
async def get_satellites():
    """
    Get a list of satellites (mocked)
    """
    satellites = [
        {
            "id": "sat-001",
            "name": "AstroShield-1",
            "status": "operational",
            "orbit": {
                "altitude": 500.2,
                "inclination": 45.0,
                "eccentricity": 0.001
            }
        },
        {
            "id": "sat-002",
            "name": "AstroShield-2",
            "status": "operational",
            "orbit": {
                "altitude": 525.7,
                "inclination": 52.5,
                "eccentricity": 0.002
            }
        }
    ]
    return satellites

# Root endpoint for basic navigation
@app.get("/")
async def root():
    """
    Root endpoint with basic API information
    """
    return {
        "message": "Welcome to AstroShield API",
        "documentation": "/docs",
        "version": "1.0.0",
        "status": "Operational"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3001) 