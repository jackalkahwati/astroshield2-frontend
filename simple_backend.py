#!/usr/bin/env python3
"""
Simple AstroShield Backend

This is a simplified version of the AstroShield backend API focused on trajectory analysis.
It provides basic endpoints for trajectory analysis without complex dependencies.
"""

import os
import json
import logging
import random
import math
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Depends, Query, status, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Models
class ObjectProperties(BaseModel):
    mass: float = Field(100.0, description="Mass of the object in kg")
    area: float = Field(1.2, description="Cross-sectional area in m²")
    cd: float = Field(2.2, description="Drag coefficient")

class BreakupModel(BaseModel):
    enabled: bool = Field(True, description="Whether breakup modeling is enabled")
    fragmentation_threshold: float = Field(50.0, description="Energy threshold for fragmentation in kJ")

class TrajectoryConfig(BaseModel):
    object_name: str = Field("Satellite Debris", description="Name of the object being analyzed")
    object_properties: ObjectProperties = Field(default_factory=ObjectProperties, description="Physical properties of the object")
    atmospheric_model: str = Field("exponential", description="Atmospheric model to use")
    wind_model: str = Field("custom", description="Wind model to use")
    monte_carlo_samples: int = Field(100, description="Number of Monte Carlo samples for uncertainty")
    breakup_model: BreakupModel = Field(default_factory=BreakupModel, description="Configuration for breakup modeling")

class TrajectoryRequest(BaseModel):
    config: TrajectoryConfig = Field(..., description="Configuration for the trajectory analysis")
    initial_state: List[float] = Field(..., description="Initial state vector [lon, lat, alt, vx, vy, vz]")

class TrajectoryPoint(BaseModel):
    time: float
    position: List[float]
    velocity: List[float]

class ImpactPrediction(BaseModel):
    time: float
    position: List[float]
    confidence: float
    energy: float
    area: float

class BreakupPoint(BaseModel):
    time: float
    position: List[float]
    fragments: int
    cause: str

class TrajectoryResult(BaseModel):
    trajectory: List[TrajectoryPoint]
    impactPrediction: ImpactPrediction
    breakupPoints: List[BreakupPoint]

# Create FastAPI app
app = FastAPI(title="AstroShield API", version="0.1.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Basic trajectory simulation for demo
def simulate_trajectory(config: Dict[str, Any], initial_state: List[float]) -> Dict[str, Any]:
    """Simple trajectory simulation for demonstration purposes"""
    logger.info(f"Simulating trajectory for {config['object_name']}")
    
    # Extract parameters
    mass = config["object_properties"]["mass"]
    altitude = initial_state[2]  # Initial altitude
    
    # Generate trajectory points
    num_points = 100
    trajectory_points = []
    
    start_time = datetime.now().timestamp()
    time_step = 60  # 1 minute per step
    
    # Starting position and velocity
    lon, lat, alt = initial_state[0:3]
    vx, vy, vz = initial_state[3:6]
    
    for i in range(num_points):
        # Time for this point
        time = start_time + i * time_step
        
        # Update altitude with some randomness for realism
        alt = max(0, altitude - (i * altitude / num_points) + random.uniform(-500, 500))
        
        # Adjust longitude and latitude based on velocity and Earth's curvature
        earth_radius = 6371000  # meters
        meters_per_degree_lon = 111320 * math.cos(math.radians(lat))
        meters_per_degree_lat = 110574
        
        lon_change = (vx * time_step) / meters_per_degree_lon
        lat_change = (vy * time_step) / meters_per_degree_lat
        
        lon += lon_change
        lat += lat_change
        
        # Ensure longitude wraps around properly
        lon = (lon + 180) % 360 - 180
        
        # Update velocities with some atmospheric drag effects
        if alt < 100000:  # Below 100km, significant atmosphere
            drag_factor = 1 - (alt / 100000) * 0.1
            vx *= drag_factor
            vy *= drag_factor
            vz *= drag_factor
        
        # Add trajectory point
        trajectory_points.append({
            "time": time,
            "position": [lon, lat, alt],
            "velocity": [vx, vy, vz]
        })
        
        # Terminal velocity increases as atmosphere gets denser
        if alt < 50000:
            vz -= 9.8 * time_step * (1 + (50000 - alt) / 10000)
        else:
            vz -= 9.8 * time_step * 0.5  # Reduced gravity effect at higher altitudes
    
    # Impact prediction
    impact_prediction = {
        "time": trajectory_points[-1]["time"],
        "position": trajectory_points[-1]["position"],
        "confidence": 0.95,
        "energy": 0.5 * mass * sum(v**2 for v in trajectory_points[-1]["velocity"]),
        "area": config["object_properties"]["area"] * (1 + random.uniform(0, 0.5))  # Slight expansion on impact
    }
    
    # Breakup points (if enabled)
    breakup_points = []
    if config["breakup_model"]["enabled"]:
        # Add random breakup points if altitude and energy conditions are met
        for i in range(1, len(trajectory_points) - 1):
            point = trajectory_points[i]
            if point["position"][2] < 80000 and random.random() < 0.1:  # 10% chance below 80km
                velocity_magnitude = math.sqrt(sum(v**2 for v in point["velocity"]))
                if velocity_magnitude > 2000:  # Only break up at high speeds
                    breakup_points.append({
                        "time": point["time"],
                        "position": point["position"],
                        "fragments": random.randint(5, 30),
                        "cause": random.choice(["Aerodynamic Stress", "Thermal Stress", "Material Failure"])
                    })
    
    return {
        "trajectory": trajectory_points,
        "impactPrediction": impact_prediction,
        "breakupPoints": breakup_points
    }

# Routes
@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "0.1.0"}

@app.get("/api/v1/health")
def api_health_check():
    """API health check endpoint"""
    return {
        "status": "healthy",
        "version": "0.1.0",
        "services": {
            "trajectory": "ready"
        }
    }

@app.post("/api/trajectory/analyze", response_model=TrajectoryResult)
async def analyze_trajectory(request: TrajectoryRequest):
    """
    Analyze a trajectory and return predictions.
    
    Parameters:
    - config: Configuration for the trajectory analysis including object properties
    - initial_state: Initial position and velocity of the object [x, y, z, vx, vy, vz]
    """
    try:
        logger.info(f"Analyzing trajectory for {request.config.object_name}")
        
        # Perform trajectory simulation
        result = simulate_trajectory(
            request.config.dict(),
            request.initial_state
        )
        
        return result
    except Exception as e:
        logger.error(f"Error analyzing trajectory: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing trajectory: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to the AstroShield API",
        "documentation": "/docs",
        "health": "/health"
    }

def main():
    """Run the application"""
    port = int(os.environ.get("PORT", 5002))
    logger.info(f"Starting simplified AstroShield backend on port {port}")
    uvicorn.run("simple_backend:app", host="0.0.0.0", port=port, reload=False)

if __name__ == "__main__":
    main() 