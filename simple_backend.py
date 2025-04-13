#!/usr/bin/env python3
"""
Simple FastAPI server for AstroShield.
This provides a basic API that works without complex dependencies.
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime
import random
import os
import json
import sqlite3
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AstroShield Simple API",
    description="A simple API for the AstroShield platform",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class Position(BaseModel):
    x: float
    y: float
    z: float

class Velocity(BaseModel):
    x: float
    y: float
    z: float

class Satellite(BaseModel):
    id: str
    name: str
    status: str = "active"
    epoch: Optional[str] = None
    position: Optional[Position] = None
    velocity: Optional[Velocity] = None
    last_update: Optional[str] = None

class Maneuver(BaseModel):
    id: str
    satellite_id: str
    status: str = "planned"
    type: str
    start_time: str
    end_time: Optional[str] = None
    description: Optional[str] = None

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint providing basic API information"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "api_name": "AstroShield Simple API",
        "documentation": "/docs"
    }

# Satellites endpoint
@app.get("/api/v1/satellites", response_model=List[Satellite])
async def get_satellites(
    limit: int = Query(10, ge=1, le=100)
):
    """Get list of satellites with position and velocity information"""
    satellites = []
    
    try:
        # Try to load from DB first
        db_path = os.environ.get("DATABASE_URL", "sqlite:///./astroshield.db")
        if db_path.startswith("sqlite:///"):
            db_path = db_path[len("sqlite:///"):]
            
        if os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT id, name, status FROM satellites LIMIT ?", (limit,))
                rows = cursor.fetchall()
                conn.close()
                
                for row in rows:
                    sat_id, name, status = row
                    epoch = datetime.utcnow().isoformat()
                    position = Position(
                        x=random.uniform(-7000, 7000),
                        y=random.uniform(-7000, 7000),
                        z=random.uniform(-7000, 7000)
                    )
                    velocity = Velocity(
                        x=random.uniform(-7, 7),
                        y=random.uniform(-7, 7),
                        z=random.uniform(-7, 7)
                    )
                    
                    satellites.append(Satellite(
                        id=sat_id,
                        name=name,
                        status=status,
                        epoch=epoch,
                        position=position,
                        velocity=velocity,
                        last_update=epoch
                    ))
                
                logger.info(f"Loaded {len(satellites)} satellites from database")
            except Exception as e:
                logger.error(f"Error loading from database: {str(e)}")
                # Fall through to mock data
    except Exception as e:
        logger.error(f"Database error: {str(e)}")
        
    # If no satellites loaded from DB, create mock data
    if not satellites:
        logger.info("Generating mock satellite data")
        for i in range(1, limit + 1):
            epoch = datetime.utcnow().isoformat()
            position = Position(
                x=random.uniform(-7000, 7000),
                y=random.uniform(-7000, 7000),
                z=random.uniform(-7000, 7000)
            )
            velocity = Velocity(
                x=random.uniform(-7, 7),
                y=random.uniform(-7, 7),
                z=random.uniform(-7, 7)
            )
            
            satellites.append(Satellite(
                id=f"sat-{i}",
                name=f"Test Satellite {i}",
                status="active" if i % 4 != 0 else "inactive",
                epoch=epoch,
                position=position,
                velocity=velocity,
                last_update=epoch
            ))
    
    return satellites

@app.get("/api/v1/satellites/{satellite_id}", response_model=Satellite)
async def get_satellite(satellite_id: str):
    """Get details for a specific satellite"""
    # Create a mock satellite with the requested ID
    epoch = datetime.utcnow().isoformat()
    position = Position(
        x=random.uniform(-7000, 7000),
        y=random.uniform(-7000, 7000),
        z=random.uniform(-7000, 7000)
    )
    velocity = Velocity(
        x=random.uniform(-7, 7),
        y=random.uniform(-7, 7),
        z=random.uniform(-7, 7)
    )
    
    # Try to look up in DB first
    db_path = os.environ.get("DATABASE_URL", "sqlite:///./astroshield.db")
    if db_path.startswith("sqlite:///"):
        db_path = db_path[len("sqlite:///"):]
        
    if os.path.exists(db_path):
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, status FROM satellites WHERE id = ?", (satellite_id,))
            row = cursor.fetchone()
            conn.close()
            
            if row:
                sat_id, name, status = row
                return Satellite(
                    id=sat_id,
                    name=name,
                    status=status,
                    epoch=epoch,
                    position=position,
                    velocity=velocity,
                    last_update=epoch
                )
        except Exception as e:
            logger.error(f"Database error: {str(e)}")
    
    # If not found in DB or DB access failed, create mock data
    for i in range(1, 11):
        if f"sat-{i}" == satellite_id:
            return Satellite(
                id=satellite_id,
                name=f"Test Satellite {i}",
                status="active" if i % 4 != 0 else "inactive",
                epoch=epoch,
                position=position,
                velocity=velocity,
                last_update=epoch
            )
    
    # If we get here, the satellite wasn't found
    raise HTTPException(status_code=404, detail=f"Satellite with ID {satellite_id} not found")

@app.get("/api/v1/maneuvers", response_model=List[Maneuver])
async def get_maneuvers():
    """Get a list of satellite maneuvers"""
    now = datetime.utcnow()
    
    # Mock maneuvers data
    maneuvers = [
        Maneuver(
            id="mnv-001",
            satellite_id="sat-001",
            status="completed",
            type="collision_avoidance",
            start_time=(now.replace(hour=now.hour-2)).isoformat(),
            end_time=(now.replace(hour=now.hour-1)).isoformat(),
            description="Collision avoidance maneuver"
        ),
        Maneuver(
            id="mnv-002",
            satellite_id="sat-001", 
            status="planned",
            type="station_keeping",
            start_time=(now.replace(hour=now.hour+5)).isoformat(),
            description="Scheduled station keeping"
        )
    ]
    
    return maneuvers

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@app.get("/api/v1/mock-udl-status")
async def mock_udl_status():
    """Get the status of the mock UDL service"""
    udl_base_url = os.environ.get("UDL_BASE_URL", "https://mock-udl-service.local/api/v1")
    mock_mode = udl_base_url.startswith(("http://localhost", "https://mock"))
    
    return {
        "mock_mode": mock_mode,
        "udl_url": udl_base_url,
        "status": "connected" if mock_mode else "using real UDL",
        "timestamp": datetime.utcnow().isoformat()
    }

# Run the server when executed directly
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("API_PORT", 5000))
    host = "0.0.0.0"
    
    logger.info(f"Starting simple AstroShield API on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port) 