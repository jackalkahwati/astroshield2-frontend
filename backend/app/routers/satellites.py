from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional

router = APIRouter()

class SatelliteStatus(BaseModel):
    id: str
    name: str
    status: str
    last_update: str
    orbit_parameters: dict[str, float]

@router.get("/satellites", response_model=List[SatelliteStatus])
@router.get("/satellites/", response_model=List[SatelliteStatus])  # Handle both with and without trailing slash
async def get_satellites():
    """Get list of satellites"""
    current_time = datetime.utcnow()  # Use UTC time consistently
    return [
        {
            "id": "sat-001",
            "name": "AstroShield-1",
            "status": "active",
            "last_update": current_time.isoformat(),
            "orbit_parameters": {
                "altitude": 500.5,
                "inclination": 45.0,
                "eccentricity": 0.001
            }
        }
    ]

@router.get("/satellites/{satellite_id}", response_model=SatelliteStatus)
async def get_satellite(satellite_id: str):
    """Get specific satellite details"""
    current_time = datetime.utcnow()  # Use UTC time consistently
    if satellite_id == "sat-001":
        return {
            "id": satellite_id,
            "name": "AstroShield-1",
            "status": "active",
            "last_update": current_time.isoformat(),
            "orbit_parameters": {
                "altitude": 500.5,
                "inclination": 45.0,
                "eccentricity": 0.001
            }
        }
    raise HTTPException(status_code=404, detail="Satellite not found") 