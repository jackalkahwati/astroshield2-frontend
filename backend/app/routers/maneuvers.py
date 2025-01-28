from fastapi import APIRouter, HTTPException
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel

class ManeuverResources(BaseModel):
    fuel_remaining: float
    power_available: float
    thruster_status: str

class ManeuverStatus(BaseModel):
    id: str
    status: str
    type: str
    start_time: datetime
    end_time: Optional[datetime]
    resources: ManeuverResources

router = APIRouter()

@router.get("/maneuvers", response_model=List[ManeuverStatus])
async def get_maneuvers():
    """Get list of maneuvers"""
    current_time = datetime.utcnow()
    # Mock data for now
    return [
        {
            "id": "mnv-001",
            "status": "completed",
            "type": "collision_avoidance",
            "start_time": current_time,  # Using same time reference
            "end_time": current_time,    # to ensure valid dates
            "resources": {
                "fuel_remaining": 85.5,
                "power_available": 90.0,
                "thruster_status": "nominal"
            }
        }
    ]

@router.get("/maneuvers/{maneuver_id}", response_model=ManeuverStatus)
async def get_maneuver(maneuver_id: str):
    """Get specific maneuver details"""
    current_time = datetime.utcnow()
    # Mock data for now
    if maneuver_id == "mnv-001":
        return {
            "id": maneuver_id,
            "status": "completed",
            "type": "collision_avoidance",
            "start_time": current_time,  # Using same time reference
            "end_time": current_time,    # to ensure valid dates
            "resources": {
                "fuel_remaining": 85.5,
                "power_available": 90.0,
                "thruster_status": "nominal"
            }
        }
    raise HTTPException(status_code=404, detail="Maneuver not found")

@router.get("/status")
async def get_maneuvers_status():
    """Get maneuvers system status"""
    return {
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
        "active_maneuvers": 1,
        "resources": {
            "fuel_remaining": 85.5,
            "power_available": 90.0,
            "thruster_status": "nominal"
        }
    } 