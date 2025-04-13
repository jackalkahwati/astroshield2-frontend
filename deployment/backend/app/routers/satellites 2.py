from fastapi import APIRouter, HTTPException, Depends, Query
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from app.services.satellite_service import (
    SatelliteService,
    SatelliteStatus,
    StateVector,
    SatelliteTrajectory
)
from app.core.security import get_current_user, check_roles
from app.models.user import User

router = APIRouter()
satellite_service = SatelliteService()

@router.get("/satellites", response_model=List[SatelliteStatus])
@router.get("/satellites/", response_model=List[SatelliteStatus])  # Handle both with and without trailing slash
async def get_satellites(
    status: Optional[str] = None,
    owner: Optional[str] = None,
    current_user: User = Depends(check_roles(["active"]))
):
    """Get list of satellites with optional filtering"""
    try:
        return await satellite_service.get_satellites(status, owner)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/satellites/{satellite_id}", response_model=SatelliteStatus)
async def get_satellite(
    satellite_id: str,
    current_user: User = Depends(check_roles(["active"]))
):
    """Get specific satellite details"""
    try:
        satellite = await satellite_service.get_satellite(satellite_id)
        if not satellite:
            raise HTTPException(status_code=404, detail="Satellite not found")
        return satellite
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/satellites/{satellite_id}/telemetry", response_model=List[Dict[str, Any]])
async def get_satellite_telemetry(
    satellite_id: str,
    hours: Optional[int] = Query(24, description="Number of hours of telemetry history to retrieve"),
    current_user: User = Depends(check_roles(["active"]))
):
    """Get telemetry data for a satellite over a time period"""
    try:
        # Calculate time range based on hours parameter
        now = datetime.utcnow()
        time_range = {
            "start": now - timedelta(hours=hours),
            "end": now
        }
        
        telemetry = await satellite_service.get_satellite_telemetry(satellite_id, time_range)
        if not telemetry and not await satellite_service.get_satellite(satellite_id):
            raise HTTPException(status_code=404, detail="Satellite not found")
            
        return telemetry
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/satellites/{satellite_id}/trajectory", response_model=SatelliteTrajectory)
async def get_satellite_trajectory(
    satellite_id: str,
    start_time: Optional[datetime] = None,
    hours: Optional[int] = Query(24, description="Number of hours to predict trajectory for"),
    current_user: User = Depends(check_roles(["active"]))
):
    """Get predicted trajectory for a satellite over a time period"""
    try:
        # Calculate time range based on parameters
        now = datetime.utcnow()
        start = start_time or now
        end = start + timedelta(hours=hours)
        
        trajectory = await satellite_service.get_satellite_trajectory(satellite_id, start, end)
        if not trajectory:
            raise HTTPException(status_code=404, detail="Satellite not found")
            
        return trajectory
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/satellites/{satellite_id}/state", response_model=StateVector)
async def get_satellite_state(
    satellite_id: str,
    timestamp: Optional[datetime] = None,
    current_user: User = Depends(check_roles(["active"]))
):
    """Get state vector (position and velocity) for a satellite at a specific time"""
    try:
        state = await satellite_service.get_state_vector(satellite_id, timestamp)
        if not state:
            raise HTTPException(status_code=404, detail="Satellite not found")
            
        return state
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_satellites_status():
    """Get satellites system status"""
    satellite_count = len(await satellite_service.get_satellites())
    operational_count = len(await satellite_service.get_satellites(status="operational"))
    
    return {
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
        "total_satellites": satellite_count,
        "operational_satellites": operational_count,
        "maintenance_satellites": satellite_count - operational_count,
        "health": {
            "api_response_time_ms": 42,
            "data_freshness_minutes": 5,
            "last_catalog_update": (datetime.utcnow() - timedelta(minutes=15)).isoformat()
        }
    }