from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional

router = APIRouter()

class DashboardData(BaseModel):
    metrics: dict[str, float]
    status: str
    alerts: List[str]
    timestamp: str

@router.get("/dashboard", response_model=DashboardData)
@router.get("/dashboard/", response_model=DashboardData)  # Handle both with and without trailing slash
async def get_dashboard_data():
    try:
        current_time = datetime.utcnow()  # Use UTC time consistently
        # Mock data for demonstration
        return {
            "metrics": {
                "orbit_stability": 95.5,
                "power_efficiency": 88.2,
                "thermal_control": 92.7,
                "communication_quality": 97.1,
                "protection_coverage": 94.3
            },
            "status": "operational",
            "alerts": [
                "Minor thermal fluctuation in Sector A",
                "Scheduled maintenance in 48 hours"
            ],
            "timestamp": current_time.isoformat()  # Use the same time reference
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_dashboard_status():
    """Get dashboard system status"""
    current_time = datetime.utcnow()  # Use UTC time consistently
    return {
        "status": "operational",
        "timestamp": current_time.isoformat(),  # Use the same time reference
        "subsystems": {
            "ccdm": "operational",
            "analytics": "operational",
            "maneuvers": "operational"
        }
    } 