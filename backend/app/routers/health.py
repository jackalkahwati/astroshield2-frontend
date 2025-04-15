from fastapi import APIRouter, Depends, status
from pydantic import BaseModel
from typing import Dict, Any

router = APIRouter()

class HealthResponse(BaseModel):
    status: str
    message: str

@router.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check():
    """Basic health check endpoint"""
    return {"status": "ok", "message": "The API is running"}

# Add a trajectory test endpoint
@router.get("/health/trajectory", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
async def trajectory_health_check():
    """Trajectory health check endpoint"""
    return {
        "status": "ok", 
        "message": "The trajectory endpoint is available",
        "trajectory_test": {
            "config": {
                "object_name": "Test Object",
                "atmospheric_model": "exponential",
                "wind_model": "custom",
                "monte_carlo_samples": 100,
                "object_properties": {
                    "mass": 100,
                    "area": 1.2,
                    "cd": 2.2
                },
                "breakup_model": {
                    "enabled": True,
                    "fragmentation_threshold": 50
                }
            },
            "initial_state": [0, 0, 400000, 7800, 0, 0]
        }
    } 