"""
Maneuver management routes
"""
from typing import List
from fastapi import APIRouter, HTTPException

from app.common.logging import logger
from app.maneuvers.models import Maneuver, ManeuverCreateRequest
from app.maneuvers.service import get_all_maneuvers, get_maneuver_by_id, create_new_maneuver

# Create router for maneuver endpoints
router = APIRouter(prefix="/api/v1/maneuvers", tags=["maneuvers"])

@router.get("", response_model=List[Maneuver])
async def get_maneuvers():
    """Get all maneuvers"""
    return get_all_maneuvers()

@router.post("", response_model=Maneuver)
async def create_maneuver(request: ManeuverCreateRequest):
    """Create a new maneuver"""
    return create_new_maneuver(request)

@router.get("/{maneuver_id}", response_model=Maneuver)
async def get_maneuver(maneuver_id: str):
    """Get a specific maneuver by ID"""
    maneuver = get_maneuver_by_id(maneuver_id)
    if not maneuver:
        raise HTTPException(status_code=404, detail=f"Maneuver with ID {maneuver_id} not found")
    return maneuver 