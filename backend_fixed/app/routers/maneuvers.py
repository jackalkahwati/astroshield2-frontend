"""
Maneuver planning and execution router.
"""
from fastapi import APIRouter, Body, Depends, HTTPException, Path, Query
from typing import List, Dict, Any, Optional
from app.models.user import User
from app.core.security import check_roles
from app.services.maneuver_service import (
    ManeuverService,
    ManeuverStatus,
    ManeuverRequest,
    ManeuverResources,
    ManeuverParameters
)
from app.core.roles import Roles
import logging

logger = logging.getLogger(__name__)
router = APIRouter()
maneuver_service = ManeuverService()

@router.get(
    "/maneuvers",
    response_model=List[ManeuverStatus],
    summary="List all maneuvers",
    description="Retrieve a list of maneuvers, optionally filtered by satellite ID and status.",
)
async def get_maneuvers(
    satellite_id: Optional[str] = Query(None, description="Filter by satellite ID"),
    status: Optional[str] = Query(None, description="Filter by maneuver status (e.g., scheduled, in_progress, completed, cancelled)"),
    current_user: User = Depends(check_roles([Roles.viewer]))
):
    """
    Retrieve a list of all maneuvers with optional filtering.
    
    - **satellite_id**: Filter maneuvers for a specific satellite
    - **status**: Filter by maneuver status (scheduled, in_progress, completed, cancelled)
    """
    maneuvers = await maneuver_service.get_maneuvers(satellite_id)
    
    # Filter by status if provided
    if status:
        maneuvers = [m for m in maneuvers if m.status == status]
    
    return maneuvers

@router.get(
    "/maneuvers/{maneuver_id}",
    response_model=ManeuverStatus,
    summary="Get maneuver details",
    description="Retrieve detailed information about a specific maneuver.",
)
async def get_maneuver(
    maneuver_id: str = Path(..., description="The unique identifier of the maneuver"),
    current_user: User = Depends(check_roles([Roles.viewer]))
):
    """
    Get detailed information about a specific maneuver.
    
    - **maneuver_id**: The unique identifier of the maneuver to retrieve
    """
    maneuver = await maneuver_service.get_maneuver(maneuver_id)
    if not maneuver:
        raise HTTPException(
            status_code=404,
            detail="Maneuver not found"
        )
    return maneuver

@router.post(
    "/maneuvers",
    response_model=ManeuverStatus,
    status_code=201,
    summary="Create a new maneuver",
    description="Create a new maneuver for a satellite.",
)
async def create_maneuver(
    request: ManeuverRequest = Body(..., description="Maneuver request details including satellite ID, type, and parameters"),
    current_user: User = Depends(check_roles([Roles.viewer]))
):
    """
    Create a new satellite maneuver.
    
    Request includes:
    - **satellite_id**: The satellite to maneuver
    - **type**: Type of maneuver (e.g., collision_avoidance, station_keeping)
    - **parameters**: Specific parameters for the maneuver
    - **scheduled_start_time**: When the maneuver should begin
    """
    return await maneuver_service.create_maneuver(request, current_user)

@router.put(
    "/maneuvers/{maneuver_id}",
    response_model=ManeuverStatus,
    summary="Update a maneuver",
    description="Update an existing maneuver with new parameters or status.",
)
async def update_maneuver(
    maneuver_id: str = Path(..., description="The unique identifier of the maneuver to update"),
    updates: Dict[str, Any] = Body(..., description="Fields to update on the maneuver"),
    current_user: User = Depends(check_roles([Roles.viewer]))
):
    """
    Update an existing maneuver.
    
    - **maneuver_id**: The unique identifier of the maneuver to update
    """
    try:
        maneuver = await maneuver_service.update_maneuver(maneuver_id, updates, current_user)
        if not maneuver:
            raise HTTPException(
                status_code=404,
                detail="Maneuver not found"
            )
        return maneuver
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

@router.post(
    "/maneuvers/{maneuver_id}/cancel",
    response_model=ManeuverStatus,
    summary="Cancel a maneuver",
    description="Cancel a scheduled or in-progress maneuver.",
)
async def cancel_maneuver(
    maneuver_id: str = Path(..., description="The unique identifier of the maneuver to cancel"),
    current_user: User = Depends(check_roles([Roles.viewer]))
):
    """
    Cancel a scheduled maneuver.
    
    - **maneuver_id**: The unique identifier of the maneuver to cancel
    
    Note: Only maneuvers in 'scheduled' or 'in_progress' state can be cancelled.
    """
    try:
        maneuver = await maneuver_service.cancel_maneuver(maneuver_id, current_user)
        if not maneuver:
            raise HTTPException(
                status_code=404,
                detail="Maneuver not found"
            )
        return maneuver
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

@router.get(
    "/satellites/{satellite_id}/maneuver-resources",
    response_model=ManeuverResources,
    summary="Get satellite maneuver resources",
    description="Get the current resources available for maneuvering a satellite."
)
async def get_maneuver_resources(
    satellite_id: str = Path(..., description="The ID of the satellite"),
    current_user: User = Depends(check_roles([Roles.viewer]))
):
    """
    Get the current resources available for maneuvering a satellite.
    
    - **satellite_id**: The ID of the satellite to check
    
    Returns information about fuel, power, and thruster status.
    """
    return await maneuver_service.get_maneuver_resources(satellite_id)

@router.post(
    "/maneuvers/simulate",
    response_model=Dict[str, Any],
    summary="Simulate a maneuver",
    description="Simulate a maneuver to see expected results without executing it."
)
async def simulate_maneuver(
    request: ManeuverRequest = Body(..., description="Maneuver to simulate"),
    current_user: User = Depends(check_roles([Roles.viewer]))
):
    """
    Simulate a satellite maneuver to preview expected results.
    
    - **satellite_id**: The satellite to maneuver
    - **type**: Type of maneuver to simulate
    - **parameters**: Maneuver parameters to use in simulation
    
    Returns expected results including fuel usage and orbit changes.
    """
    return await maneuver_service.simulate_maneuver(request)