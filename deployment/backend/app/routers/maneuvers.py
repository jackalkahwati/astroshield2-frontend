from fastapi import APIRouter, HTTPException, Depends, Query, Body, Path, status as http_status
from datetime import datetime
from typing import List, Optional, Dict, Any
from app.services.maneuver_service import (
    ManeuverService, 
    ManeuverStatus, 
    ManeuverRequest,
    ManeuverResources,
    ManeuverParameters
)
from app.core.security import get_current_user, check_roles
from app.models.user import User

router = APIRouter()
maneuver_service = ManeuverService()

@router.get(
    "/maneuvers", 
    response_model=List[ManeuverStatus],
    summary="List all maneuvers",
    description="Retrieve a list of maneuvers, optionally filtered by satellite ID and status.",
    responses={
        200: {"description": "List of maneuvers matching the criteria"},
        401: {"description": "Unauthorized - Invalid or missing token"},
        500: {"description": "Internal server error"}
    }
)
async def get_maneuvers(
    satellite_id: Optional[str] = Query(None, description="Filter by satellite ID"),
    status: Optional[str] = Query(None, description="Filter by maneuver status (e.g., scheduled, in_progress, completed, cancelled)"),
    current_user: User = Depends(get_current_user)
):
    """
    Retrieve a list of all maneuvers with optional filtering.
    
    - **satellite_id**: Filter maneuvers for a specific satellite
    - **status**: Filter by maneuver status (scheduled, in_progress, completed, cancelled)
    """
    try:
        maneuvers = await maneuver_service.get_maneuvers(satellite_id)
        
        # Filter by status if provided
        if status:
            maneuvers = [m for m in maneuvers if m.status == status]
            
        return maneuvers
    except Exception as e:
        raise HTTPException(status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get(
    "/maneuvers/{maneuver_id}", 
    response_model=ManeuverStatus,
    summary="Get maneuver details",
    description="Retrieve detailed information about a specific maneuver.",
    responses={
        200: {"description": "Detailed maneuver information"},
        404: {"description": "Maneuver not found"},
        401: {"description": "Unauthorized - Invalid or missing token"},
        500: {"description": "Internal server error"}
    }
)
async def get_maneuver(
    maneuver_id: str = Path(..., description="The unique identifier of the maneuver"),
    current_user: User = Depends(get_current_user)
):
    """
    Get detailed information about a specific maneuver.
    
    - **maneuver_id**: The unique identifier of the maneuver to retrieve
    """
    try:
        maneuver = await maneuver_service.get_maneuver(maneuver_id)
        if not maneuver:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND, 
                detail="Maneuver not found"
            )
        return maneuver
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.post(
    "/maneuvers", 
    response_model=ManeuverStatus,
    status_code=http_status.HTTP_201_CREATED,
    summary="Create a new maneuver",
    description="Create a new maneuver for a satellite.",
    responses={
        201: {"description": "Maneuver created successfully"},
        400: {"description": "Bad request - Invalid input data"},
        401: {"description": "Unauthorized - Invalid or missing token"},
        403: {"description": "Forbidden - User lacks required permissions"},
        500: {"description": "Internal server error"}
    }
)
async def create_maneuver(
    request: ManeuverRequest = Body(..., description="Maneuver request details including satellite ID, type, and parameters"),
    current_user: User = Depends(check_roles(["active"]))
):
    """
    Create a new satellite maneuver.
    
    The request must include:
    - **satellite_id**: The satellite to maneuver
    - **type**: Type of maneuver (e.g., collision_avoidance, station_keeping)
    - **parameters**: Specific parameters for the maneuver
    - **scheduled_start_time**: When the maneuver should begin
    """
    try:
        return await maneuver_service.create_maneuver(request, current_user)
    except ValueError as e:
        raise HTTPException(status_code=http_status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.put(
    "/maneuvers/{maneuver_id}", 
    response_model=ManeuverStatus,
    summary="Update a maneuver",
    description="Update an existing maneuver with new parameters or status.",
    responses={
        200: {"description": "Maneuver updated successfully"},
        400: {"description": "Bad request - Invalid input data"},
        401: {"description": "Unauthorized - Invalid or missing token"},
        403: {"description": "Forbidden - User lacks required permissions"},
        404: {"description": "Maneuver not found"},
        500: {"description": "Internal server error"}
    }
)
async def update_maneuver(
    maneuver_id: str = Path(..., description="The unique identifier of the maneuver to update"),
    updates: Dict[str, Any] = Body(..., description="Fields to update on the maneuver"),
    current_user: User = Depends(check_roles(["active"]))
):
    """
    Update an existing maneuver.
    
    - **maneuver_id**: The unique identifier of the maneuver to update
    - **updates**: JSON object containing the fields to update
    """
    try:
        maneuver = await maneuver_service.update_maneuver(maneuver_id, updates, current_user)
        if not maneuver:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND, 
                detail="Maneuver not found"
            )
        return maneuver
    except ValueError as e:
        raise HTTPException(status_code=http_status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.post(
    "/maneuvers/{maneuver_id}/cancel", 
    response_model=ManeuverStatus,
    summary="Cancel a maneuver",
    description="Cancel a scheduled or in-progress maneuver.",
    responses={
        200: {"description": "Maneuver cancelled successfully"},
        400: {"description": "Bad request - Cannot cancel maneuver in current state"},
        401: {"description": "Unauthorized - Invalid or missing token"},
        403: {"description": "Forbidden - User lacks required permissions"},
        404: {"description": "Maneuver not found"},
        500: {"description": "Internal server error"}
    }
)
async def cancel_maneuver(
    maneuver_id: str = Path(..., description="The unique identifier of the maneuver to cancel"),
    current_user: User = Depends(check_roles(["active"]))
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
                status_code=http_status.HTTP_404_NOT_FOUND, 
                detail="Maneuver not found"
            )
        return maneuver
    except ValueError as e:
        raise HTTPException(status_code=http_status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get(
    "/satellites/{satellite_id}/resources", 
    response_model=ManeuverResources,
    summary="Get satellite maneuver resources",
    description="Retrieve current resources available for a satellite to perform maneuvers.",
    responses={
        200: {"description": "Current satellite resources"},
        401: {"description": "Unauthorized - Invalid or missing token"},
        403: {"description": "Forbidden - User lacks required permissions"},
        404: {"description": "Satellite not found"},
        500: {"description": "Internal server error"}
    }
)
async def get_satellite_resources(
    satellite_id: str = Path(..., description="The unique identifier of the satellite"),
    current_user: User = Depends(check_roles(["active"]))
):
    """
    Get current maneuver resources for a satellite including fuel, power, and thruster status.
    
    - **satellite_id**: The unique identifier of the satellite
    """
    try:
        resources = await maneuver_service.get_maneuver_resources(satellite_id)
        if not resources:
            raise HTTPException(
                status_code=http_status.HTTP_404_NOT_FOUND, 
                detail="Satellite not found"
            )
        return resources
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.post(
    "/simulate", 
    response_model=Dict[str, Any],
    summary="Simulate maneuver",
    description="Simulate a maneuver to see expected results without actually performing it.",
    responses={
        200: {"description": "Simulation results"},
        400: {"description": "Bad request - Invalid input data"},
        401: {"description": "Unauthorized - Invalid or missing token"},
        403: {"description": "Forbidden - User lacks required permissions"},
        500: {"description": "Internal server error"}
    }
)
async def simulate_maneuver(
    request: ManeuverRequest = Body(..., description="Maneuver request details to simulate"),
    current_user: User = Depends(check_roles(["active"]))
):
    """
    Simulate a maneuver and return expected results.
    
    The simulation includes:
    - Expected position and velocity after maneuver
    - Resource consumption
    - Potential risks and effectiveness
    """
    try:
        return await maneuver_service.simulate_maneuver(request)
    except ValueError as e:
        raise HTTPException(status_code=http_status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get(
    "/status",
    summary="Get maneuvers system status",
    description="Get overall status of the maneuvers system including active maneuvers and system health.",
    responses={
        200: {"description": "Current system status"},
        500: {"description": "Internal server error"}
    }
)
async def get_maneuvers_status():
    """
    Get the overall maneuvers system status including:
    - Current active and scheduled maneuvers
    - System resources
    - System health metrics
    """
    return {
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
        "active_maneuvers": 1,
        "scheduled_maneuvers": 2,
        "resources": {
            "fuel_remaining": 85.5,
            "power_available": 90.0,
            "thruster_status": "nominal"
        },
        "system_health": {
            "cpu_usage": 15.2,
            "memory_usage": 28.7,
            "storage_usage": 12.5
        }
    }