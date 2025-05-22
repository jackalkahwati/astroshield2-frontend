from fastapi import APIRouter, HTTPException, Depends, Query, Path, status, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from datetime import datetime

from app.models.trajectory import (
    TrajectoryConfig, TrajectoryRequest, TrajectoryComparisonCreate,
    TrajectoryInDB, TrajectoryCreate, TrajectoryUpdate, TrajectoryComparison,
    TrajectoryResult
)
from app.models.user import User
from app.services.trajectory_service import TrajectoryService
from app.db.session import get_db
from app.core.security import check_roles
from app.core.roles import Roles
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post(
    "/trajectory/analyze",
    response_model=TrajectoryResult,
    summary="Analyze trajectory",
    description="Analyze trajectory based on provided parameters and initial state."
)
async def analyze_trajectory(
    request: TrajectoryRequest = Body(...),
    current_user: User = Depends(check_roles([Roles.viewer]))
):
    """
    Analyze a trajectory and return predictions.
    
    Parameters:
    - config: Configuration for the trajectory analysis including object properties
    - initial_state: Initial position and velocity of the object [x, y, z, vx, vy, vz]
    """
    try:
        logger.info(f"Analyzing trajectory for user {current_user.email}")
        trajectory_data = trajectory_service.create_trajectory(
            TrajectoryCreate(
                name=request.config.object_name,
                description=f"Trajectory analysis for {request.config.object_name}",
                config=request.config,
                initial_state=request.initial_state
            ),
            current_user.id
        )
        
        # Analyze the trajectory using the physics model
        result = trajectory_service.analyze_trajectory(
            trajectory_data.id,
            request.config,
            request.initial_state
        )
        
        return result
    except Exception as e:
        logger.error(f"Error analyzing trajectory: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing trajectory: {str(e)}")

@router.post("/trajectories/", response_model=TrajectoryInDB, status_code=status.HTTP_201_CREATED)
async def create_trajectory(data: TrajectoryCreate, 
                           db: Session = Depends(get_db),
                           current_user: User = Depends(check_roles([Roles.viewer]))):
    """Create a new trajectory record."""
    service = TrajectoryService(db)
    return service.create_trajectory(data, current_user.id)

@router.get(
    "/trajectories",
    response_model=List[Dict[str, Any]],
    summary="List trajectories",
    description="List all trajectories created by the current user."
)
async def list_trajectories(
    skip: int = Query(0, description="Number of records to skip"),
    limit: int = Query(100, description="Maximum number of records to return"),
    current_user: User = Depends(check_roles([Roles.viewer]))
):
    """
    List all trajectories created by the current user.
    
    Parameters:
    - skip: Number of records to skip
    - limit: Maximum number of records to return
    """
    trajectories = trajectory_service.list_trajectories(current_user.id, skip, limit)
    
    return [
        {
            "id": t.id,
            "name": t.name,
            "description": t.description,
            "created_at": t.created_at,
            "has_results": t.result is not None,
            "object_name": t.config.object_name if t.config else None
        }
        for t in trajectories
    ]

@router.get(
    "/trajectory/{trajectory_id}",
    response_model=TrajectoryResult,
    summary="Get trajectory details",
    description="Get detailed results for a previously analyzed trajectory."
)
async def get_trajectory(
    trajectory_id: int,
    current_user: User = Depends(check_roles([Roles.viewer]))
):
    """
    Get details of a previously analyzed trajectory.
    
    Parameters:
    - trajectory_id: The ID of the trajectory to retrieve
    """
    trajectory = trajectory_service.get_trajectory(trajectory_id)
    if not trajectory:
        raise HTTPException(status_code=404, detail="Trajectory not found")
        
    if trajectory.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to access this trajectory")
        
    if not trajectory.result:
        raise HTTPException(status_code=404, detail="Trajectory analysis not found")
        
    return trajectory.result

@router.put("/trajectories/{trajectory_id}", response_model=TrajectoryInDB)
async def update_trajectory(trajectory_id: int,
                           data: TrajectoryUpdate,
                           db: Session = Depends(get_db),
                           current_user: User = Depends(check_roles([Roles.viewer]))):
    """Update a trajectory record."""
    service = TrajectoryService(db)
    
    # Check if trajectory exists and user owns it
    trajectory = service.get_trajectory(trajectory_id)
    if not trajectory:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Trajectory not found"
        )
        
    if trajectory.user_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this trajectory"
        )
    
    updated = service.update_trajectory(trajectory_id, data)
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update trajectory"
        )
        
    return updated

@router.delete("/trajectories/{trajectory_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_trajectory(trajectory_id: int,
                           db: Session = Depends(get_db),
                           current_user: User = Depends(check_roles([Roles.viewer]))):
    """Delete a trajectory record."""
    service = TrajectoryService(db)
    
    # Check if trajectory exists and user owns it
    trajectory = service.get_trajectory(trajectory_id)
    if not trajectory:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Trajectory not found"
        )
        
    if trajectory.user_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this trajectory"
        )
    
    success = service.delete_trajectory(trajectory_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete trajectory"
        )

@router.post(
    "/trajectory/compare",
    response_model=TrajectoryComparison,
    summary="Compare trajectories",
    description="Compare multiple trajectories and identify key differences."
)
async def compare_trajectories(
    comparison: TrajectoryComparisonCreate = Body(...),
    current_user: User = Depends(check_roles([Roles.viewer]))
):
    """
    Compare multiple trajectories to identify differences and patterns.
    
    Parameters:
    - trajectory_ids: List of trajectory IDs to compare
    - name: Optional name for the comparison
    - description: Optional description of the comparison
    """
    try:
        result = trajectory_service.create_comparison(comparison, current_user.id)
        return result
    except Exception as e:
        logger.error(f"Error comparing trajectories: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error comparing trajectories: {str(e)}")
