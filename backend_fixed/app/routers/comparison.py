"""API endpoints for trajectory comparisons."""

from fastapi import APIRouter, HTTPException, Depends, Query, Path, status
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models.trajectory import (
    TrajectoryComparisonCreate, TrajectoryComparison,
    TrajectoryInDB
)
from app.models.user import User
from app.services.trajectory_service import TrajectoryService
from app.db.session import get_db
from app.core.security import check_roles
from app.core.roles import Roles

router = APIRouter()

@router.get("/comparisons/", response_model=List[TrajectoryComparison])
async def list_comparisons(skip: int = 0, 
                          limit: int = 100,
                          db: Session = Depends(get_db),
                          current_user: User = Depends(check_roles([Roles.viewer]))):
    """List trajectory comparisons for the current user."""
    # This would be implemented to query the database for comparisons
    # For now, we'll return a placeholder empty list
    return []

@router.get("/comparisons/{comparison_id}", response_model=Dict[str, Any])
async def get_comparison(comparison_id: int = Path(..., title="Comparison ID"),
                        db: Session = Depends(get_db),
                        current_user: User = Depends(check_roles([Roles.viewer]))):
    """Get a specific comparison by ID with detailed metrics."""
    # This would query the database for the comparison
    # For now, return a 404 since it's not implemented yet
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Comparison not found or not yet implemented"
    )

@router.get("/comparisons/{comparison_id}/trajectories", response_model=List[TrajectoryInDB])
async def get_comparison_trajectories(comparison_id: int = Path(..., title="Comparison ID"),
                                     db: Session = Depends(get_db),
                                     current_user: User = Depends(check_roles([Roles.viewer]))):
    """Get all trajectories associated with a comparison."""
    # This would query the database for the trajectories in the comparison
    # For now, return a 404 since it's not implemented yet
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Comparison not found or not yet implemented"
    )

@router.post("/comparisons/", response_model=Dict[str, Any])
async def create_comparison(data: TrajectoryComparisonCreate,
                           db: Session = Depends(get_db),
                           current_user: User = Depends(check_roles([Roles.viewer]))):
    """Create a new comparison between multiple trajectories."""
    service = TrajectoryService(db)
    
    # Verify user has access to all trajectories
    for trajectory_id in data.trajectory_ids:
        trajectory = service.get_trajectory(trajectory_id)
        if not trajectory:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Trajectory {trajectory_id} not found"
            )
            
        if trajectory.user_id != current_user.id and not current_user.is_superuser:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Not authorized to access trajectory {trajectory_id}"
            )
    
    try:
        return service.create_comparison(data, current_user.id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating comparison: {str(e)}"
        )