from fastapi import APIRouter, HTTPException, Depends, Query, Path, status
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from datetime import datetime

from app.models.trajectory import (
    TrajectoryConfig, TrajectoryRequest, TrajectoryComparisonCreate,
    TrajectoryInDB, TrajectoryCreate, TrajectoryUpdate, TrajectoryComparison
)
from app.models.user import User
from app.services.trajectory_service import TrajectoryService
from app.db.session import get_db
from app.core.security import check_roles

router = APIRouter()

@router.post("/trajectory/analyze", status_code=status.HTTP_200_OK)
async def analyze_trajectory(request: TrajectoryRequest, 
                            db: Session = Depends(get_db),
                            current_user: Optional[User] = Depends(check_roles(["active"]))  # Can be run without auth for now
                            ):
    """Analyze trajectory with given configuration and initial state."""
    try:
        service = TrajectoryService(db)
        result = service.predictor.analyze_trajectory(
            config=request.config.dict(), 
            initial_state=request.initial_state
        )
        
        # Save results if user is authenticated
        if current_user:
            service.analyze_trajectory(
                config=request.config.dict(),
                initial_state=request.initial_state,
                user_id=current_user.id,
                save=True
            )
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error in trajectory analysis: {str(e)}"
        )

@router.post("/trajectories/", response_model=TrajectoryInDB, status_code=status.HTTP_201_CREATED)
async def create_trajectory(data: TrajectoryCreate, 
                           db: Session = Depends(get_db),
                           current_user: User = Depends(check_roles(["active"]))):
    """Create a new trajectory record."""
    service = TrajectoryService(db)
    return service.create_trajectory(data, current_user.id)

@router.get("/trajectories/", response_model=List[TrajectoryInDB])
async def list_trajectories(skip: int = 0, 
                           limit: int = 100,
                           db: Session = Depends(get_db),
                           current_user: User = Depends(check_roles(["active"]))):
    """List trajectories for the current user."""
    service = TrajectoryService(db)
    return service.list_trajectories(current_user.id, skip, limit)

@router.get("/trajectories/{trajectory_id}", response_model=TrajectoryInDB)
async def get_trajectory(trajectory_id: int = Path(..., title="Trajectory ID"),
                        db: Session = Depends(get_db),
                        current_user: User = Depends(check_roles(["active"]))):
    """Get a specific trajectory by ID."""
    service = TrajectoryService(db)
    trajectory = service.get_trajectory(trajectory_id)
    
    if not trajectory:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Trajectory not found"
        )
    
    # Check if user owns this trajectory or is admin
    if trajectory.user_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this trajectory"
        )
        
    return trajectory

@router.put("/trajectories/{trajectory_id}", response_model=TrajectoryInDB)
async def update_trajectory(trajectory_id: int,
                           data: TrajectoryUpdate,
                           db: Session = Depends(get_db),
                           current_user: User = Depends(check_roles(["active"]))):
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
                           current_user: User = Depends(check_roles(["active"]))):
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

@router.post("/trajectory/compare", response_model=Dict[str, Any])
async def create_trajectory_comparison(data: TrajectoryComparisonCreate,
                                     db: Session = Depends(get_db),
                                     current_user: User = Depends(check_roles(["active"]))):
    """Create a comparison between multiple trajectories."""
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
