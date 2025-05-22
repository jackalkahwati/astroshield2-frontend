"""API endpoints for trajectory comparisons."""

from fastapi import APIRouter, HTTPException, Depends, Query, Path, status
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime, timedelta
import uuid

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

# Mock data for trajectory comparisons
MOCK_COMPARISONS = {
    1: {
        "id": 1,
        "name": "LEO to GEO Transfer Analysis",
        "description": "Comparison of different transfer trajectories to GEO",
        "trajectory_ids": [1, 2, 3],
        "created_at": datetime.utcnow() - timedelta(days=2),
        "updated_at": datetime.utcnow() - timedelta(hours=3),
        "user_id": 1,
        "metrics": {
            "total_delta_v": [3.2, 3.4, 3.1],  # km/s for each trajectory
            "transfer_time": [5.2, 4.8, 5.5],  # hours
            "fuel_efficiency": [0.85, 0.82, 0.87],  # efficiency ratio
            "collision_risk": [0.001, 0.002, 0.0008]  # probability
        },
        "status": "completed"
    },
    2: {
        "id": 2,
        "name": "Station Keeping Strategies",
        "description": "Comparison of different station keeping approaches",
        "trajectory_ids": [4, 5],
        "created_at": datetime.utcnow() - timedelta(days=1),
        "updated_at": datetime.utcnow() - timedelta(minutes=30),
        "user_id": 1,
        "metrics": {
            "total_delta_v": [0.05, 0.045],  # km/s per month
            "fuel_consumption": [2.1, 1.9],  # kg per month
            "position_accuracy": [0.1, 0.08],  # km deviation
            "maintenance_frequency": [30, 28]  # days between maneuvers
        },
        "status": "completed"
    }
}

MOCK_TRAJECTORIES = {
    1: {
        "id": 1,
        "name": "Hohmann Transfer",
        "description": "Standard Hohmann transfer trajectory",
        "trajectory_type": "transfer",
        "created_at": datetime.utcnow() - timedelta(days=3),
        "user_id": 1,
        "orbital_elements": {
            "semi_major_axis": 26560.0,  # km
            "eccentricity": 0.7,
            "inclination": 0.0,  # degrees
            "raan": 0.0,
            "arg_periapsis": 0.0,
            "true_anomaly": 0.0
        }
    },
    2: {
        "id": 2,
        "name": "Bi-elliptic Transfer",
        "description": "Bi-elliptic transfer trajectory for efficiency",
        "trajectory_type": "transfer",
        "created_at": datetime.utcnow() - timedelta(days=3),
        "user_id": 1,
        "orbital_elements": {
            "semi_major_axis": 35000.0,  # km
            "eccentricity": 0.8,
            "inclination": 0.0,
            "raan": 0.0,
            "arg_periapsis": 0.0,
            "true_anomaly": 0.0
        }
    },
    3: {
        "id": 3,
        "name": "Spiral Transfer",
        "description": "Low-thrust spiral transfer trajectory",
        "trajectory_type": "transfer",
        "created_at": datetime.utcnow() - timedelta(days=3),
        "user_id": 1,
        "orbital_elements": {
            "semi_major_axis": 24000.0,  # km
            "eccentricity": 0.6,
            "inclination": 0.0,
            "raan": 0.0,
            "arg_periapsis": 0.0,
            "true_anomaly": 0.0
        }
    },
    4: {
        "id": 4,
        "name": "East-West Station Keeping",
        "description": "Traditional east-west station keeping",
        "trajectory_type": "station_keeping",
        "created_at": datetime.utcnow() - timedelta(days=2),
        "user_id": 1,
        "orbital_elements": {
            "semi_major_axis": 42164.0,  # km (GEO)
            "eccentricity": 0.001,
            "inclination": 0.1,
            "raan": 0.0,
            "arg_periapsis": 0.0,
            "true_anomaly": 0.0
        }
    },
    5: {
        "id": 5,
        "name": "Hybrid Station Keeping",
        "description": "Hybrid approach with north-south control",
        "trajectory_type": "station_keeping",
        "created_at": datetime.utcnow() - timedelta(days=2),
        "user_id": 1,
        "orbital_elements": {
            "semi_major_axis": 42164.0,  # km (GEO)
            "eccentricity": 0.0008,
            "inclination": 0.05,
            "raan": 0.0,
            "arg_periapsis": 0.0,
            "true_anomaly": 0.0
        }
    }
}

@router.get("/comparisons/", response_model=List[Dict[str, Any]])
async def list_comparisons(skip: int = 0, 
                          limit: int = 100,
                          db: Session = Depends(get_db),
                          # Demo mode - authentication disabled
                          # current_user: User = Depends(check_roles([Roles.viewer]))
                          ):
    """List trajectory comparisons for the current user."""
    comparisons = list(MOCK_COMPARISONS.values())
    
    # Apply pagination
    total = len(comparisons)
    comparisons = comparisons[skip:skip + limit]
    
    # Format for response
    formatted_comparisons = []
    for comp in comparisons:
        formatted_comparisons.append({
            "id": comp["id"],
            "name": comp["name"],
            "description": comp["description"],
            "trajectory_count": len(comp["trajectory_ids"]),
            "created_at": comp["created_at"].isoformat(),
            "updated_at": comp["updated_at"].isoformat(),
            "status": comp["status"]
        })
    
    return formatted_comparisons

@router.get("/comparisons/{comparison_id}", response_model=Dict[str, Any])
async def get_comparison(comparison_id: int = Path(..., title="Comparison ID"),
                        db: Session = Depends(get_db),
                        # Demo mode - authentication disabled
                        # current_user: User = Depends(check_roles([Roles.viewer]))
                        ):
    """Get a specific comparison by ID with detailed metrics."""
    if comparison_id not in MOCK_COMPARISONS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Comparison not found"
        )
    
    comparison = MOCK_COMPARISONS[comparison_id]
    
    # Add trajectory details
    trajectories = []
    for traj_id in comparison["trajectory_ids"]:
        if traj_id in MOCK_TRAJECTORIES:
            traj = MOCK_TRAJECTORIES[traj_id]
            trajectories.append({
                "id": traj["id"],
                "name": traj["name"],
                "description": traj["description"],
                "trajectory_type": traj["trajectory_type"]
            })
    
    return {
        "id": comparison["id"],
        "name": comparison["name"],
        "description": comparison["description"],
        "trajectories": trajectories,
        "metrics": comparison["metrics"],
        "created_at": comparison["created_at"].isoformat(),
        "updated_at": comparison["updated_at"].isoformat(),
        "status": comparison["status"],
        "analysis": {
            "best_delta_v": min(comparison["metrics"]["total_delta_v"]),
            "best_efficiency": max(comparison["metrics"]["fuel_efficiency"]) if "fuel_efficiency" in comparison["metrics"] else None,
            "lowest_risk": min(comparison["metrics"]["collision_risk"]) if "collision_risk" in comparison["metrics"] else None,
            "recommended_trajectory": comparison["trajectory_ids"][comparison["metrics"]["total_delta_v"].index(min(comparison["metrics"]["total_delta_v"]))]
        }
    }

@router.get("/comparisons/{comparison_id}/trajectories", response_model=List[Dict[str, Any]])
async def get_comparison_trajectories(comparison_id: int = Path(..., title="Comparison ID"),
                                     db: Session = Depends(get_db),
                                     # Demo mode - authentication disabled
                                     # current_user: User = Depends(check_roles([Roles.viewer]))
                                     ):
    """Get all trajectories associated with a comparison."""
    if comparison_id not in MOCK_COMPARISONS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Comparison not found"
        )
    
    comparison = MOCK_COMPARISONS[comparison_id]
    trajectories = []
    
    for traj_id in comparison["trajectory_ids"]:
        if traj_id in MOCK_TRAJECTORIES:
            traj = MOCK_TRAJECTORIES[traj_id]
            trajectories.append({
                "id": traj["id"],
                "name": traj["name"],
                "description": traj["description"],
                "trajectory_type": traj["trajectory_type"],
                "created_at": traj["created_at"].isoformat(),
                "orbital_elements": traj["orbital_elements"]
            })
    
    return trajectories

@router.post("/comparisons/", response_model=Dict[str, Any])
async def create_comparison(data: TrajectoryComparisonCreate,
                           db: Session = Depends(get_db),
                           # Demo mode - authentication disabled
                           # current_user: User = Depends(check_roles([Roles.viewer]))
                           ):
    """Create a new comparison between multiple trajectories."""
    
    # Validate trajectory IDs exist
    for trajectory_id in data.trajectory_ids:
        if trajectory_id not in MOCK_TRAJECTORIES:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Trajectory {trajectory_id} not found"
            )
    
    # Create new comparison
    new_id = max(MOCK_COMPARISONS.keys()) + 1 if MOCK_COMPARISONS else 1
    
    # Generate mock metrics based on trajectory count
    metrics = {
        "total_delta_v": [round(3.0 + i * 0.2, 2) for i in range(len(data.trajectory_ids))],
        "transfer_time": [round(5.0 + i * 0.3, 1) for i in range(len(data.trajectory_ids))],
        "fuel_efficiency": [round(0.85 - i * 0.02, 3) for i in range(len(data.trajectory_ids))]
    }
    
    new_comparison = {
        "id": new_id,
        "name": data.name,
        "description": data.description or f"Comparison of {len(data.trajectory_ids)} trajectories",
        "trajectory_ids": data.trajectory_ids,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "user_id": 1,  # Demo user
        "metrics": metrics,
        "status": "completed"
    }
    
    MOCK_COMPARISONS[new_id] = new_comparison
    
    return {
        "id": new_comparison["id"],
        "name": new_comparison["name"],
        "description": new_comparison["description"],
        "trajectory_ids": new_comparison["trajectory_ids"],
        "created_at": new_comparison["created_at"].isoformat(),
        "status": new_comparison["status"]
    }