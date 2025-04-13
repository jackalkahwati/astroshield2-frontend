"""Trajectory analysis service that integrates with the physics model."""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from sqlalchemy.orm import Session

from app.models.trajectory import (
    TrajectoryCreate, TrajectoryORM, TrajectoryUpdate, 
    TrajectoryInDB, TrajectoryComparisonCreate, TrajectoryComparisonORM,
    TrajectoryResult
)
from app.services.trajectory_predictor import TrajectoryPredictor


class TrajectoryService:
    """Service for trajectory analysis and management."""

    def __init__(self, db: Session):
        """Initialize the service with a database session."""
        self.db = db
        self.predictor = TrajectoryPredictor()

    def create_trajectory(self, data: TrajectoryCreate, user_id: Optional[int] = None) -> TrajectoryInDB:
        """Create a new trajectory record."""
        # Convert Pydantic model to ORM model
        trajectory_orm = TrajectoryORM(
            name=data.name,
            description=data.description,
            config=data.config.dict(),
            initial_state=data.initial_state,
            created_at=datetime.utcnow(),
            user_id=user_id
        )
        
        # Add to database
        self.db.add(trajectory_orm)
        self.db.commit()
        self.db.refresh(trajectory_orm)
        
        # Convert back to Pydantic model and return
        return TrajectoryInDB(
            id=trajectory_orm.id,
            name=trajectory_orm.name,
            description=trajectory_orm.description,
            config=data.config,
            initial_state=trajectory_orm.initial_state,
            created_at=trajectory_orm.created_at,
            user_id=trajectory_orm.user_id
        )
    
    def get_trajectory(self, trajectory_id: int) -> Optional[TrajectoryInDB]:
        """Get a trajectory by ID."""
        trajectory = self.db.query(TrajectoryORM).filter(TrajectoryORM.id == trajectory_id).first()
        if not trajectory:
            return None
            
        return TrajectoryInDB(
            id=trajectory.id,
            name=trajectory.name,
            description=trajectory.description,
            config=trajectory.config,
            initial_state=trajectory.initial_state,
            result=trajectory.result,
            created_at=trajectory.created_at,
            updated_at=trajectory.updated_at,
            user_id=trajectory.user_id
        )
    
    def list_trajectories(self, user_id: Optional[int] = None, skip: int = 0, limit: int = 100) -> List[TrajectoryInDB]:
        """List trajectories, optionally filtered by user ID."""
        query = self.db.query(TrajectoryORM)
        if user_id is not None:
            query = query.filter(TrajectoryORM.user_id == user_id)
            
        trajectories = query.offset(skip).limit(limit).all()
        
        return [
            TrajectoryInDB(
                id=traj.id,
                name=traj.name,
                description=traj.description,
                config=traj.config,
                initial_state=traj.initial_state,
                result=traj.result,
                created_at=traj.created_at,
                updated_at=traj.updated_at,
                user_id=traj.user_id
            ) for traj in trajectories
        ]
    
    def update_trajectory(self, trajectory_id: int, data: TrajectoryUpdate) -> Optional[TrajectoryInDB]:
        """Update a trajectory's metadata."""
        trajectory = self.db.query(TrajectoryORM).filter(TrajectoryORM.id == trajectory_id).first()
        if not trajectory:
            return None
            
        # Update fields from the data model
        for field, value in data.dict(exclude_unset=True).items():
            setattr(trajectory, field, value)
            
        trajectory.updated_at = datetime.utcnow()
        
        self.db.commit()
        self.db.refresh(trajectory)
        
        return TrajectoryInDB(
            id=trajectory.id,
            name=trajectory.name,
            description=trajectory.description,
            config=trajectory.config,
            initial_state=trajectory.initial_state,
            result=trajectory.result,
            created_at=trajectory.created_at,
            updated_at=trajectory.updated_at,
            user_id=trajectory.user_id
        )
    
    def delete_trajectory(self, trajectory_id: int) -> bool:
        """Delete a trajectory record."""
        trajectory = self.db.query(TrajectoryORM).filter(TrajectoryORM.id == trajectory_id).first()
        if not trajectory:
            return False
            
        self.db.delete(trajectory)
        self.db.commit()
        return True
    
    def analyze_trajectory(self, config: Dict[str, Any], initial_state: List[float], user_id: Optional[int] = None, save: bool = True) -> TrajectoryResult:
        """Run trajectory analysis and optionally save the results."""
        # Run analysis
        try:
            result = self.predictor.analyze_trajectory(config, initial_state)
            
            # Save results if requested
            if save and user_id:
                # Create a record with a default name based on timestamp
                trajectory_create = TrajectoryCreate(
                    name=f"Trajectory Analysis {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}",
                    config=config,
                    initial_state=initial_state,
                    user_id=user_id
                )
                
                trajectory = self.create_trajectory(trajectory_create, user_id)
                
                # Update with results
                trajectory_orm = self.db.query(TrajectoryORM).filter(TrajectoryORM.id == trajectory.id).first()
                trajectory_orm.result = result.dict()
                trajectory_orm.updated_at = datetime.utcnow()
                
                self.db.commit()
            
            return result
            
        except Exception as e:
            # Log the error
            print(f"Error during trajectory analysis: {str(e)}")
            raise
    
    def create_comparison(self, data: TrajectoryComparisonCreate, user_id: Optional[int] = None) -> Dict[str, Any]:
        """Create a comparison between multiple trajectories."""
        # Verify all trajectories exist
        trajectories = []
        for traj_id in data.trajectory_ids:
            traj = self.db.query(TrajectoryORM).filter(TrajectoryORM.id == traj_id).first()
            if not traj or not traj.result:
                raise ValueError(f"Trajectory {traj_id} not found or has no results")
            trajectories.append(traj)
            
        # Compute comparison metrics
        comparison_metrics = self._compute_comparison_metrics(trajectories)
        
        # Create comparison record
        comparison = TrajectoryComparisonORM(
            name=data.name,
            description=data.description,
            trajectory_ids=data.trajectory_ids,
            comparison_metrics=comparison_metrics,
            created_at=datetime.utcnow(),
            user_id=user_id
        )
        
        self.db.add(comparison)
        self.db.commit()
        self.db.refresh(comparison)
        
        return {
            "id": comparison.id,
            "name": comparison.name,
            "description": comparison.description,
            "trajectory_ids": comparison.trajectory_ids,
            "comparison_metrics": comparison.comparison_metrics,
            "created_at": comparison.created_at,
            "user_id": comparison.user_id
        }
    
    def _compute_comparison_metrics(self, trajectories: List[TrajectoryORM]) -> Dict[str, Any]:
        """Compute metrics for comparing trajectories."""
        # Extract key metrics for each trajectory
        metrics = {
            "impact_locations": [],
            "impact_times": [],
            "impact_velocities": [],
            "uncertainty_radii": [],
            "trajectory_count": len(trajectories)
        }
        
        for traj in trajectories:
            result = traj.result
            if not result or "impact_prediction" not in result:
                continue
                
            impact = result["impact_prediction"]
            metrics["impact_locations"].append({
                "trajectory_id": traj.id,
                "name": traj.name,
                "lat": impact["location"]["lat"],
                "lon": impact["location"]["lon"]
            })
            
            metrics["impact_times"].append({
                "trajectory_id": traj.id,
                "name": traj.name,
                "time": impact["time"]
            })
            
            metrics["impact_velocities"].append({
                "trajectory_id": traj.id,
                "name": traj.name,
                "magnitude": impact["velocity"]["magnitude"]
            })
            
            metrics["uncertainty_radii"].append({
                "trajectory_id": traj.id,
                "name": traj.name,
                "radius_km": impact["uncertainty_radius_km"]
            })
        
        # Compute summary statistics
        if metrics["impact_locations"]:
            # Calculate centroid of impact locations
            lats = [loc["lat"] for loc in metrics["impact_locations"]]
            lons = [loc["lon"] for loc in metrics["impact_locations"]]
            metrics["impact_centroid"] = {
                "lat": sum(lats) / len(lats),
                "lon": sum(lons) / len(lons)
            }
            
            # Calculate maximum distance between impact points
            max_distance = 0
            for i in range(len(metrics["impact_locations"])):
                for j in range(i+1, len(metrics["impact_locations"])):
                    loc1 = metrics["impact_locations"][i]
                    loc2 = metrics["impact_locations"][j]
                    distance = self._haversine_distance(
                        loc1["lat"], loc1["lon"],
                        loc2["lat"], loc2["lon"]
                    )
                    max_distance = max(max_distance, distance)
            
            metrics["max_impact_distance_km"] = max_distance
        
        return metrics
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate the great circle distance between two points on earth (specified in decimal degrees)"""
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371  # Radius of earth in kilometers
        return c * r
