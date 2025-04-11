from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship

from app.db.base_class import Base


class TrajectoryBase(BaseModel):
    """Base Trajectory model for shared attributes"""
    name: str
    description: Optional[str] = None
    created_at: datetime = datetime.utcnow()
    user_id: Optional[int] = None


class TrajectoryConfig(BaseModel):
    """Configuration for trajectory analysis."""
    atmospheric_model: str
    wind_model: str
    monte_carlo_samples: int
    object_properties: Dict[str, float]
    breakup_model: Dict[str, Any]


class TrajectoryRequest(BaseModel):
    """Request body for trajectory analysis."""
    config: TrajectoryConfig
    initial_state: List[float]


class TrajectoryPoint(BaseModel):
    """Single point in a trajectory."""
    time: float
    position: List[float]
    velocity: List[float]


class BreakupEvent(BaseModel):
    """Breakup event details."""
    time: str
    altitude: float
    fragments: int


class ImpactPrediction(BaseModel):
    """Impact prediction details."""
    time: str
    location: Dict[str, float]
    velocity: Dict[str, Any]
    uncertainty_radius_km: float
    confidence: float
    monte_carlo_stats: Optional[Dict[str, float]] = None


class TrajectoryResult(BaseModel):
    """Complete trajectory analysis result."""
    trajectory: List[TrajectoryPoint]
    impact_prediction: ImpactPrediction
    breakup_events: Optional[List[BreakupEvent]] = None


class TrajectoryCreate(TrajectoryBase):
    """Model for creating a trajectory record."""
    config: TrajectoryConfig
    initial_state: List[float]


class TrajectoryUpdate(BaseModel):
    """Model for updating a trajectory record."""
    name: Optional[str] = None
    description: Optional[str] = None


class TrajectoryInDB(TrajectoryBase):
    """Model for trajectory records in the database."""
    id: int
    config: TrajectoryConfig
    initial_state: List[float]
    result: Optional[TrajectoryResult] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# SQLAlchemy ORM models
class TrajectoryORM(Base):
    """ORM model for trajectory records."""
    __tablename__ = "trajectories"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String, nullable=True)
    config = Column(JSON, nullable=False)
    initial_state = Column(JSON, nullable=False)
    result = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    # Relationships
    user = relationship("UserORM", back_populates="trajectories")
    comparisons = relationship("TrajectoryComparisonORM", back_populates="trajectories")


class TrajectoryComparisonORM(Base):
    """ORM model for trajectory comparisons."""
    __tablename__ = "trajectory_comparisons"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(String, nullable=True)
    trajectory_ids = Column(JSON, nullable=False)  # List of trajectory IDs
    comparison_metrics = Column(JSON, nullable=True)  # Computed comparison metrics
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    # Relationships
    user = relationship("UserORM", back_populates="trajectory_comparisons")
    trajectories = relationship("TrajectoryORM", back_populates="comparisons")


class TrajectoryComparison(BaseModel):
    """Model for trajectory comparisons."""
    id: int
    name: str
    description: Optional[str] = None
    trajectory_ids: List[int]
    comparison_metrics: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    user_id: Optional[int] = None

    class Config:
        from_attributes = True


class TrajectoryComparisonCreate(BaseModel):
    """Model for creating a trajectory comparison."""
    name: str
    description: Optional[str] = None
    trajectory_ids: List[int]
