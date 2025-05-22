from sqlalchemy import Column, Integer, String, Boolean, DateTime, JSON
from sqlalchemy.orm import relationship
from datetime import datetime

from app.db.base_class import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, index=True, nullable=True)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean(), default=True, nullable=False)
    is_superuser = Column(Boolean(), default=False, nullable=False)
    roles = Column(JSON, default=lambda: [], nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=True)

    # Define relationships as needed
    # trajectories = relationship("TrajectoryORM", back_populates="user", cascade="all, delete-orphan")

    # CCDM relationships - commented out until these models are created
    # ccdm_analyses = relationship("CCDMAnalysisORM", back_populates="user")
    # threat_assessments = relationship("ThreatAssessmentORM", back_populates="user") 
    # historical_analyses = relationship("HistoricalAnalysisORM", back_populates="user")
    # shape_changes = relationship("ShapeChangeORM", back_populates="user")