"""Database models for CCDM functionality."""

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON, Enum, Boolean, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import enum

from backend.infrastructure.database import Base

class ThreatLevel(str, enum.Enum):
    """Threat level enum for CCDM assessments."""
    NONE = "NONE"
    LOW = "LOW" 
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class Spacecraft(Base):
    """Model representing a spacecraft."""
    __tablename__ = "spacecraft"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    norad_id = Column(String(20), unique=True, nullable=True)
    object_type = Column(String(50), nullable=True)
    owner = Column(String(100), nullable=True)
    launch_date = Column(DateTime, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    indicators = relationship("CCDMIndicator", back_populates="spacecraft")
    assessments = relationship("CCDMAssessment", back_populates="spacecraft")
    observations = relationship("Observation", back_populates="spacecraft")
    
    def to_dict(self):
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "norad_id": self.norad_id,
            "object_type": self.object_type,
            "owner": self.owner,
            "launch_date": self.launch_date.isoformat() if self.launch_date else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

class CCDMIndicator(Base):
    """Model representing a CCDM indicator for a spacecraft."""
    __tablename__ = "ccdm_indicators"
    
    id = Column(Integer, primary_key=True)
    spacecraft_id = Column(Integer, ForeignKey("spacecraft.id"), nullable=False)
    
    # Indicator data
    conjunction_type = Column(String(50), nullable=True)
    relative_velocity = Column(Float, nullable=True)
    miss_distance = Column(Float, nullable=True)
    time_to_closest_approach = Column(Float, nullable=True)
    probability_of_collision = Column(Float, nullable=True)
    
    # Optional detailed data
    details = Column(JSON, nullable=True)
    
    # Timestamps
    timestamp = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    spacecraft = relationship("Spacecraft", back_populates="indicators")
    
    def to_dict(self):
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "spacecraft_id": self.spacecraft_id,
            "conjunction_type": self.conjunction_type,
            "relative_velocity": self.relative_velocity,
            "miss_distance": self.miss_distance,
            "time_to_closest_approach": self.time_to_closest_approach,
            "probability_of_collision": self.probability_of_collision,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "created_at": self.created_at.isoformat()
        }

class CCDMAssessment(Base):
    """Model representing a CCDM assessment for a spacecraft."""
    __tablename__ = "ccdm_assessments"
    
    id = Column(Integer, primary_key=True)
    spacecraft_id = Column(Integer, ForeignKey("spacecraft.id"), nullable=False)
    
    # Assessment data
    assessment_type = Column(String(50), nullable=False)
    threat_level = Column(Enum(ThreatLevel), nullable=False)
    confidence_level = Column(Float, nullable=False)
    summary = Column(Text, nullable=True)
    
    # Detailed results and recommendations
    results = Column(JSON, nullable=True)
    recommendations = Column(JSON, nullable=True)
    
    # Timestamps
    timestamp = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    spacecraft = relationship("Spacecraft", back_populates="assessments")
    
    def to_dict(self):
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "spacecraft_id": self.spacecraft_id,
            "assessment_type": self.assessment_type,
            "threat_level": self.threat_level.value,
            "confidence_level": self.confidence_level,
            "summary": self.summary,
            "results": self.results,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp.isoformat(),
            "created_at": self.created_at.isoformat()
        }

class Observation(Base):
    """Model representing an observation of a spacecraft."""
    __tablename__ = "observations"
    
    id = Column(Integer, primary_key=True)
    spacecraft_id = Column(Integer, ForeignKey("spacecraft.id"), nullable=False)
    
    # Observation data
    sensor_id = Column(String(100), nullable=False)
    observation_type = Column(String(50), nullable=False)
    data = Column(JSON, nullable=True)
    
    # Timestamps
    timestamp = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    spacecraft = relationship("Spacecraft", back_populates="observations")
    
    def to_dict(self):
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "spacecraft_id": self.spacecraft_id,
            "sensor_id": self.sensor_id,
            "observation_type": self.observation_type,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "created_at": self.created_at.isoformat()
        } 