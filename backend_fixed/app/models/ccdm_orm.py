from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean, JSON, Enum
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

from app.db.base_class import Base
from app.models.ccdm import ThreatLevel, PropulsionType


class CCDMAnalysisORM(Base):
    """ORM model for CCDM object analysis records."""
    __tablename__ = "ccdm_analysis"

    id = Column(Integer, primary_key=True, index=True)
    norad_id = Column(Integer, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    summary = Column(String)
    meta_data = Column(JSON, nullable=True)

    # Relationships
    results = relationship("AnalysisResultORM", back_populates="analysis", cascade="all, delete-orphan")


class ThreatAssessmentORM(Base):
    """ORM model for threat assessments."""
    __tablename__ = "threat_assessments"

    id = Column(Integer, primary_key=True, index=True)
    norad_id = Column(Integer, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    overall_threat = Column(String)
    confidence = Column(Float)
    threat_components = Column(JSON, nullable=True)
    recommendations = Column(JSON, nullable=True)  # Stored as JSON array
    meta_data = Column(JSON, nullable=True)


class AnalysisResultORM(Base):
    """ORM model for individual analysis results."""
    __tablename__ = "analysis_results"
    
    id = Column(Integer, primary_key=True, index=True)
    analysis_id = Column(Integer, ForeignKey("ccdm_analysis.id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    confidence = Column(Float)
    threat_level = Column(String)
    details = Column(JSON, nullable=True)

    # Relationships
    analysis = relationship("CCDMAnalysisORM", back_populates="results")


class HistoricalAnalysisORM(Base):
    """ORM model for historical analyses."""
    __tablename__ = "historical_analyses"
    
    id = Column(Integer, primary_key=True, index=True)
    norad_id = Column(Integer, index=True)
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    trend_summary = Column(String)
    meta_data = Column(JSON, nullable=True)

    # Relationships
    analysis_points = relationship("HistoricalAnalysisPointORM", back_populates="historical_analysis", cascade="all, delete-orphan")


class HistoricalAnalysisPointORM(Base):
    """ORM model for individual historical analysis points."""
    __tablename__ = "historical_analysis_points"
    
    id = Column(Integer, primary_key=True, index=True)
    historical_analysis_id = Column(Integer, ForeignKey("historical_analyses.id"))
    timestamp = Column(DateTime)
    confidence = Column(Float)
    threat_level = Column(String)
    details = Column(JSON, nullable=True)

    # Relationships
    historical_analysis = relationship("HistoricalAnalysisORM", back_populates="analysis_points")


class ShapeChangeORM(Base):
    """ORM model for shape change detections."""
    __tablename__ = "shape_changes"
    
    id = Column(Integer, primary_key=True, index=True)
    norad_id = Column(Integer, index=True)
    summary = Column(String)
    meta_data = Column(JSON, nullable=True)

    # Relationships
    detections = relationship("ShapeChangeDetectionORM", back_populates="shape_change", cascade="all, delete-orphan")


class ShapeChangeDetectionORM(Base):
    """ORM model for individual shape change detections."""
    __tablename__ = "shape_change_detections"
    
    id = Column(Integer, primary_key=True, index=True)
    shape_change_id = Column(Integer, ForeignKey("shape_changes.id"))
    timestamp = Column(DateTime)
    confidence = Column(Float)
    description = Column(String)
    before_shape = Column(String)
    after_shape = Column(String)
    significance = Column(Float)

    # Relationships
    shape_change = relationship("ShapeChangeORM", back_populates="detections") 