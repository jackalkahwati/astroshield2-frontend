"""Database models for event storage."""

from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, 
    ForeignKey, JSON, Enum, Table
)
from sqlalchemy.orm import relationship

from app.db.base_class import Base
from app.models.events import EventType, EventStatus, ThreatLevel

class EventORM(Base):
    """ORM model for events."""
    __tablename__ = "events"

    id = Column(String, primary_key=True)
    event_type = Column(Enum(EventType), nullable=False, index=True)
    object_id = Column(String, nullable=False, index=True)
    status = Column(Enum(EventStatus), nullable=False, index=True)
    creation_time = Column(DateTime, nullable=False, index=True)
    update_time = Column(DateTime, nullable=False)
    detection_data = Column(JSON, nullable=False)
    processing_steps = Column(JSON, nullable=False, default=[])
    hostility_assessment = Column(JSON, nullable=True)
    threat_level = Column(Enum(ThreatLevel), nullable=True)
    coa_recommendation = Column(JSON, nullable=True)

    # Add relationship to processing logs if needed

class EventProcessingLogORM(Base):
    """ORM model for detailed event processing logs."""
    __tablename__ = "event_processing_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String, ForeignKey("events.id"), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    step_name = Column(String, nullable=False)
    status = Column(String, nullable=False)
    input_data = Column(JSON, nullable=True)
    output_data = Column(JSON, nullable=True)
    error = Column(String, nullable=True)
    processing_time_ms = Column(Float, nullable=True)
    
    # Relationship to the event
    event = relationship("EventORM", backref="processing_logs")

class EventMetricsORM(Base):
    """ORM model for event processing metrics."""
    __tablename__ = "event_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String, ForeignKey("events.id"), nullable=False, index=True)
    total_processing_time_ms = Column(Float, nullable=True)
    processing_steps_count = Column(Integer, nullable=False, default=0)
    data_processing_volume_bytes = Column(Integer, nullable=True)
    confidence_score = Column(Float, nullable=True)
    accuracy_score = Column(Float, nullable=True)
    false_positive_probability = Column(Float, nullable=True)
    
    # Relationship to the event
    event = relationship("EventORM", backref="metrics")