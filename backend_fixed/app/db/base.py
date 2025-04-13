# Import all models here to ensure they are registered with SQLAlchemy
from sqlalchemy.ext.declarative import declarative_base
from app.db.base_class import Base
from app.models.user import UserORM
from app.models.trajectory import TrajectoryORM, TrajectoryComparisonORM
from app.models.event_store import EventORM, EventProcessingLogORM, EventMetricsORM
from app.models.ccdm_orm import (
    CCDMAnalysisORM,
    ThreatAssessmentORM,
    AnalysisResultORM,
    HistoricalAnalysisORM,
    HistoricalAnalysisPointORM,
    ShapeChangeORM,
    ShapeChangeDetectionORM
)

Base = declarative_base()
