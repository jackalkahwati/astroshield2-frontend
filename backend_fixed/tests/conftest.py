import pytest
from unittest.mock import MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
import os
import sys

# Add the parent directory to sys.path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.base import Base
from app.db.session import get_db
from app.models.ccdm_orm import (
    CCDMAnalysisORM, 
    ThreatAssessmentORM,
    AnalysisResultORM, 
    HistoricalAnalysisORM,
    HistoricalAnalysisPointORM,
    ShapeChangeORM,
    ShapeChangeDetectionORM
)
from app.main import app


# Set up in-memory SQLite database for testing
@pytest.fixture(scope="session")
def test_db_engine():
    """Create an in-memory SQLite database engine for testing"""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool
    )
    Base.metadata.create_all(bind=engine)
    return engine


@pytest.fixture(scope="function")
def db_session(test_db_engine):
    """Create a new database session for a test"""
    TestingSessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=test_db_engine
    )
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture(scope="function")
def override_get_db(db_session):
    """Override the get_db dependency"""
    def _get_test_db():
        try:
            yield db_session
        finally:
            pass
    return _get_test_db


@pytest.fixture(scope="function")
def client(override_get_db):
    """Create a test client with the test database"""
    from fastapi.testclient import TestClient
    
    # Override the get_db dependency
    app.dependency_overrides[get_db] = override_get_db
    
    # Create test client
    test_client = TestClient(app)
    
    yield test_client
    
    # Clean up
    app.dependency_overrides = {}


@pytest.fixture(scope="function")
def sample_data(db_session):
    """Create sample test data in the database"""
    from datetime import datetime, timedelta
    import random
    from app.models.ccdm import ThreatLevel
    
    # Sample satellite NORAD IDs
    satellites = [
        {"norad_id": 25544, "name": "ISS"},
        {"norad_id": 33591, "name": "Hubble Space Telescope"},
        {"norad_id": 43013, "name": "NOAA-20"},
        {"norad_id": 48274, "name": "Starlink-1234"},
        {"norad_id": 27424, "name": "XMM-Newton"}
    ]
    
    for satellite in satellites:
        norad_id = satellite["norad_id"]
        
        # Create object analysis
        analysis = CCDMAnalysisORM(
            norad_id=norad_id,
            timestamp=datetime.utcnow(),
            summary=f"Analysis of {satellite['name']} (NORAD ID: {norad_id}) completed successfully.",
            metadata={"satellite_name": satellite["name"]}
        )
        db_session.add(analysis)
        db_session.flush()  # Get ID
        
        # Create analysis results
        for i in range(3):
            result = AnalysisResultORM(
                analysis_id=analysis.id,
                timestamp=datetime.utcnow() - timedelta(hours=i),
                confidence=random.uniform(0.7, 0.95),
                threat_level=random.choice(list(ThreatLevel)),
                details={"component": f"subsystem-{i+1}", "anomaly_score": random.uniform(0, 1)}
            )
            db_session.add(result)
        
        # Create threat assessment
        assessment = ThreatAssessmentORM(
            norad_id=norad_id,
            timestamp=datetime.utcnow(),
            overall_threat=random.choice(list(ThreatLevel)),
            confidence=random.uniform(0.7, 0.95),
            threat_components={
                "collision": random.choice(list(ThreatLevel)).__str__(),
                "debris": random.choice(list(ThreatLevel)).__str__(),
                "maneuver": random.choice(list(ThreatLevel)).__str__(),
                "radiation": random.choice(list(ThreatLevel)).__str__()
            },
            recommendations=["Monitor for changes", "Assess trajectory", "Verify telemetry"],
            metadata={"satellite_name": satellite["name"]}
        )
        db_session.add(assessment)
        
        # Create historical analysis
        historical = HistoricalAnalysisORM(
            norad_id=norad_id,
            start_date=datetime.utcnow() - timedelta(days=7),
            end_date=datetime.utcnow(),
            trend_summary=f"No significant anomalies detected for {satellite['name']} over the past week.",
            metadata={"satellite_name": satellite["name"]}
        )
        db_session.add(historical)
        db_session.flush()  # Get ID
        
        # Create historical analysis points
        for i in range(7):
            point = HistoricalAnalysisPointORM(
                historical_analysis_id=historical.id,
                timestamp=datetime.utcnow() - timedelta(days=i),
                confidence=random.uniform(0.7, 0.95),
                threat_level=random.choice(list(ThreatLevel)),
                details={"day": i, "anomaly_count": random.randint(0, 5)}
            )
            db_session.add(point)
        
        # Create shape change record
        shape_change = ShapeChangeORM(
            norad_id=norad_id,
            summary=f"Minor changes detected in {satellite['name']} profile over time.",
            metadata={"satellite_name": satellite["name"]}
        )
        db_session.add(shape_change)
        db_session.flush()  # Get ID
        
        # Create shape change detections
        for i in range(2):
            detection = ShapeChangeDetectionORM(
                shape_change_id=shape_change.id,
                timestamp=datetime.utcnow() - timedelta(days=i*3),
                confidence=random.uniform(0.6, 0.9),
                description=f"Change in {random.choice(['solar panel', 'antenna', 'module'])} configuration",
                before_shape="standard_configuration",
                after_shape="modified_configuration",
                significance=random.uniform(0.1, 0.5)
            )
            db_session.add(detection)
    
    db_session.commit()
    
    return satellites 