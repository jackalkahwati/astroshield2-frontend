#!/usr/bin/env python
"""
Database initialization script for AstroShield platform.

This script initializes the PostgreSQL database by:
1. Creating all database tables
2. Adding default users
3. Adding sample CCDM data for testing
"""

import sys
import os
import logging
from datetime import datetime, timedelta
import random

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.session import engine, SessionLocal
from app.db.base import Base
from app.core.security import get_password_hash
from app.models.user import UserORM
from app.models.ccdm_orm import (
    CCDMAnalysisORM, 
    ThreatAssessmentORM,
    AnalysisResultORM, 
    HistoricalAnalysisORM,
    HistoricalAnalysisPointORM,
    ShapeChangeORM,
    ShapeChangeDetectionORM
)
from app.models.ccdm import ThreatLevel, PropulsionType

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def init_db():
    """Initialize the database with tables and sample data."""
    logger.info("Creating database tables...")
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    # Create a database session
    db = SessionLocal()
    
    try:
        # Create default users
        create_default_users(db)
        
        # Create sample CCDM data
        create_sample_ccdm_data(db)
        
        logger.info("Database initialization completed successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
    finally:
        db.close()

def create_default_users(db):
    """Create default users if they don't already exist."""
    logger.info("Creating default users...")
    
    # Check if admin user exists
    admin_user = db.query(UserORM).filter(UserORM.email == "admin@example.com").first()
    if not admin_user:
        admin_user = UserORM(
            email="admin@example.com",
            hashed_password=get_password_hash("admin123"),  # Insecure, only for development
            is_active=True,
            is_superuser=True,
            full_name="Admin User"
        )
        db.add(admin_user)
        logger.info("Created admin user")
    
    # Check if regular user exists
    user = db.query(UserORM).filter(UserORM.email == "user@example.com").first()
    if not user:
        user = UserORM(
            email="user@example.com",
            hashed_password=get_password_hash("user123"),  # Insecure, only for development
            is_active=True,
            is_superuser=False,
            full_name="Regular User"
        )
        db.add(user)
        logger.info("Created regular user")
    
    db.commit()

def create_sample_ccdm_data(db):
    """Create sample CCDM data for testing."""
    logger.info("Creating sample CCDM data...")
    
    # Sample satellite NORAD IDs
    satellites = [
        {"norad_id": 25544, "name": "ISS"},
        {"norad_id": 48274, "name": "Starlink-1234"},
        {"norad_id": 43013, "name": "NOAA-20"},
        {"norad_id": 33591, "name": "Hubble Space Telescope"},
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
        db.add(analysis)
        db.flush()  # Get ID
        
        # Create analysis results
        for i in range(3):
            result = AnalysisResultORM(
                analysis_id=analysis.id,
                timestamp=datetime.utcnow() - timedelta(hours=i),
                confidence=random.uniform(0.7, 0.95),
                threat_level=random.choice(list(ThreatLevel)),
                details={"component": f"subsystem-{i+1}", "anomaly_score": random.uniform(0, 1)}
            )
            db.add(result)
        
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
        db.add(assessment)
        
        # Create historical analysis
        historical = HistoricalAnalysisORM(
            norad_id=norad_id,
            start_date=datetime.utcnow() - timedelta(days=7),
            end_date=datetime.utcnow(),
            trend_summary=f"No significant anomalies detected for {satellite['name']} over the past week.",
            metadata={"satellite_name": satellite["name"]}
        )
        db.add(historical)
        db.flush()  # Get ID
        
        # Create historical analysis points
        for i in range(7):
            point = HistoricalAnalysisPointORM(
                historical_analysis_id=historical.id,
                timestamp=datetime.utcnow() - timedelta(days=i),
                confidence=random.uniform(0.7, 0.95),
                threat_level=random.choice(list(ThreatLevel)),
                details={"day": i, "anomaly_count": random.randint(0, 5)}
            )
            db.add(point)
        
        # Create shape change record
        shape_change = ShapeChangeORM(
            norad_id=norad_id,
            summary=f"Minor changes detected in {satellite['name']} profile over time.",
            metadata={"satellite_name": satellite["name"]}
        )
        db.add(shape_change)
        db.flush()  # Get ID
        
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
            db.add(detection)
    
    db.commit()
    logger.info(f"Created sample data for {len(satellites)} satellites")

if __name__ == "__main__":
    logger.info("Starting database initialization...")
    init_db()
    logger.info("Database initialization completed") 