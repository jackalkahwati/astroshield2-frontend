"""Database module for SQLAlchemy configuration and session management."""

import os
import logging
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Get database connection string from environment variable, or use SQLite as fallback
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./astroshield.db"
)

# Echo SQL queries in development mode
ECHO_SQL = os.getenv("ECHO_SQL", "False").lower() in ("true", "1", "t")

# Create SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    echo=ECHO_SQL,
    pool_pre_ping=True,  # Check connection before using from pool
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

# Naming convention for constraints
naming_convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s"
}

Base.metadata = MetaData(naming_convention=naming_convention)

def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides a database session.
    
    Yields:
        Database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def db_session() -> Generator[Session, None, None]:
    """
    Context manager for database sessions outside of FastAPI.
    
    Yields:
        Database session
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        db.rollback()
        logger.error(f"Database error: {str(e)}")
        raise
    finally:
        db.close()

def init_db() -> None:
    """Initialize database, creating tables if they don't exist."""
    try:
        # Import all models to ensure they're registered with Base.metadata
        from backend.models.ccdm import CCDMIndicator, Spacecraft
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization error: {str(e)}")
        raise

def create_sample_data(db: Session) -> None:
    """
    Create sample data for development/testing.
    
    Args:
        db: Database session
    """
    from backend.models.ccdm import CCDMIndicator, Spacecraft
    from datetime import datetime, timedelta
    
    try:
        # Check if we already have data
        if db.query(Spacecraft).first():
            logger.info("Sample data already exists, skipping creation")
            return
            
        # Create sample spacecraft
        spacecraft_data = [
            {"name": "ISS (ZARYA)"},
            {"name": "STARLINK-1234"},
            {"name": "COSMOS 2542"},
            {"name": "GPS IIR-21 (USA 206)"},
            {"name": "NOAA 19"}
        ]
        
        spacecraft_objects = []
        for data in spacecraft_data:
            spacecraft = Spacecraft(**data)
            db.add(spacecraft)
            spacecraft_objects.append(spacecraft)
        
        db.commit()
        
        # Create sample CCDM indicators
        now = datetime.utcnow()
        
        indicator_data = []
        for i, spacecraft in enumerate(spacecraft_objects):
            # Create several indicators for each spacecraft
            for days_ago in range(10):
                indicator_data.append({
                    "spacecraft_id": spacecraft.id,
                    "conjunction_type": "CLOSE_APPROACH" if days_ago % 3 == 0 else "MANEUVER_DETECTED",
                    "relative_velocity": 0.5 + (i * 0.25) + (days_ago * 0.1), 
                    "miss_distance": 10.0 + (i * 5.0) - (days_ago * 0.5),
                    "time_to_closest_approach": 24.0 - (days_ago * 0.5),
                    "probability_of_collision": 0.001 + (days_ago * 0.0001),
                    "timestamp": now - timedelta(days=days_ago, hours=i)
                })
        
        for data in indicator_data:
            indicator = CCDMIndicator(**data)
            db.add(indicator)
            
        db.commit()
        logger.info(f"Created {len(spacecraft_data)} spacecraft and {len(indicator_data)} indicators")
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating sample data: {str(e)}")
        raise 