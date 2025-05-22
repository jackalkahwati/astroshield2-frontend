from typing import Generator
import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from app.core.config import settings

# Get database URL from environment variable or use SQLite by default
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./astroshield.db")

# Create SQLAlchemy engine
engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True, connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {})

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db() -> Session: # Added return type hint
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
