"""Database initialization script"""

import logging
from sqlalchemy.orm import Session
import os # Import os for environment variables

from app.db.base import Base
from app.db.session import engine
from app.core.security import get_password_hash
from app.models.user import UserORM
from app.models.event_store import EventORM, EventProcessingLogORM, EventMetricsORM

# Import common logging utility
from src.asttroshield.common.logging_utils import get_logger

# Get logger using common utility (configuration is done in main.py)
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
logger = get_logger(__name__)


def init_db(db: Session) -> None:
    """Initialize the database with required tables and initial data."""
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    # Create default users if they don't exist
    create_default_users(db)
    
    logger.info("Database initialized successfully")


def create_default_users(db: Session) -> None:
    """Create default users if they don't exist."""
    
    # Get default passwords from environment or use dev defaults
    # WARNING: These defaults are insecure and only for local development!
    test_user_password = os.getenv("DEFAULT_TEST_USER_PASSWORD", "password")
    admin_user_password = os.getenv("DEFAULT_ADMIN_USER_PASSWORD", "admin")
    
    # Check if test user exists
    test_user = db.query(UserORM).filter(UserORM.email == "test@example.com").first()
    if not test_user:
        logger.info("Creating test user")
        if test_user_password == "password":
             logger.warning("Using insecure default password for test user! Set DEFAULT_TEST_USER_PASSWORD env var.")
        test_user = UserORM(
            email="test@example.com",
            hashed_password=get_password_hash(test_user_password),
            is_active=True,
            is_superuser=False,
            full_name="Test User"
        )
        db.add(test_user)
    
    # Check if admin user exists
    admin_user = db.query(UserORM).filter(UserORM.email == "admin@example.com").first()
    if not admin_user:
        logger.info("Creating admin user")
        if admin_user_password == "admin":
             logger.warning("Using insecure default password for admin user! Set DEFAULT_ADMIN_USER_PASSWORD env var.")
        admin_user = UserORM(
            email="admin@example.com",
            hashed_password=get_password_hash(admin_user_password),
            is_active=True,
            is_superuser=True,
            full_name="Admin User"
        )
        db.add(admin_user)
    
    db.commit()