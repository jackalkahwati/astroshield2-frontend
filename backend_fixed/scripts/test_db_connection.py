#!/usr/bin/env python
"""
Test database connection script for AstroShield platform.

This script tests the connection to the PostgreSQL database and lists tables.
"""

import sys
import os
import logging
from datetime import datetime
from sqlalchemy import text

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.session import engine, SessionLocal

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_db_connection():
    """Test connection to the PostgreSQL database."""
    logger.info("Testing database connection...")
    
    # Create a database session
    db = SessionLocal()
    
    try:
        # Test connection with a simple query
        result = db.execute(text("SELECT 1")).scalar()
        logger.info(f"Connection test result: {result}")
        
        # Get database version
        version = db.execute(text("SELECT version()")).scalar()
        logger.info(f"Database version: {version}")
        
        # List all tables
        logger.info("Listing tables:")
        tables = db.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)).fetchall()
        
        if not tables:
            logger.warning("No tables found in the database.")
        else:
            for i, table in enumerate(tables, 1):
                table_name = table[0]
                count = db.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()
                logger.info(f"{i}. {table_name} - {count} records")
        
        # Test specific CCDM tables
        ccdm_tables = [
            "ccdm_analyses", 
            "threat_assessments", 
            "analysis_results",
            "historical_analyses", 
            "historical_analysis_points",
            "shape_changes", 
            "shape_change_detections"
        ]
        
        logger.info("\nTesting CCDM tables:")
        for table in ccdm_tables:
            try:
                count = db.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                logger.info(f"  ✓ {table} - {count} records")
            except Exception as e:
                logger.error(f"  ✗ {table} - Error: {str(e)}")
        
        logger.info("\nDatabase connection test completed successfully.")
        return True
    except Exception as e:
        logger.error(f"Error testing database connection: {str(e)}")
        return False
    finally:
        db.close()

if __name__ == "__main__":
    logger.info("Starting database connection test...")
    success = test_db_connection()
    if success:
        logger.info("Database connection is working correctly.")
        sys.exit(0)
    else:
        logger.error("Database connection test failed.")
        sys.exit(1) 