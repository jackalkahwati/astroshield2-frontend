from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.db.session import get_db
import logging

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/health",
    tags=["health"],
)

@router.get("")
async def health_check(db: Session = Depends(get_db)):
    """
    Health check endpoint that verifies database connectivity.
    
    Returns:
        dict: Health status information
    """
    try:
        # Check database connection
        db_status = "connected"
        try:
            # Execute a simple query to verify the connection
            db.execute(text("SELECT 1"))
            db_status = "connected"
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            db_status = "disconnected"
        
        # Verify other services as needed
        return {
            "status": "healthy",
            "services": {
                "database": db_status,
                "api": "online",
                "processing": "active"
            },
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/database")
async def database_check(db: Session = Depends(get_db)):
    """
    Check if the database connection is working.
    
    Returns:
        dict: Database status
    """
    try:
        # Execute a simple query to verify the connection
        result = db.execute(text("SELECT 1")).fetchone()
        
        # Check if CCDM tables exist
        ccdm_tables = [
            "ccdm_analyses",
            "threat_assessments",
            "analysis_results",
            "historical_analyses",
            "historical_analysis_points",
            "shape_changes",
            "shape_change_detections"
        ]
        
        existing_tables = []
        for table in ccdm_tables:
            try:
                count = db.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                existing_tables.append({"name": table, "record_count": count})
            except Exception:
                # Table doesn't exist or can't be queried
                pass
        
        return {
            "status": "connected",
            "type": "PostgreSQL" if "postgresql" in db.bind.dialect.name else db.bind.dialect.name,
            "tables": existing_tables
        }
    except Exception as e:
        logger.error(f"Database check failed: {str(e)}")
        return {
            "status": "disconnected",
            "error": str(e)
        } 