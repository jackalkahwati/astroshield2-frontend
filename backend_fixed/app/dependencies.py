from app.services.ccdm import CCDMService
from app.db.session import get_db
from sqlalchemy.orm import Session

# Singleton instance to avoid creating multiple instances
_ccdm_service_instance = None

async def get_ccdm_service(db: Session = None) -> CCDMService:
    """
    Factory function to get a CCDMService instance.
    Uses a singleton pattern to avoid recreating the service for each request.
    
    Args:
        db: SQLAlchemy database session
    
    Returns:
        A CCDMService instance
    """
    global _ccdm_service_instance
    
    if _ccdm_service_instance is None:
        _ccdm_service_instance = CCDMService(db=db)
    else:
        # Update database session if needed
        _ccdm_service_instance.db = db
    
    return _ccdm_service_instance 