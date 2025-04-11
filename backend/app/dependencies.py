from app.services.ccdm import CCDMService

# Dependency function to get the CCDM service instance
def get_ccdm_service() -> CCDMService:
    # In a real application, this might involve more complex setup,
    # like getting database connections or other resources.
    # For now, it simply returns a new instance.
    return CCDMService() 