"""
Satellites router for the AstroShield API.
Provides endpoints for retrieving and managing satellites.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
import logging
from pydantic import BaseModel
from datetime import datetime

# Import UDL client
from app.services.udl_client import get_udl_client, StateVector, UDLServiceException

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()

class SatelliteResponse(BaseModel):
    """Response model for satellite data"""
    id: str
    name: str
    epoch: str
    position: Optional[dict] = None
    velocity: Optional[dict] = None
    status: Optional[str] = "active"
    last_update: Optional[str] = None

@router.get("/satellites", response_model=List[SatelliteResponse])
async def get_satellites(
    limit: int = Query(10, ge=1, le=100, description="Maximum number of satellites to return"),
    use_mock: bool = Query(False, description="Force use of mock data"),
):
    """
    Get a list of satellites.
    
    Retrieves satellite data from UDL with position and velocity information.
    """
    try:
        # Get UDL client
        udl_client = await get_udl_client()
        
        # Get state vectors from UDL
        if use_mock:
            # Force mock data
            state_vectors = udl_client._get_mock_state_vectors(limit)
        else:
            # Normal operation
            state_vectors = await udl_client.get_state_vectors(limit)
        
        # Convert to response model
        satellites = []
        for sv in state_vectors:
            satellite = SatelliteResponse(
                id=sv.id,
                name=sv.name,
                epoch=sv.epoch,
                position=sv.position.dict() if hasattr(sv.position, "dict") else sv.position,
                velocity=sv.velocity.dict() if hasattr(sv.velocity, "dict") else sv.velocity,
                status="active",
                last_update=datetime.utcnow().isoformat()
            )
            satellites.append(satellite)
        
        return satellites
        
    except UDLServiceException as e:
        # This will be handled by our exception handler
        raise
    except Exception as e:
        logger.error(f"Error getting satellites: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/satellites/{satellite_id}", response_model=SatelliteResponse)
async def get_satellite(satellite_id: str):
    """
    Get details for a specific satellite.
    
    Retrieves detailed information for a satellite by ID.
    """
    try:
        # Get UDL client
        udl_client = await get_udl_client()
        
        # Get state vectors from UDL
        state_vectors = await udl_client.get_state_vectors(limit=20)
        
        # Find the requested satellite
        for sv in state_vectors:
            if sv.id == satellite_id:
                return SatelliteResponse(
                    id=sv.id,
                    name=sv.name,
                    epoch=sv.epoch,
                    position=sv.position.dict() if hasattr(sv.position, "dict") else sv.position,
                    velocity=sv.velocity.dict() if hasattr(sv.velocity, "dict") else sv.velocity,
                    status="active",
                    last_update=datetime.utcnow().isoformat()
                )
        
        # If we get here, the satellite wasn't found
        from app.core.error_handling import ResourceNotFoundException
        raise ResourceNotFoundException(f"Satellite with ID {satellite_id} not found")
        
    except Exception as e:
        logger.error(f"Error getting satellite {satellite_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")