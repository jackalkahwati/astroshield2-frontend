"""
CCDM routes
"""
from typing import List
from fastapi import APIRouter, HTTPException

from app.common.logging import logger
from app.ccdm.models import ConjunctionEvent, ConjunctionCreateRequest, ConjunctionFilterRequest
from app.ccdm.service import get_all_conjunctions, get_conjunction_by_id, create_conjunction, filter_conjunctions

# Create router for CCDM endpoints
router = APIRouter(prefix="/api/v1/conjunctions", tags=["conjunctions"])

@router.get("", response_model=List[ConjunctionEvent])
async def get_conjunctions():
    """Get all conjunction events"""
    return get_all_conjunctions()

@router.post("", response_model=ConjunctionEvent)
async def create_conjunction_event(request: ConjunctionCreateRequest):
    """Create a new conjunction event"""
    return create_conjunction(request)

@router.get("/{conjunction_id}", response_model=ConjunctionEvent)
async def get_conjunction(conjunction_id: str):
    """Get a specific conjunction by ID"""
    conjunction = get_conjunction_by_id(conjunction_id)
    if not conjunction:
        raise HTTPException(status_code=404, detail=f"Conjunction with ID {conjunction_id} not found")
    return conjunction

@router.post("/filter", response_model=List[ConjunctionEvent])
async def filter_conjunction_events(filter_request: ConjunctionFilterRequest):
    """Filter conjunction events based on criteria"""
    return filter_conjunctions(filter_request) 