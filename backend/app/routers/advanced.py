from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from ..models.indicator_models import (
    SystemInteraction, EclipsePeriod, TrackingData,
    UNRegistryEntry, OrbitOccupancyData, StimulationEvent,
    LaunchTrackingData
)
from ..analysis.advanced_indicators import (
    StimulationEvaluator, LaunchTrackingEvaluator,
    EclipseTrackingEvaluator, OrbitOccupancyEvaluator,
    UNRegistryEvaluator
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize evaluators
stimulation_evaluator = StimulationEvaluator()
launch_tracking_evaluator = LaunchTrackingEvaluator()
eclipse_tracking_evaluator = EclipseTrackingEvaluator()
orbit_occupancy_evaluator = OrbitOccupancyEvaluator()
un_registry_evaluator = UNRegistryEvaluator()

class RegistryData(BaseModel):
    registry_data: Dict[str, Any]
    object_data: Optional[Dict[str, Any]] = None

@router.post("/registry/verify/{spacecraft_id}")
async def verify_un_registry(spacecraft_id: str, data: RegistryData):
    """Verify UN registry status for a spacecraft"""
    try:
        registry_data = UNRegistryEntry(**data.registry_data)
        
        # Analyze registry status
        indicators = un_registry_evaluator.analyze_un_registry(
            {'spacecraft_id': spacecraft_id, **(data.object_data or {})},
            {'entries': [registry_data.dict()]}
        )
        
        return {
            'status': 'success',
            'spacecraft_id': spacecraft_id,
            'indicators': [i.dict() for i in indicators]
        }
    except Exception as e:
        logger.error(f"UN registry verification error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add other advanced endpoints here as needed 