from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel
from app.services.ccdm import CCDMService
from app.models.ccdm import (
    ObjectAnalysisRequest,
    ObjectAnalysisResponse,
    ShapeChangeResponse,
    ThermalSignatureResponse,
    PropulsiveCapabilityResponse,
    HistoricalAnalysis
)
from app.core.security import get_current_user, check_roles
from app.models.user import User

router = APIRouter()
ccdm_service = CCDMService()

@router.post("/analyze", response_model=ObjectAnalysisResponse)
async def analyze_object(
    request: ObjectAnalysisRequest,
    current_user: User = Depends(check_roles(["active"]))
) -> ObjectAnalysisResponse:
    """Analyze a space object using CCDM techniques"""
    try:
        result = await ccdm_service.analyze_object(request.object_id, request.observation_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/detect_shape_changes", response_model=ShapeChangeResponse)
async def detect_shape_changes(
    object_id: str,
    start_time: datetime,
    end_time: datetime,
    current_user: User = Depends(check_roles(["active"]))
):
    """Detect changes in object shape over time"""
    try:
        result = await ccdm_service.detect_shape_changes(object_id, start_time, end_time)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/assess_thermal_signature", response_model=ThermalSignatureResponse)
async def assess_thermal_signature(
    object_id: str,
    timestamp: datetime,
    current_user: User = Depends(check_roles(["active"]))
):
    """Assess thermal signature of an object"""
    try:
        result = await ccdm_service.assess_thermal_signature(object_id, timestamp)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/evaluate_propulsive_capabilities", response_model=PropulsiveCapabilityResponse)
async def evaluate_propulsive_capabilities(
    object_id: str,
    analysis_period: int,
    current_user: User = Depends(check_roles(["active", "admin"]))
):
    """Evaluate object's propulsive capabilities"""
    try:
        result = await ccdm_service.evaluate_propulsive_capabilities(object_id, analysis_period)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/{object_id}", response_model=List[HistoricalAnalysis])
async def retrieve_historical_analysis(
    object_id: str,
    current_user: User = Depends(check_roles(["active", "admin"]))
) -> List[HistoricalAnalysis]:
    """Retrieve historical CCDM analysis for an object"""
    try:
        return await ccdm_service.get_historical_analysis(object_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_ccdm_status(
    current_user: User = Depends(check_roles())
) -> dict:
    return {
        "status": "operational",
        "timestamp": datetime.utcnow(),
        "active_analyses": 3
    } 