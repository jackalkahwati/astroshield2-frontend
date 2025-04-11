from fastapi import APIRouter, HTTPException, Depends, Query, status
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel
from app.services.ccdm import CCDMService
from app.dependencies import get_ccdm_service
from app.models.ccdm import (
    ObjectAnalysisRequest,
    ObjectAnalysisResponse,
    ShapeChangeResponse,
    ThermalSignatureResponse,
    PropulsiveCapabilityResponse,
    HistoricalAnalysis,
    CCDMAssessment,
    AnomalyDetection
)
from app.core.security import get_current_user, check_roles
from app.models.user import User
from app.exceptions import ObjectNotFoundError, AnalysisError, InvalidInputError, ServiceError

router = APIRouter()

async def handle_service_error(exc: ServiceError):
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=exc.detail)

async def handle_not_found_error(exc: ObjectNotFoundError):
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=exc.detail)

async def handle_invalid_input_error(exc: InvalidInputError):
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=exc.detail)

async def handle_analysis_error(exc: AnalysisError):
    raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=exc.detail)

@router.post("/analyze", response_model=ObjectAnalysisResponse)
async def analyze_object(
    request: ObjectAnalysisRequest,
    current_user: User = Depends(check_roles(["active"])),
    ccdm_service: CCDMService = Depends(get_ccdm_service)
) -> ObjectAnalysisResponse:
    """Analyze a space object using CCDM techniques"""
    try:
        result = await ccdm_service.analyze_object(request.object_id, request.metadata)
        return result
    except ObjectNotFoundError as e:
        await handle_not_found_error(e)
    except InvalidInputError as e:
        await handle_invalid_input_error(e)
    except AnalysisError as e:
        await handle_analysis_error(e)
    except ServiceError as e:
        await handle_service_error(e)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected internal server error: {e}")

@router.post("/detect_shape_changes", response_model=ShapeChangeResponse)
async def detect_shape_changes(
    object_id: str,
    start_time: datetime,
    end_time: datetime,
    current_user: User = Depends(check_roles(["active"])),
    ccdm_service: CCDMService = Depends(get_ccdm_service)
):
    """Detect changes in object shape over time"""
    try:
        result = await ccdm_service.detect_shape_changes(object_id, start_time, end_time)
        return result
    except ObjectNotFoundError as e:
        await handle_not_found_error(e)
    except InvalidInputError as e:
        await handle_invalid_input_error(e)
    except AnalysisError as e:
        await handle_analysis_error(e)
    except ServiceError as e:
        await handle_service_error(e)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected internal server error: {e}")

@router.post("/assess_thermal_signature", response_model=ThermalSignatureResponse)
async def assess_thermal_signature(
    object_id: str,
    timestamp: datetime,
    current_user: User = Depends(check_roles(["active"])),
    ccdm_service: CCDMService = Depends(get_ccdm_service)
):
    """Assess thermal signature of an object"""
    try:
        result = await ccdm_service.assess_thermal_signature(object_id, timestamp)
        return result
    except ObjectNotFoundError as e:
        await handle_not_found_error(e)
    except InvalidInputError as e:
        await handle_invalid_input_error(e)
    except AnalysisError as e:
        await handle_analysis_error(e)
    except ServiceError as e:
        await handle_service_error(e)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected internal server error: {e}")

@router.post("/evaluate_propulsive_capabilities", response_model=PropulsiveCapabilityResponse)
async def evaluate_propulsive_capabilities(
    object_id: str,
    analysis_period: int,
    current_user: User = Depends(check_roles(["active", "admin"])),
    ccdm_service: CCDMService = Depends(get_ccdm_service)
):
    """Evaluate object's propulsive capabilities"""
    try:
        result = await ccdm_service.evaluate_propulsive_capabilities(object_id, analysis_period)
        return result
    except ObjectNotFoundError as e:
        await handle_not_found_error(e)
    except InvalidInputError as e:
        await handle_invalid_input_error(e)
    except AnalysisError as e:
        await handle_analysis_error(e)
    except ServiceError as e:
        await handle_service_error(e)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected internal server error: {e}")

@router.get("/history/{object_id}", response_model=List[HistoricalAnalysis])
async def retrieve_historical_analysis(
    object_id: str,
    current_user: User = Depends(check_roles(["active", "admin"])),
    ccdm_service: CCDMService = Depends(get_ccdm_service)
) -> List[HistoricalAnalysis]:
    """Retrieve historical CCDM analysis for an object"""
    try:
        return await ccdm_service.get_historical_analysis(object_id)
    except ObjectNotFoundError as e:
        await handle_not_found_error(e)
    except ServiceError as e:
        await handle_service_error(e)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected internal server error: {e}")

@router.get("/assessment/{object_id}", response_model=CCDMAssessment)
async def get_ccdm_assessment(
    object_id: str,
    current_user: User = Depends(check_roles(["active"])),
    ccdm_service: CCDMService = Depends(get_ccdm_service)
) -> CCDMAssessment:
    """Get a comprehensive CCDM assessment for an object"""
    try:
        return await ccdm_service.get_ccdm_assessment(object_id)
    except ObjectNotFoundError as e:
        await handle_not_found_error(e)
    except AnalysisError as e:
        await handle_analysis_error(e)
    except ServiceError as e:
        await handle_service_error(e)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected internal server error: {e}")

@router.get("/anomalies/{object_id}", response_model=List[AnomalyDetection])
async def get_anomaly_detections(
    object_id: str,
    days: int = Query(30, description="Number of days to look back for anomalies"),
    current_user: User = Depends(check_roles(["active"])),
    ccdm_service: CCDMService = Depends(get_ccdm_service)
) -> List[AnomalyDetection]:
    """Get anomaly detections for an object over a period"""
    try:
        return await ccdm_service.get_anomaly_detections(object_id, days)
    except ObjectNotFoundError as e:
        await handle_not_found_error(e)
    except ServiceError as e:
        await handle_service_error(e)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected internal server error: {e}")

@router.get("/status")
async def get_ccdm_status() -> dict:
    """Get CCDM system status"""
    return {
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
        "active_analyses": 3,
        "system_health": {
            "cpu_usage": 32.5,
            "memory_usage": 45.8,
            "storage_usage": 28.3
        },
        "queue": {
            "pending_analyses": 2,
            "estimated_completion_time": (datetime.utcnow()).isoformat()
        }
    }