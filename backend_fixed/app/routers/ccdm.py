from fastapi import APIRouter, HTTPException, Depends, Query, status
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.services.ccdm import CCDMService, get_ccdm_service
from app.dependencies import get_ccdm_service
from app.db.session import get_db
from app.models.ccdm import (
    ObjectAnalysisRequest,
    ObjectAnalysisResponse,
    ShapeChangeResponse,
    ThermalSignatureResponse,
    PropulsiveCapabilityResponse,
    HistoricalAnalysis,
    CCDMAssessment,
    AnomalyDetection,
    ObjectThreatAssessment,
    HistoricalAnalysisRequest,
    HistoricalAnalysisResponse,
    ShapeChangeRequest,
    ThreatAssessmentRequest
)
from app.core.security import get_current_user, check_roles
from app.models.user import User
from app.exceptions import ObjectNotFoundError, AnalysisError, InvalidInputError, ServiceError
from app.core.roles import Roles
import logging

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/ccdm",
    tags=["ccdm"],
    responses={404: {"description": "Not found"}},
)

async def handle_service_error(exc: ServiceError):
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=exc.detail)

async def handle_not_found_error(exc: ObjectNotFoundError):
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=exc.detail)

async def handle_invalid_input_error(exc: InvalidInputError):
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=exc.detail)

async def handle_analysis_error(exc: AnalysisError):
    raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=exc.detail)

def get_ccdm_service() -> CCDMService:
    """Dependency to get the CCDM service."""
    return CCDMService()

@router.post("/analyze", response_model=ObjectAnalysisResponse)
def analyze_object(
    request: ObjectAnalysisRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(check_roles([Roles.viewer]))
):
    """
    Analyze a space object based on its NORAD ID
    """
    try:
        ccdm_service = CCDMService(db)
        return ccdm_service.analyze_object(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing object: {str(e)}")

@router.post("/threat-assessment", response_model=ObjectThreatAssessment)
def assess_threat(
    request: ThreatAssessmentRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(check_roles([Roles.viewer]))
):
    """
    Assess the threat level of a space object
    """
    try:
        ccdm_service = CCDMService(db)
        return ccdm_service.assess_threat(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error assessing threat: {str(e)}")

@router.post("/historical", response_model=HistoricalAnalysisResponse)
def get_historical_analysis(
    request: HistoricalAnalysisRequest,
    page: int = Query(1, ge=1, description="Page number for paginated results"),
    page_size: int = Query(50, ge=10, le=500, description="Number of data points per page"),
    db: Session = Depends(get_db),
    current_user: User = Depends(check_roles([Roles.viewer]))
):
    """
    Get historical analysis data for a space object
    
    Returns paginated results for large datasets. Default page size is 50 data points.
    Maximum page size is 500 to prevent memory issues with large result sets.
    """
    try:
        # Validate norad_id (must be a positive integer)
        if request.norad_id <= 0:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "INVALID_INPUT",
                    "message": "NORAD ID must be a positive integer",
                    "field": "norad_id",
                    "value": request.norad_id
                }
            )
            
        # Validate that dates are not in the future
        current_time = datetime.utcnow()
        if request.start_date > current_time:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "INVALID_INPUT",
                    "message": "Start date cannot be in the future",
                    "field": "start_date",
                    "value": request.start_date.isoformat()
                }
            )
            
        if request.end_date > current_time:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "INVALID_INPUT",
                    "message": "End date cannot be in the future",
                    "field": "end_date",
                    "value": request.end_date.isoformat()
                }
            )
            
        # Validate that end_date is after start_date
        if request.end_date <= request.start_date:
            raise HTTPException(
                status_code=400, 
                detail={
                    "error": "INVALID_INPUT",
                    "message": "End date must be after start date",
                    "field": "end_date",
                    "value": request.end_date.isoformat()
                }
            )
            
        # Validate that the date range isn't too large (e.g., limit to 90 days)
        max_range_days = 90
        date_range_days = (request.end_date - request.start_date).days
        if date_range_days > max_range_days:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "INVALID_INPUT",
                    "message": f"Date range too large. Maximum allowed range is {max_range_days} days",
                    "field": "date_range",
                    "value": date_range_days
                }
            )
            
        # Process the request with pagination parameters
        ccdm_service = CCDMService(db)
        return ccdm_service.get_historical_analysis(request, page, page_size)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting historical analysis: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail={
                "error": "SERVER_ERROR",
                "message": f"Error getting historical analysis: {str(e)}",
                "type": str(type(e).__name__)
            }
        )

@router.post("/shape-changes", response_model=ShapeChangeResponse)
def detect_shape_changes(
    request: ShapeChangeRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(check_roles([Roles.viewer]))
):
    """
    Detect shape changes for a space object
    """
    try:
        # Validate that end_date is after start_date
        if request.end_date <= request.start_date:
            raise HTTPException(
                status_code=400, 
                detail="End date must be after start date"
            )
            
        ccdm_service = CCDMService(db)
        return ccdm_service.detect_shape_changes(request)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting shape changes: {str(e)}")

@router.get("/quick-assessment/{norad_id}", response_model=ObjectThreatAssessment)
def quick_assess(
    norad_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(check_roles([Roles.viewer]))
):
    """
    Quick threat assessment for a space object by NORAD ID
    """
    try:
        ccdm_service = CCDMService(db)
        return ccdm_service.quick_assess_norad_id(norad_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing quick assessment: {str(e)}")

@router.get("/last-week-analysis/{norad_id}", response_model=HistoricalAnalysisResponse)
def get_last_week_analysis(
    norad_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(check_roles([Roles.viewer]))
):
    """
    Get analysis data for the last week for a space object
    """
    try:
        ccdm_service = CCDMService(db)
        return ccdm_service.get_last_week_analysis(norad_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting last week analysis: {str(e)}")

@router.post("/assess_thermal_signature", response_model=ThermalSignatureResponse)
async def assess_thermal_signature(
    object_id: str,
    timestamp: datetime,
    current_user: User = Depends(check_roles([Roles.viewer])),
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
    current_user: User = Depends(check_roles([Roles.admin])),
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
    current_user: User = Depends(check_roles([Roles.admin])),
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
    current_user: User = Depends(check_roles([Roles.viewer])),
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
    current_user: User = Depends(check_roles([Roles.viewer])),
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