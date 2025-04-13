from fastapi import APIRouter, HTTPException, Depends, Query
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import random
from app.services.analytics_service import AnalyticsService, PerformanceMetrics
from app.core.security import get_current_user, check_roles
from app.models.user import User

router = APIRouter()
analytics_service = AnalyticsService()

@router.get("/status")
async def get_analytics_status():
    """Get analytics system status"""
    return {
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
        "active_processes": random.randint(3, 8)
    }

@router.get("/analytics")
async def get_analytics_dashboard(
    current_user: User = Depends(check_roles(["active"]))
):
    """Get analytics dashboard data"""
    try:
        metrics = await analytics_service.get_dashboard_metrics()
        return {
            "status": "operational",
            "timestamp": datetime.utcnow().isoformat(),
            "data": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/trends")
async def get_analytics_trends(
    days: int = Query(7, description="Number of days of trend data to retrieve"),
    current_user: User = Depends(check_roles(["active"]))
):
    """Get time-series analytics data for trends"""
    try:
        # Get trend data from service
        data_points = await analytics_service.get_trends(days)
        
        # Calculate time range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        return {
            "status": "operational",
            "timestamp": datetime.utcnow().isoformat(),
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "days": days
            },
            "data": data_points
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/performance")
async def get_performance_metrics(
    hours: int = Query(24, description="Number of hours of performance data to retrieve"),
    current_user: User = Depends(check_roles(["active", "admin"]))
):
    """Get system performance metrics"""
    try:
        # Get performance metrics from service
        metrics = await analytics_service.get_performance_metrics(hours)
        
        # Convert to dict format for API response
        data_points = [
            {
                "timestamp": m.timestamp.isoformat(),
                "cpu_usage": m.cpu_usage,
                "memory_usage": m.memory_usage,
                "api_requests": m.api_requests,
                "response_time_ms": m.response_time_ms,
                "error_rate": m.error_rate
            } for m in metrics
        ]
        
        # Calculate time range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        return {
            "status": "operational",
            "timestamp": datetime.utcnow().isoformat(),
            "period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "hours": hours
            },
            "data": data_points
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/satellite/{satellite_id}")
async def get_satellite_analytics(
    satellite_id: str,
    current_user: User = Depends(check_roles(["active"]))
):
    """Get analytics specifically for one satellite"""
    try:
        data = await analytics_service.get_satellite_analytics(satellite_id)
        return {
            "satellite_id": satellite_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/conjunctions")
async def get_conjunction_analytics(
    days: int = Query(30, description="Number of days of conjunction data to analyze"),
    current_user: User = Depends(check_roles(["active"]))
):
    """Get analytics for conjunction events"""
    try:
        data = await analytics_service.get_conjunction_analytics(days)
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "period_days": days,
            "data": data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/predictive")
async def get_predictive_analytics(
    current_user: User = Depends(check_roles(["active"]))
):
    """Get predictive analytics for future events"""
    try:
        data = await analytics_service.get_predictive_analytics()
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "prediction_timestamp": datetime.utcnow().isoformat(),
            "data": data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))