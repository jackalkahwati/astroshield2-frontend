from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
import random

router = APIRouter()

@router.get("/api/comprehensive/data")
async def get_comprehensive_data():
    return {
        "metrics": {
            "attitude_stability": 95.5,
            "orbit_stability": 98.2,
            "thermal_stability": 87.3,
            "power_stability": 92.8,
            "communication_stability": 96.1
        },
        "status": "nominal",
        "alerts": [],
        "timestamp": "2024-01-21T12:00:00Z"
    }

@router.get("/api/stability/metrics")
async def get_stability_metrics():
    return {
        "metrics": {
            "attitude_stability": 95.5,
            "orbit_stability": 98.2,
            "thermal_stability": 87.3,
            "power_stability": 92.8,
            "communication_stability": 96.1
        },
        "status": "nominal",
        "timestamp": "2024-01-21T12:00:00Z"
    }

@router.get("/api/analytics/data")
async def get_analytics_data():
    # Generate mock analytics data
    current_time = datetime.now()
    
    # Generate daily trends data
    daily_trends = []
    for i in range(24):  # Last 24 hours
        timestamp = current_time - timedelta(hours=23-i)
        daily_trends.append({
            "timestamp": timestamp.isoformat(),
            "stability_score": random.uniform(85, 98),
            "anomaly_count": random.randint(0, 3),
            "power_efficiency": random.uniform(90, 99),
            "thermal_status": random.uniform(85, 95),
            "communication_quality": random.uniform(92, 99)
        })
    
    return {
        "summary": {
            "total_operational_time": 720,  # 30 days in hours
            "total_anomalies_detected": 15,
            "average_stability": 94.5,
            "current_health_score": 96.2
        },
        "current_metrics": {
            "power_consumption": {
                "value": 95.5,
                "trend": "stable",
                "status": "nominal"
            },
            "thermal_control": {
                "value": 92.3,
                "trend": "improving",
                "status": "nominal"
            },
            "communication_quality": {
                "value": 97.8,
                "trend": "stable",
                "status": "nominal"
            },
            "orbit_stability": {
                "value": 98.1,
                "trend": "stable",
                "status": "nominal"
            }
        },
        "trends": {
            "daily": daily_trends,
            "weekly_summary": {
                "average_stability": 93.8,
                "total_anomalies": 8,
                "power_efficiency": 95.2,
                "communication_uptime": 99.1
            },
            "monthly_summary": {
                "average_stability": 94.1,
                "total_anomalies": 32,
                "power_efficiency": 94.8,
                "communication_uptime": 98.7
            }
        }
    }
