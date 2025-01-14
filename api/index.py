from fastapi import FastAPI, Header, HTTPException, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mangum import Mangum
from typing import Optional, Dict, List
import random
from datetime import datetime, timedelta
import logging
import traceback
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware with more permissive settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global settings store (in a real app, this would be in a database)
app_settings = {
    "theme": "dark",
    "notifications": True,
    "autoUpdate": True
}

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    try:
        response = await call_next(request)
        logger.info(f"Response status: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Request failed: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"detail": str(e)}
        )

@app.get("/")
async def read_root():
    return {
        "message": "Welcome to AstroShield API",
        "version": "1.0.1",
        "status": "operational"
    }

@app.get("/api/settings")
async def get_settings():
    try:
        return app_settings
    except Exception as e:
        logger.error(f"Error in get_settings: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.put("/api/settings")
async def update_settings(settings: Dict = Body(...)):
    try:
        # Update only the provided settings
        app_settings.update(settings)
        return app_settings
    except Exception as e:
        logger.error(f"Error in update_settings: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.get("/api/comprehensive/data")
async def get_comprehensive_data():
    try:
        logger.info("Fetching comprehensive data")
        return {
            "metrics": {
                "attitude_stability": round(random.uniform(85, 98), 2),
                "orbit_stability": round(random.uniform(88, 99), 2),
                "thermal_stability": round(random.uniform(80, 95), 2),
                "power_stability": round(random.uniform(85, 97), 2),
                "communication_stability": round(random.uniform(87, 98), 2)
            },
            "status": "nominal",
            "alerts": [],
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in get_comprehensive_data: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.get("/api/stability/metrics")
async def get_stability_metrics():
    try:
        return {
            "metrics": {
                "attitude_stability": random.uniform(85, 98),
                "orbit_stability": random.uniform(88, 99),
                "thermal_stability": random.uniform(80, 95),
                "power_stability": random.uniform(85, 97),
                "communication_stability": random.uniform(87, 98)
            },
            "status": "nominal",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in get_stability_metrics: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.get("/api/analytics/data")
async def get_analytics_data(timeRange: str = "24h"):
    """Get analytics data for the specified time range."""
    try:
        # Generate mock data
        current_metrics = {
            "power_consumption": {
                "value": round(random.uniform(85, 95), 2),
                "trend": random.choice(["stable", "increasing", "decreasing"]),
                "status": random.choice(["normal", "warning", "critical"])
            },
            "thermal_control": {
                "value": round(random.uniform(88, 98), 2),
                "trend": random.choice(["stable", "increasing", "decreasing"]),
                "status": random.choice(["normal", "warning", "critical"])
            },
            "communication_quality": {
                "value": round(random.uniform(90, 99), 2),
                "trend": random.choice(["stable", "increasing", "decreasing"]),
                "status": random.choice(["normal", "warning", "critical"])
            },
            "orbit_stability": {
                "value": round(random.uniform(92, 99), 2),
                "trend": random.choice(["stable", "increasing", "decreasing"]),
                "status": random.choice(["normal", "warning", "critical"])
            }
        }

        # Generate daily trends
        now = datetime.now()
        daily_trends = []
        for i in range(24):
            timestamp = (now - timedelta(hours=i)).isoformat()
            daily_trends.append({
                "timestamp": timestamp,
                "stability_score": round(random.uniform(85, 98), 2),
                "anomaly_count": random.randint(0, 3),
                "power_efficiency": round(random.uniform(88, 96), 2),
                "thermal_status": round(random.uniform(90, 98), 2),
                "communication_quality": round(random.uniform(92, 99), 2)
            })

        return {
            "summary": {
                "total_operational_time": round(random.uniform(720, 744), 1),
                "total_anomalies_detected": random.randint(5, 15),
                "average_stability": round(random.uniform(92, 98), 2),
                "current_health_score": round(random.uniform(90, 99), 2)
            },
            "current_metrics": current_metrics,
            "trends": {
                "daily": daily_trends,
                "weekly_summary": {
                    "average_stability": round(random.uniform(90, 95), 2),
                    "total_anomalies": random.randint(20, 40),
                    "power_efficiency": round(random.uniform(88, 94), 2),
                    "communication_uptime": round(random.uniform(95, 99), 2)
                },
                "monthly_summary": {
                    "average_stability": round(random.uniform(89, 94), 2),
                    "total_anomalies": random.randint(80, 120),
                    "power_efficiency": round(random.uniform(87, 93), 2),
                    "communication_uptime": round(random.uniform(94, 98), 2)
                }
            }
        }
    except Exception as e:
        logger.error(f"Error generating analytics data: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate analytics data")

@app.get("/api/maneuvers")
async def get_maneuvers():
    try:
        data = {
            "planned_maneuvers": [
                {
                    "id": "m1",
                    "type": "orbit_correction",
                    "scheduled_time": (datetime.now() + timedelta(hours=24)).isoformat(),
                    "status": "scheduled"
                }
            ],
            "completed_maneuvers": [
                {
                    "id": "m0",
                    "type": "debris_avoidance",
                    "execution_time": (datetime.now() - timedelta(hours=12)).isoformat(),
                    "status": "completed"
                }
            ]
        }
        return data
    except Exception as e:
        logger.error(f"Error in get_maneuvers: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={"detail": str(e)})

# Configure handler for Vercel serverless deployment
handler = Mangum(app, lifespan="off")

# Export the handler for Vercel
__all__ = ['handler']

# Make sure we don't reassign app
# app = app  # Remove this line 