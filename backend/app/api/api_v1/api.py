from fastapi import APIRouter
from datetime import datetime

api_router = APIRouter()

@api_router.get("/status")
async def get_api_status():
    """Get API status"""
    return {
        "status": "operational",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    } 