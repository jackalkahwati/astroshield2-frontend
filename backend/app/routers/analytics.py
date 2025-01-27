from fastapi import APIRouter
from datetime import datetime

router = APIRouter()

@router.get("/status")
async def get_analytics_status():
    """Get analytics system status"""
    return {
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
        "active_processes": 0
    } 