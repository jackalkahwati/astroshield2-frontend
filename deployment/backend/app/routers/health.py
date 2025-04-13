from fastapi import APIRouter
from datetime import datetime

router = APIRouter()

@router.get("/health")
@router.get("/health/")  # Handle both with and without trailing slash
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    } 