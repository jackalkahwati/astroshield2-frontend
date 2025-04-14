"""
Maneuver management services
"""
import random
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import uuid

from app.common.logging import logger
from app.maneuvers.models import Maneuver, ManeuverCreateRequest

# In-memory storage for maneuvers
maneuvers_db = [
    {
        "id": "MNV-1001",
        "satellite_id": "SAT-001",
        "type": "hohmann",
        "status": "completed",
        "scheduledTime": (datetime.now() - timedelta(days=7)).isoformat(),
        "completedTime": (datetime.now() - timedelta(days=7, hours=2)).isoformat(),
        "created_by": "system",
        "created_at": (datetime.now() - timedelta(days=7, hours=5)).isoformat(),
        "details": {
            "delta_v": 3.5,
            "duration": 120,
            "fuel_required": 5.2,
            "fuel_used": 5.3,
            "target_orbit": {
                "altitude": 700,
                "inclination": 51.6
            }
        }
    },
    {
        "id": "MNV-1002",
        "satellite_id": "SAT-002",
        "type": "stationkeeping",
        "status": "scheduled",
        "scheduledTime": (datetime.now() + timedelta(days=2)).isoformat(),
        "created_by": "operator",
        "created_at": (datetime.now() - timedelta(days=1)).isoformat(),
        "details": {
            "delta_v": 0.8,
            "duration": 35,
            "fuel_required": 1.1,
            "target_orbit": {
                "altitude": 420,
                "inclination": 45.0
            }
        }
    },
    {
        "id": "MNV-1003",
        "satellite_id": "SAT-001",
        "type": "collision",
        "status": "executing",
        "scheduledTime": (datetime.now() - timedelta(hours=2)).isoformat(),
        "created_by": "auto",
        "created_at": (datetime.now() - timedelta(hours=3)).isoformat(),
        "details": {
            "delta_v": 1.2,
            "duration": 45,
            "fuel_required": 2.0
        }
    }
]

def get_all_maneuvers() -> List[Dict]:
    """Get all maneuvers"""
    logger.info("Fetching all maneuvers")
    return maneuvers_db

def get_maneuver_by_id(maneuver_id: str) -> Optional[Dict]:
    """Get a specific maneuver by ID"""
    logger.info(f"Fetching maneuver with ID: {maneuver_id}")
    
    for maneuver in maneuvers_db:
        if maneuver["id"] == maneuver_id:
            return maneuver
    
    return None

def create_new_maneuver(request: ManeuverCreateRequest) -> Dict:
    """Create a new maneuver"""
    logger.info(f"Creating new {request.type} maneuver")
    
    # Generate a new maneuver ID
    maneuver_id = f"MNV-{random.randint(1000, 9999)}"
    
    # Create the new maneuver
    new_maneuver = {
        "id": maneuver_id,
        "satellite_id": request.satellite_id,
        "type": request.type,
        "status": request.status,
        "scheduledTime": request.scheduledTime,
        "completedTime": None,
        "created_by": "user",
        "created_at": datetime.now().isoformat(),
        "details": request.details.dict(exclude_none=True)
    }
    
    # Add to database
    maneuvers_db.append(new_maneuver)
    
    return new_maneuver 