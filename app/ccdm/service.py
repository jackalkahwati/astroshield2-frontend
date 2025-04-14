"""
CCDM services
"""
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import random

from app.common.logging import logger
from app.ccdm.models import ConjunctionEvent, ConjunctionCreateRequest, ConjunctionFilterRequest

# In-memory storage for conjunctions
conjunctions_db = [
    {
        "id": "CONJ-001",
        "time_of_closest_approach": (datetime.now() + timedelta(hours=6)).isoformat(),
        "miss_distance": 125.5,  # meters
        "probability_of_collision": 0.000125,
        "relative_velocity": 10500.0,  # m/s
        "primary_object": {
            "object_id": "2021-045A",
            "name": "Starlink-2567",
            "type": "PAYLOAD",
            "size": 2.5,
            "mass": 260.0,
            "owner": "SpaceX",
            "country": "USA"
        },
        "secondary_object": {
            "object_id": "1999-025DK",
            "name": "Cosmos 2345 Debris",
            "type": "DEBRIS",
            "size": 0.15,
            "owner": "ROSCOSMOS",
            "country": "RUS"
        },
        "created_at": datetime.now().isoformat(),
        "status": "PENDING"
    },
    {
        "id": "CONJ-002",
        "time_of_closest_approach": (datetime.now() + timedelta(days=1)).isoformat(),
        "miss_distance": 350.2,  # meters
        "probability_of_collision": 0.0000032,
        "relative_velocity": 9800.0,  # m/s
        "primary_object": {
            "object_id": "2018-092A",
            "name": "ISS",
            "type": "PAYLOAD",
            "size": 108.5,
            "mass": 420000.0,
            "owner": "NASA",
            "country": "USA"
        },
        "secondary_object": {
            "object_id": "2008-039B",
            "name": "Rocket Body",
            "type": "ROCKET_BODY",
            "size": 4.2,
            "owner": "CNSA",
            "country": "CHN"
        },
        "created_at": (datetime.now() - timedelta(hours=12)).isoformat(),
        "status": "ANALYZING"
    }
]

def get_all_conjunctions() -> List[Dict]:
    """Get all conjunction events"""
    logger.info("Fetching all conjunction events")
    return conjunctions_db

def get_conjunction_by_id(conjunction_id: str) -> Optional[Dict]:
    """Get a specific conjunction by ID"""
    logger.info(f"Fetching conjunction with ID: {conjunction_id}")
    
    for conjunction in conjunctions_db:
        if conjunction["id"] == conjunction_id:
            return conjunction
    
    return None

def create_conjunction(request: ConjunctionCreateRequest) -> Dict:
    """Create a new conjunction event"""
    logger.info(f"Creating new conjunction event")
    
    # Generate a unique ID
    conjunction_id = f"CONJ-{str(uuid.uuid4())[:8]}"
    
    new_conjunction = {
        "id": conjunction_id,
        "time_of_closest_approach": request.time_of_closest_approach,
        "miss_distance": request.miss_distance,
        "probability_of_collision": request.probability_of_collision,
        "relative_velocity": request.relative_velocity,
        "primary_object": request.primary_object.dict(),
        "secondary_object": request.secondary_object.dict(),
        "created_at": datetime.now().isoformat(),
        "status": "PENDING"
    }
    
    conjunctions_db.append(new_conjunction)
    
    return new_conjunction

def filter_conjunctions(filter_request: ConjunctionFilterRequest) -> List[Dict]:
    """Filter conjunction events based on criteria"""
    logger.info("Filtering conjunction events")
    
    filtered = conjunctions_db
    
    if filter_request.start_date:
        start = datetime.fromisoformat(filter_request.start_date)
        filtered = [c for c in filtered if datetime.fromisoformat(c["time_of_closest_approach"]) >= start]
    
    if filter_request.end_date:
        end = datetime.fromisoformat(filter_request.end_date)
        filtered = [c for c in filtered if datetime.fromisoformat(c["time_of_closest_approach"]) <= end]
    
    if filter_request.object_id:
        filtered = [c for c in filtered if 
                  c["primary_object"]["object_id"] == filter_request.object_id or 
                  c["secondary_object"]["object_id"] == filter_request.object_id]
    
    if filter_request.min_probability is not None:
        filtered = [c for c in filtered if c["probability_of_collision"] >= filter_request.min_probability]
    
    if filter_request.status:
        filtered = [c for c in filtered if c["status"] == filter_request.status]
    
    return filtered 