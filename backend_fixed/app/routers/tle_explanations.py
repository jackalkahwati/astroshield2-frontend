"""
TLE Explanations Router
Provides endpoints for natural language TLE explanations
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..ml_infrastructure.tle_explainer import (
    TLEExplainerInput, 
    TLEExplanation,
    MockTLEExplainerService
)

router = APIRouter(prefix="/tle-explanations", tags=["TLE Explanations"])

# Initialize service (will be replaced by configuration)
tle_explainer = MockTLEExplainerService()

@router.post("/explain", response_model=TLEExplanation)
async def explain_single_tle(tle_input: TLEExplainerInput):
    """Get natural language explanation for a single TLE"""
    try:
        explanation = await tle_explainer.explain_tle(tle_input)
        return explanation
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/explain-batch", response_model=List[TLEExplanation])
async def explain_multiple_tles(tle_inputs: List[TLEExplainerInput]):
    """Get explanations for multiple TLEs"""
    explanations = []
    
    for tle_input in tle_inputs:
        try:
            explanation = await tle_explainer.explain_tle(tle_input)
            explanations.append(explanation)
        except Exception as e:
            # Log error but continue processing
            print(f"Error explaining TLE {tle_input.norad_id}: {e}")
            
    return explanations

@router.get("/satellites/{norad_id}/explanation")
async def get_satellite_tle_explanation(norad_id: str):
    """Get TLE explanation for a specific satellite by NORAD ID"""
    # Mock TLE data (ISS)
    if norad_id == "25544":
        tle_input = TLEExplainerInput(
            norad_id="25544",
            satellite_name="ISS (ZARYA)",
            line1="1 25544U 98067A   24079.07757601 .00016717 00000+0 10270-3 0  9994",
            line2="2 25544  51.6400 337.6640 0007776  35.5310 330.5120 15.50377579499263"
        )
        
        explanation = await tle_explainer.explain_tle(tle_input)
        return explanation
    else:
        raise HTTPException(status_code=404, detail=f"Satellite {norad_id} not found")

@router.get("/risk-assessment/{risk_level}")
async def get_satellites_by_risk(risk_level: str, limit: int = 10):
    """Get satellites filtered by decay risk level"""
    valid_levels = ["LOW", "MEDIUM", "HIGH"]
    
    if risk_level.upper() not in valid_levels:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid risk level. Must be one of: {valid_levels}"
        )
    
    return {
        "risk_level": risk_level.upper(),
        "count": 0,
        "satellites": []
    }

@router.get("/educational/{topic}")
async def get_educational_content(topic: str):
    """Get educational content about orbital mechanics topics"""
    topics = {
        "tle": {
            "title": "Understanding Two-Line Elements",
            "content": "A TLE (Two-Line Element set) is a data format used to convey orbital information about Earth-orbiting objects...",
            "examples": ["ISS TLE breakdown", "How to read TLE data"]
        },
        "orbits": {
            "title": "Types of Earth Orbits",
            "content": "Satellites orbit Earth at various altitudes and inclinations, each serving different purposes...",
            "types": ["LEO", "MEO", "GEO", "HEO", "SSO"]
        },
        "decay": {
            "title": "Orbital Decay and Reentry",
            "content": "Atmospheric drag causes satellites in low orbits to gradually lose altitude...",
            "factors": ["Altitude", "Solar activity", "Satellite shape"]
        }
    }
    
    if topic not in topics:
        raise HTTPException(
            status_code=404, 
            detail=f"Topic not found. Available topics: {list(topics.keys())}"
        )
    
    return topics[topic] 