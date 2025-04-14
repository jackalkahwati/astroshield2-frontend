"""
Trajectory analysis routes
"""
from fastapi import APIRouter, HTTPException
from app.common.logging import logger
from app.trajectory.models import TrajectoryRequest, TrajectoryResult
from app.trajectory.service import simulate_trajectory

# Create router for trajectory endpoints
router = APIRouter(prefix="/api/trajectory", tags=["trajectory"])

@router.post("/analyze", response_model=TrajectoryResult)
async def analyze_trajectory(request: TrajectoryRequest):
    """
    Analyze a trajectory and return predictions.
    
    Parameters:
    - config: Configuration for the trajectory analysis including object properties
    - initial_state: Initial position and velocity of the object [x, y, z, vx, vy, vz]
    """
    try:
        logger.info(f"Analyzing trajectory for {request.config.object_name}")
        
        # Perform trajectory simulation
        result = simulate_trajectory(
            request.config.dict(),
            request.initial_state
        )
        
        return result
    except Exception as e:
        logger.error(f"Error analyzing trajectory: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing trajectory: {str(e)}") 