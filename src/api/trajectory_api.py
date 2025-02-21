"""FastAPI endpoint for trajectory analysis."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Generator
from datetime import datetime
import numpy as np
from itertools import islice

from models.trajectory_predictor import TrajectoryPredictor

app = FastAPI()

class TrajectoryConfig(BaseModel):
    """Configuration for trajectory analysis."""
    atmospheric_model: str
    wind_model: str
    monte_carlo_samples: int
    object_properties: Dict[str, float]
    breakup_model: Dict[str, Any]

class TrajectoryRequest(BaseModel):
    """Request body for trajectory analysis."""
    config: TrajectoryConfig
    initial_state: List[float]

def generate_trajectory_points(predictor: TrajectoryPredictor,
                            initial_state: np.ndarray,
                            mass: float,
                            end_time: float,
                            step: float = 1.0) -> Generator[Dict[str, Any], None, None]:
    """Generate trajectory points efficiently using a generator."""
    t = 0
    while t <= end_time:
        state = predictor._dynamics(initial_state, t, mass)
        yield {
            'time': t,
            'position': state[:3].tolist(),
            'velocity': state[3:].tolist()
        }
        t += step

def process_trajectory_batch(points: List[Dict[str, Any]], 
                           predictor: TrajectoryPredictor,
                           mass: float,
                           breakup_enabled: bool) -> List[Dict[str, Any]]:
    """Process a batch of trajectory points for breakup events."""
    breakup_events = []
    
    if breakup_enabled:
        for point in points:
            altitude = np.linalg.norm(point['position']) - predictor.R
            velocity = np.linalg.norm(point['velocity'])
            
            # Get atmospheric conditions
            conditions = predictor.get_atmospheric_conditions(altitude)
            
            # Calculate dynamic pressure
            q = 0.5 * conditions.density * velocity**2
            
            # Breakup threshold (example: 50 kPa)
            if q > 50000:
                fragments = predictor.model_breakup(
                    np.concatenate([point['position'], point['velocity']]),
                    mass,
                    altitude
                )
                
                breakup_events.append({
                    'time': datetime.utcnow().isoformat(),
                    'altitude': altitude / 1000.0,  # Convert to km
                    'fragments': len(fragments)
                })
                break  # Only model first breakup event for now
    
    return breakup_events

@app.post("/api/trajectory/analyze")
async def analyze_trajectory(request: TrajectoryRequest):
    """Analyze trajectory with given configuration and initial state."""
    try:
        # Initialize trajectory predictor
        predictor = TrajectoryPredictor(request.config.dict())
        
        # Convert initial state to numpy array
        initial_state = np.array(request.initial_state)
        
        # Get object properties
        mass = request.config.object_properties['mass']
        
        # Predict trajectory with Monte Carlo analysis
        impact_prediction = predictor.predict_impact(
            initial_state=initial_state,
            mass=mass,
            time_step=1.0,
            max_time=3600.0,
            monte_carlo=True
        )
        
        if impact_prediction is None:
            raise HTTPException(
                status_code=400,
                detail="No impact predicted within time limit"
            )
        
        # Generate trajectory points in batches
        trajectory_points = []
        batch_size = 100
        point_generator = generate_trajectory_points(
            predictor,
            initial_state,
            mass,
            impact_prediction['time']
        )
        
        breakup_events = []
        while True:
            batch = list(islice(point_generator, batch_size))
            if not batch:
                break
                
            trajectory_points.extend(batch)
            
            # Process batch for breakup events
            if request.config.breakup_model['enabled']:
                events = process_trajectory_batch(
                    batch,
                    predictor,
                    mass,
                    request.config.breakup_model['enabled']
                )
                breakup_events.extend(events)
        
        return {
            'trajectory': trajectory_points,
            'impact_prediction': impact_prediction,
            'breakup_events': breakup_events
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error in trajectory analysis: {str(e)}"
        ) 