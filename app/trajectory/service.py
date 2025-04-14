"""
Trajectory analysis services
"""
import random
import math
from typing import List, Dict, Any
from datetime import datetime
import numpy as np

from app.common.logging import logger
from app.trajectory.models import TrajectoryResult

def simulate_trajectory(config: Dict[str, Any], initial_state: List[float]) -> Dict[str, Any]:
    """Simple trajectory simulation for demonstration purposes"""
    logger.info(f"Simulating trajectory for {config['object_name']}")
    
    # Extract parameters
    mass = config["object_properties"]["mass"]
    altitude = initial_state[2]  # Initial altitude
    
    # Generate trajectory points
    num_points = 100
    trajectory_points = []
    
    start_time = datetime.now().timestamp()
    time_step = 60  # 1 minute per step
    
    # Starting position and velocity
    lon, lat, alt = initial_state[0:3]
    vx, vy, vz = initial_state[3:6]
    
    for i in range(num_points):
        # Time for this point
        time = start_time + i * time_step
        
        # Update altitude with some randomness for realism
        alt = max(0, altitude - (i * altitude / num_points) + random.uniform(-500, 500))
        
        # Adjust longitude and latitude based on velocity and Earth's curvature
        earth_radius = 6371000  # meters
        meters_per_degree_lon = 111320 * math.cos(math.radians(lat))
        meters_per_degree_lat = 110574
        
        lon_change = (vx * time_step) / meters_per_degree_lon
        lat_change = (vy * time_step) / meters_per_degree_lat
        
        lon += lon_change
        lat += lat_change
        
        # Ensure longitude wraps around properly
        lon = (lon + 180) % 360 - 180
        
        # Update velocities with some atmospheric drag effects
        if alt < 100000:  # Below 100km, significant atmosphere
            drag_factor = 1 - (alt / 100000) * 0.1
            vx *= drag_factor
            vy *= drag_factor
            vz *= drag_factor
        
        # Add trajectory point
        trajectory_points.append({
            "time": time,
            "position": [lon, lat, alt],
            "velocity": [vx, vy, vz]
        })
        
        # Terminal velocity increases as atmosphere gets denser
        if alt < 50000:
            vz -= 9.8 * time_step * (1 + (50000 - alt) / 10000)
        else:
            vz -= 9.8 * time_step * 0.5  # Reduced gravity effect at higher altitudes
    
    # Impact prediction
    impact_prediction = {
        "time": trajectory_points[-1]["time"],
        "position": trajectory_points[-1]["position"],
        "confidence": 0.95,
        "energy": 0.5 * mass * sum(v**2 for v in trajectory_points[-1]["velocity"]),
        "area": config["object_properties"]["area"] * (1 + random.uniform(0, 0.5))  # Slight expansion on impact
    }
    
    # Breakup points (if enabled)
    breakup_points = []
    if config["breakup_model"]["enabled"]:
        # Add random breakup points if altitude and energy conditions are met
        for i in range(1, len(trajectory_points) - 1):
            point = trajectory_points[i]
            if point["position"][2] < 80000 and random.random() < 0.1:  # 10% chance below 80km
                velocity_magnitude = math.sqrt(sum(v**2 for v in point["velocity"]))
                if velocity_magnitude > 2000:  # Only break up at high speeds
                    breakup_points.append({
                        "time": point["time"],
                        "position": point["position"],
                        "fragments": random.randint(5, 30),
                        "cause": random.choice(["Aerodynamic Stress", "Thermal Stress", "Material Failure"])
                    })
    
    return {
        "trajectory": trajectory_points,
        "impactPrediction": impact_prediction,
        "breakupPoints": breakup_points
    } 