"""
Maneuver detection algorithms and utilities
"""
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional

def detect_maneuvers_from_states(states: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze state data to detect maneuvers
    
    Args:
        states: List of state vectors
        
    Returns:
        Dictionary with maneuver detection results
    """
    if len(states) < 2:
        return {
            "detected": False, 
            "reason": "insufficient_data"
        }
        
    # Calculate delta-v between consecutive states
    delta_vs = []
    timestamps = []
    
    for i in range(len(states) - 1):
        v1 = np.array([states[i]["velocity"]["x"], states[i]["velocity"]["y"], states[i]["velocity"]["z"]])
        v2 = np.array([states[i+1]["velocity"]["x"], states[i+1]["velocity"]["y"], states[i+1]["velocity"]["z"]])
        
        delta_v = np.linalg.norm(v2 - v1)
        
        if delta_v > 0.001:  # Minimum threshold in km/s
            delta_vs.append(delta_v)
            timestamps.append(states[i+1]["epoch"])
    
    if not delta_vs:
        return {
            "detected": False,
            "reason": "no_significant_changes"
        }
        
    # Find the largest delta-v
    max_delta_v_index = np.argmax(delta_vs)
    max_delta_v = delta_vs[max_delta_v_index]
    max_delta_v_time = timestamps[max_delta_v_index]
    
    # Classify maneuver
    maneuver_type = classify_maneuver(max_delta_v)
    
    return {
        "detected": True,
        "delta_v": max_delta_v,
        "time": max_delta_v_time,
        "maneuver_type": maneuver_type,
        "confidence": calculate_confidence(max_delta_v)
    }

def classify_maneuver(delta_v: float) -> str:
    """
    Classify the type of maneuver based on delta-v magnitude
    
    Args:
        delta_v: The delta-v magnitude in km/s
        
    Returns:
        Maneuver type classification
    """
    if delta_v > 0.5:
        return "ORBIT_CHANGE"
    elif delta_v > 0.1:
        return "ORBIT_ADJUSTMENT"
    elif delta_v > 0.05:
        return "STATION_KEEPING"
    elif delta_v > 0.01:
        return "ATTITUDE_ADJUSTMENT"
    else:
        return "MINOR_CORRECTION"

def calculate_confidence(delta_v: float) -> float:
    """
    Calculate confidence score for maneuver detection
    
    Args:
        delta_v: The delta-v magnitude in km/s
        
    Returns:
        Confidence score between 0 and 1
    """
    # Simple confidence model based on delta-v magnitude
    # Higher delta-v typically means higher confidence
    base_confidence = min(0.9, 0.5 + delta_v)
    
    # Cap confidence between 0.1 and 0.95
    return max(0.1, min(0.95, base_confidence)) 