import numpy as np
from datetime import datetime, timedelta
import json

def detect_maneuvers_from_states(states):
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
        "confidence": min(0.9, 0.5 + max_delta_v)
    }

def classify_maneuver(delta_v):
    """Classify maneuver type based on delta-v magnitude"""
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

def create_sample_state_data(include_maneuver=True):
    """Create sample state vector data for testing"""
    now = datetime.utcnow()
    states = []
    
    # Create a circular orbit
    semi_major_axis = 7000.0  # km
    period = 2 * np.pi * np.sqrt(semi_major_axis**3 / 398600.4418)  # seconds
    
    for i in range(10):
        t = now + timedelta(minutes=i*10)
        epoch = t.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        
        # Calculate position and velocity for circular orbit
        angle = 2 * np.pi * (i * 600) / period
        
        # Position vector (km)
        x = semi_major_axis * np.cos(angle)
        y = semi_major_axis * np.sin(angle)
        z = 0.0
        
        # Velocity vector (km/s)
        vx = -np.sqrt(398600.4418 / semi_major_axis) * np.sin(angle)
        vy = np.sqrt(398600.4418 / semi_major_axis) * np.cos(angle)
        vz = 0.0
        
        # Add maneuver in the middle of the dataset
        if include_maneuver and i == 5:
            # Increase velocity for orbit raising maneuver
            vx *= 1.1
            vy *= 1.1
        
        state = {
            "epoch": epoch,
            "position": {
                "x": x,
                "y": y,
                "z": z
            },
            "velocity": {
                "x": vx,
                "y": vy,
                "z": vz
            }
        }
        states.append(state)
    
    return states

if __name__ == "__main__":
    # Create sample data with a maneuver
    states_with_maneuver = create_sample_state_data(include_maneuver=True)
    
    # Detect maneuvers
    detection_result = detect_maneuvers_from_states(states_with_maneuver)
    
    print("Maneuver Detection Results:")
    print(json.dumps(detection_result, indent=2))
    
    if detection_result.get("detected", False):
        # Create the standardized message format
        maneuver_event = {
            "header": {
                "messageType": "maneuver-detected",
                "source": "dmd-od-integration",
                "timestamp": detection_result.get("time")
            },
            "payload": {
                "catalogId": "DMD-12345",
                "deltaV": detection_result.get("delta_v"),
                "confidence": detection_result.get("confidence", 0.5),
                "maneuverType": detection_result.get("maneuver_type", "UNKNOWN"),
                "detectionTime": detection_result.get("time")
            }
        }
        
        print("\nStandardized Maneuver Event:")
        print(json.dumps(maneuver_event, indent=2))
    
    # Test with data without maneuvers
    states_without_maneuver = create_sample_state_data(include_maneuver=False)
    no_maneuver_result = detect_maneuvers_from_states(states_without_maneuver)
    
    print("\nNo Maneuver Detection Results:")
    print(json.dumps(no_maneuver_result, indent=2)) 