import numpy as np
from datetime import datetime, timedelta

def generate_test_data(object_id: str = "TEST-001") -> dict:
    """Generate test data for demonstration"""
    
    # Generate time series data
    time_points = 100
    timestamps = [
        (datetime.now() - timedelta(hours=i)).isoformat()
        for i in range(time_points)
    ]
    
    # Orbital features (position, velocity, acceleration)
    orbital_features = np.random.randn(time_points, 32).astype(np.float32)
    
    # Trajectory features
    trajectory_features = np.random.randn(time_points, 32).astype(np.float32)
    
    # Historical stability (with intentional anomaly)
    historical_stability = np.ones(time_points)
    historical_stability[40:60] += np.sin(np.linspace(0, 2*np.pi, 20)) * 0.5
    
    # RF emissions data
    rf_emissions = [
        {
            'timestamp': ts,
            'frequency': 1000 + np.random.randn(),
            'power': np.random.rand(),
            'modulation': 'unknown'
        }
        for ts in timestamps[::10]  # Every 10th timestamp
    ]
    
    # Signature features
    signature_features = np.random.randn(32).astype(np.float32)
    
    # Orbital elements
    orbital_elements = {
        'semi_major_axis': 7000 + np.random.randn(),
        'eccentricity': 0.01 + np.random.rand() * 0.01,
        'inclination': 45 + np.random.randn(),
        'raan': np.random.rand() * 360,
        'arg_perigee': np.random.rand() * 360,
        'mean_anomaly': np.random.rand() * 360
    }
    
    # Space environment data
    space_environment = {
        'radiation_flux': np.random.rand() * 100,
        'particle_density': np.random.rand() * 10,
        'magnetic_field': np.random.randn(3),
        'solar_activity': np.random.rand()
    }
    
    # Launch data
    launch_data = {
        'launch_site': 'SITE-001',
        'launch_time': (datetime.now() - timedelta(days=30)).isoformat(),
        'expected_objects': 1,
        'tracked_objects': 2 if np.random.rand() > 0.7 else 1,
        'launch_profile': 'suspicious' if np.random.rand() > 0.8 else 'normal'
    }
    
    # Compliance data
    compliance_data = {
        'itu_filing': {
            'status': 'filed' if np.random.rand() > 0.2 else 'missing',
            'frequency_bands': ['C-band', 'Ku-band'],
            'orbital_position': orbital_elements['semi_major_axis']
        },
        'un_registry': {
            'status': 'registered' if np.random.rand() > 0.3 else 'unregistered',
            'registration_date': (datetime.now() - timedelta(days=15)).isoformat()
            if np.random.rand() > 0.3 else None
        }
    }
    
    # Maneuver history
    maneuver_history = [
        {
            'timestamp': ts,
            'delta_v': np.random.randn(3),
            'type': 'station-keeping' if np.random.rand() > 0.8 else 'unknown',
            'confidence': np.random.rand()
        }
        for ts in timestamps[::20]  # Every 20th timestamp
    ]
    
    # System interactions
    system_interactions = {
        'rf_systems': [
            {
                'interaction_time': timestamps[i],
                'interrogation_frequency': 1000 + np.random.randn(),
                'response_detected': bool(np.random.rand() > 0.7)
            }
            for i in range(0, len(timestamps), 25)
        ],
        'optical_systems': [
            {
                'interaction_time': timestamps[i],
                'laser_wavelength': 1064 + np.random.randn(),
                'response_detected': bool(np.random.rand() > 0.8)
            }
            for i in range(0, len(timestamps), 30)
        ]
    }
    
    return {
        'object_id': object_id,
        'orbital_features': orbital_features,
        'trajectory_features': trajectory_features,
        'historical_stability': historical_stability,
        'rf_emissions': rf_emissions,
        'signature_features': signature_features,
        'orbital_elements': orbital_elements,
        'space_environment': space_environment,
        'launch_data': launch_data,
        'compliance_data': compliance_data,
        'maneuver_history': maneuver_history,
        'system_interactions': system_interactions
    } 