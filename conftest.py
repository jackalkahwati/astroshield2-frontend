"""
Pytest configuration file
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timedelta

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Define fixtures that can be used across all tests
@pytest.fixture
def mock_space_object_data():
    """
    Fixture that provides mock space object data for testing.
    """
    return {
        "object_id": "12345",
        "state_vector": {
            "position": {"x": 1000.0, "y": 2000.0, "z": 3000.0},
            "velocity": {"x": 1.0, "y": 2.0, "z": 3.0},
            "epoch": datetime.utcnow().isoformat()
        },
        "state_history": [
            {
                "time": (datetime.utcnow() - timedelta(hours=1)).isoformat(),
                "position": {"x": 900.0, "y": 1900.0, "z": 2900.0},
                "velocity": {"x": 0.9, "y": 1.9, "z": 2.9}
            },
            {
                "time": (datetime.utcnow() - timedelta(hours=2)).isoformat(),
                "position": {"x": 800.0, "y": 1800.0, "z": 2800.0},
                "velocity": {"x": 0.8, "y": 1.8, "z": 2.8}
            }
        ],
        "maneuver_history": [
            {
                "time": (datetime.utcnow() - timedelta(days=1)).isoformat(),
                "delta_v": 0.5,
                "thrust_vector": {"x": 0.1, "y": 0.2, "z": 0.3},
                "duration": 60.0,
                "confidence": 0.85
            },
            {
                "time": (datetime.utcnow() - timedelta(days=7)).isoformat(),
                "delta_v": 0.8,
                "thrust_vector": {"x": -0.1, "y": 0.3, "z": 0.2},
                "duration": 90.0,
                "confidence": 0.9
            }
        ],
        "rf_history": [
            {
                "time": (datetime.utcnow() - timedelta(hours=6)).isoformat(),
                "frequency": 2200.0,
                "power": -90.0,
                "duration": 300.0,
                "bandwidth": 5.0,
                "confidence": 0.75
            }
        ],
        "radar_signature": {
            "rcs": 1.5,
            "timestamp": datetime.utcnow().isoformat(),
            "sensor_id": "radar-001",
            "confidence": 0.9
        },
        "optical_signature": {
            "magnitude": 6.5,
            "timestamp": datetime.utcnow().isoformat(),
            "sensor_id": "optical-002",
            "confidence": 0.85
        },
        "baseline_signatures": {
            "radar": {
                "rcs_mean": 1.2,
                "rcs_std": 0.3,
                "rcs_min": 0.5,
                "rcs_max": 2.0
            },
            "optical": {
                "magnitude_mean": 7.0,
                "magnitude_std": 0.5,
                "magnitude_min": 6.0,
                "magnitude_max": 8.0
            }
        },
        "baseline_pol": {
            "rf": {
                "max_power": -95.0,
                "frequencies": [2200.0, 8400.0],
                "duty_cycles": [0.1, 0.05]
            },
            "maneuvers": {
                "typical_delta_v": 0.3,
                "typical_intervals": [14, 30]
            }
        },
        "orbit_data": {
            "semi_major_axis": 7000.0,
            "eccentricity": 0.001,
            "inclination": 51.6,
            "raan": 120.0,
            "arg_perigee": 180.0,
            "mean_anomaly": 0.0,
            "mean_motion": 15.5
        },
        "parent_orbit_data": {
            "parent_object_id": "12340",
            "semi_major_axis": 7000.0,
            "inclination": 51.6,
            "eccentricity": 0.001
        },
        "population_data": {
            "orbit_regime": "LEO",
            "density": 15.0,
            "mean_amr": 0.01,
            "std_amr": 0.003
        },
        "anomaly_baseline": {
            "thermal_profile": [270, 280, 275, 265],
            "maneuver_frequency": 0.06
        },
        "object_events": [
            {
                "type": "maneuver",
                "time": (datetime.utcnow() - timedelta(days=1)).isoformat(),
                "data": {"delta_v": 0.5}
            },
            {
                "type": "rf_emission",
                "time": (datetime.utcnow() - timedelta(hours=6)).isoformat(),
                "data": {"frequency": 2200.0, "power": -90.0}
            },
            {
                "type": "conjunction",
                "time": (datetime.utcnow() - timedelta(days=2)).isoformat(),
                "data": {"miss_distance": 45.0, "secondary_object": "23456"}
            }
        ],
        "filing_data": {
            "itu_filing": "ABC123",
            "authorized_frequencies": [2200.0, 8400.0]
        },
        "registry_data": {
            "registered_ids": ["12345"],
            "registry_authority": "UNOOSA"
        }
    }

@pytest.fixture
def mock_udl_client():
    """
    Fixture that provides a mock UDL client.
    """
    client = MagicMock()
    
    # Set up mock return values for common methods
    client.get_state_vector.return_value = {
        "position": {"x": 1000.0, "y": 2000.0, "z": 3000.0},
        "velocity": {"x": 1.0, "y": 2.0, "z": 3.0},
        "epoch": datetime.utcnow().isoformat()
    }
    
    client.get_state_vector_history.return_value = [
        {
            "time": (datetime.utcnow() - timedelta(hours=i)).isoformat(),
            "position": {"x": 1000.0 - i*100, "y": 2000.0 - i*100, "z": 3000.0 - i*100},
            "velocity": {"x": 1.0 - i*0.1, "y": 2.0 - i*0.1, "z": 3.0 - i*0.1}
        } for i in range(1, 24)
    ]
    
    client.get_elset_data.return_value = {
        "semi_major_axis": 7000.0,
        "eccentricity": 0.001,
        "inclination": 51.6,
        "raan": 120.0,
        "arg_perigee": 180.0,
        "mean_anomaly": 0.0,
        "mean_motion": 15.5
    }
    
    client.get_maneuver_data.return_value = [
        {
            "time": (datetime.utcnow() - timedelta(days=i*7)).isoformat(),
            "delta_v": 0.5 + (i*0.1),
            "thrust_vector": {"x": 0.1, "y": 0.2, "z": 0.3},
            "duration": 60.0 + (i*10),
            "confidence": 0.85
        } for i in range(0, 3)
    ]
    
    client.get_conjunction_data.return_value = {
        "events": [
            {
                "timestamp": (datetime.utcnow() - timedelta(days=i)).isoformat(),
                "miss_distance": 40.0 + (i*5),
                "probability": 1e-6 * (i+1)
            } for i in range(0, 5)
        ]
    }
    
    client.get_rf_interference.return_value = {
        "measurements": [
            {
                "timestamp": (datetime.utcnow() - timedelta(hours=i*6)).isoformat(),
                "frequency": 2200.0,
                "power_level": -90.0 - (i*2),
                "duration": 300.0,
                "bandwidth": 5.0
            } for i in range(0, 4)
        ]
    }
    
    return client

@pytest.fixture
def mock_kafka_consumer():
    """
    Fixture that provides a mock Kafka consumer.
    """
    consumer = MagicMock()
    
    # Setup for async context manager behavior
    async def mock_aenter(self):
        return self
    
    async def mock_aexit(self, exc_type, exc_val, exc_tb):
        pass
    
    consumer.__aenter__ = mock_aenter
    consumer.__aexit__ = mock_aexit
    
    # Setup async receive method
    async def mock_receive():
        return {
            "topic": "test-topic",
            "value": {
                "object_id": "12345",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {"position": {"x": 1000.0, "y": 2000.0, "z": 3000.0}}
            }
        }
    
    consumer.receive = AsyncMock(side_effect=mock_receive)
    
    return consumer
