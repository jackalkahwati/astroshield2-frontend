"""Test suite for UDL integration features."""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any

from asttroshield.api_client.udl_client import UDLClient
from asttroshield.models.threat_indicators import UCTAnalyzer, BOGEYScorer

@pytest.fixture
def udl_client():
    """Create a UDL client for testing."""
    return UDLClient('https://unifieddatalibrary.com/udl', api_key='test-key')

@pytest.fixture
def mock_elset_data():
    """Create mock ELSET data."""
    return [
        {
            'epoch': (datetime.utcnow() - timedelta(hours=i)).isoformat(),
            'mean_motion': 1.0027,
            'eccentricity': 0.0001,
            'inclination': 0.05,
            'raan': 180.0,
            'arg_perigee': 0.0,
            'mean_anomaly': 45.0
        }
        for i in range(24)
    ]

@pytest.fixture
def mock_sgp4xp_data():
    """Create mock SGP4-XP TLE data."""
    return {
        'line1': '1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927',
        'line2': '2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537',
        'force_model': {
            'drag': True,
            'srp': True,
            'gravity': 20,
            'third_body': ['MOON', 'SUN']
        }
    }

@pytest.mark.integration
@pytest.mark.elset
def test_elset_history_retrieval(udl_client, mock_elset_data):
    """Test retrieval of ELSET history."""
    result = udl_client.get_elset_history(
        'TEST-SAT-001',
        (datetime.utcnow() - timedelta(days=1)).isoformat(),
        datetime.utcnow().isoformat()
    )
    assert isinstance(result, list)

@pytest.mark.integration
@pytest.mark.sgp4xp
def test_sgp4xp_tle_retrieval(udl_client, mock_sgp4xp_data):
    """Test retrieval of SGP4-XP force model TLE."""
    result = udl_client.get_sgp4xp_tle('TEST-SAT-001')
    assert isinstance(result, dict)

@pytest.mark.integration
@pytest.mark.orbit_determination
def test_orbit_determination_retrieval(udl_client):
    """Test retrieval of orbit determination data."""
    result = udl_client.get_orbit_determination('TEST-SAT-001')
    assert isinstance(result, dict)

@pytest.mark.integration
@pytest.mark.notifications
def test_geo_notification_creation(udl_client):
    """Test creation of GEO proximity notifications."""
    result = udl_client.create_geo_notification(
        'TEST-SAT-001',
        {
            'distance_to_geo': 45.0,
            'threshold': 50.0,
            'position': {'x': 42164.0, 'y': 10.0, 'z': 0.0}
        }
    )
    assert isinstance(result, dict)

@pytest.mark.integration
@pytest.mark.visual_magnitude
def test_visual_magnitude_retrieval(udl_client):
    """Test retrieval of normalized visual magnitude."""
    result = udl_client.get_visual_magnitude('TEST-SAT-001')
    assert isinstance(result, dict)

@pytest.mark.integration
@pytest.mark.state_accuracy
def test_state_accuracy_retrieval(udl_client):
    """Test retrieval of state accuracy data."""
    result = udl_client.get_state_accuracy('TEST-SAT-001')
    assert isinstance(result, dict)

@pytest.mark.integration
@pytest.mark.environmental
def test_environmental_data_integration(udl_client):
    """Test integration of environmental data in UCT analysis."""
    analyzer = UCTAnalyzer()
    result = analyzer._analyze_environment(
        {},  # illumination_data
        None,  # lunar_data
        {'kp_index': 6.0},  # space_weather
        {'level': 4}  # radiation_belt
    )
    assert isinstance(result, dict)
    assert 'solar_activity' in result
    assert 'radiation_belt_activity' in result

@pytest.mark.integration
@pytest.mark.confidence
def test_confidence_calculation_with_udl_data(udl_client):
    """Test confidence calculation with UDL data integration."""
    analyzer = UCTAnalyzer()
    result = analyzer._calculate_confidence({
        'ccd_indicators': [
            {'type': 'signature_management'},
            {'type': 'unusual_maneuver'}
        ],
        'environmental_factors': {
            'solar_activity': 'high',
            'radiation_belt_activity': 'high'
        },
        'sensor_correlations': 2,
        'confidence': 0.9
    })
    assert isinstance(result, float)
    assert 0 <= result <= 1.0
    assert result < 0.9  # Environmental factors should reduce confidence 