"""Test suite for UDL data integration with atmospheric transit detection."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import numpy as np

from models.atmospheric_transit import (
    AtmosphericTransitDetector,
    UDLDataIntegrator,
    GeophysicalData,
    SDRMeasurement,
    TransitObject,
    TransitType
)
from models.config.atmospheric_transit_config import DETECTOR_CONFIG

@pytest.fixture
def mock_udl_client():
    """Create a mock UDL client."""
    client = MagicMock()
    
    # Mock space weather data
    client.get_space_weather_data.return_value = {
        'tec_data': [
            {
                'time': datetime.utcnow().isoformat(),
                'tec_value': 25.5,
                'b_field_x': 100.0,
                'b_field_y': -50.0,
                'b_field_z': 25.0,
                'latitude': 45.0,
                'longitude': -75.0,
                'altitude': 250.0,
                'confidence': 0.9
            }
        ]
    }
    
    # Mock RF interference data
    client.get_rf_interference.return_value = {
        'measurements': [
            {
                'time': datetime.utcnow().isoformat(),
                'frequency': 300e6,
                'power_level': -60.0,
                'doppler_shift': 150.0,
                'latitude': 45.0,
                'longitude': -75.0,
                'altitude': 250.0,
                'confidence': 0.85
            }
        ]
    }
    
    # Mock state vector data
    current_time = datetime.utcnow()
    client.get_state_vector.return_value = {
        'epoch': current_time.isoformat(),
        'xpos': 6771000.0,  # ~400km altitude
        'ypos': 0.0,
        'zpos': 0.0,
        'xvel': 7.8,  # Orbital velocity
        'yvel': 0.0,
        'zvel': 0.0,
        'uct': True
    }
    
    client.get_state_vector_history.return_value = [
        {
            'epoch': (current_time - timedelta(minutes=5)).isoformat(),
            'xpos': 6771000.0,
            'ypos': 0.0,
            'zpos': 0.0,
            'xvel': 7.8,
            'yvel': 0.0,
            'zvel': 0.0
        },
        {
            'epoch': current_time.isoformat(),
            'xpos': 6771000.0 + 7.8 * 300,  # 5 minutes of motion
            'ypos': 0.0,
            'zpos': 0.0,
            'xvel': 7.8,
            'yvel': 0.0,
            'zvel': 0.0
        }
    ]
    
    return client

@pytest.fixture
def detector(mock_udl_client):
    """Create an atmospheric transit detector with mock UDL client."""
    return AtmosphericTransitDetector(DETECTOR_CONFIG, mock_udl_client)

def test_ionospheric_data_integration(detector):
    """Test integration of ionospheric data from UDL."""
    geo_data = detector.udl_integrator.get_ionospheric_data()
    
    assert len(geo_data) > 0
    for data in geo_data:
        assert isinstance(data, GeophysicalData)
        assert data.ionospheric_tec > 0
        assert all(k in data.magnetic_field for k in ['x', 'y', 'z'])
        assert all(k in data.location for k in ['lat', 'lon', 'alt'])
        assert 0 <= data.confidence <= 1

def test_sdr_data_integration(detector):
    """Test integration of SDR data from UDL."""
    frequency_range = {
        'min': DETECTOR_CONFIG['sdr']['frequency_bands'][0]['min_freq'],
        'max': DETECTOR_CONFIG['sdr']['frequency_bands'][-1]['max_freq']
    }
    
    sdr_data = detector.udl_integrator.get_sdr_data(frequency_range)
    
    assert len(sdr_data) > 0
    for data in sdr_data:
        assert isinstance(data, SDRMeasurement)
        assert frequency_range['min'] <= data.frequency <= frequency_range['max']
        assert data.power <= 0  # dBm should be negative
        assert all(k in data.location for k in ['lat', 'lon', 'alt'])
        assert 0 <= data.confidence <= 1

def test_state_vector_integration(detector):
    """Test integration of state vector data from UDL."""
    state_data = detector.udl_integrator.get_state_vector_data('test-object')
    
    assert 'current' in state_data
    assert 'history' in state_data
    assert len(state_data['history']) >= 2
    
    current = state_data['current']
    assert all(k in current for k in ['epoch', 'xpos', 'ypos', 'zpos', 'xvel', 'yvel', 'zvel'])
    
    for state in state_data['history']:
        assert all(k in state for k in ['epoch', 'xpos', 'ypos', 'zpos', 'xvel', 'yvel', 'zvel'])

def test_trajectory_analysis(detector):
    """Test trajectory analysis from state vector data."""
    state_data = detector.udl_integrator.get_state_vector_data('test-object')
    transit_obj = detector.udl_integrator.analyze_trajectory(state_data)
    
    assert isinstance(transit_obj, TransitObject)
    assert all(k in transit_obj.velocity for k in ['vx', 'vy', 'vz'])
    assert all(k in transit_obj.location for k in ['lat', 'lon', 'alt'])
    assert DETECTOR_CONFIG['altitude_range']['min'] <= transit_obj.location['alt'] <= DETECTOR_CONFIG['altitude_range']['max']
    assert 0 <= transit_obj.confidence <= 1

def test_combined_detection(detector):
    """Test combined detection using all UDL data sources."""
    transit_objects = detector.detect_transits_with_udl('test-object')
    
    assert len(transit_objects) > 0
    for obj in transit_objects:
        assert isinstance(obj, TransitObject)
        assert isinstance(obj.transit_type, TransitType)
        assert obj.confidence >= DETECTOR_CONFIG['confidence_threshold']
        
        # Check velocity magnitude
        v_mag = np.sqrt(
            obj.velocity['vx']**2 +
            obj.velocity['vy']**2 +
            obj.velocity['vz']**2
        )
        assert v_mag >= DETECTOR_CONFIG['velocity_threshold']
        
        # Check altitude range
        assert DETECTOR_CONFIG['altitude_range']['min'] <= obj.location['alt'] <= DETECTOR_CONFIG['altitude_range']['max']
        
        # Check impact prediction for reentry objects
        if obj.transit_type == TransitType.REENTRY:
            assert obj.predicted_impact is not None
            assert 'time' in obj.predicted_impact
            assert 'location' in obj.predicted_impact
            assert 'uncertainty_radius_km' in obj.predicted_impact
            assert obj.predicted_impact['confidence'] >= DETECTOR_CONFIG['impact']['min_confidence']

def test_detection_without_udl(detector):
    """Test detection without UDL client."""
    detector_no_udl = AtmosphericTransitDetector(DETECTOR_CONFIG)
    transit_objects = detector_no_udl.detect_transits_with_udl()
    
    assert len(transit_objects) == 0  # Should return empty list without UDL client 