"""Test suite for UDL client functionality."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from api_client.udl_client import UDLClient
from api_client.config import (
    SPACE_WEATHER_CONFIG,
    CONJUNCTION_CONFIG,
    RF_CONFIG,
    ORBITAL_CONFIG,
    MANEUVER_CONFIG,
    LINK_STATUS_CONFIG,
    HEALTH_CONFIG
)

@pytest.fixture
def udl_client():
    """Create a UDL client for testing."""
    client = UDLClient('https://unifieddatalibrary.com/udl')
    client.api_key = "test_api_key"  # Set API key in fixture
    return client

@pytest.fixture
def mock_response():
    """Create a mock response object."""
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    return mock

def test_sensor_status(udl_client, mock_response):
    """Test getting sensor status."""
    with patch('requests.Session.get') as mock_get:
        # Setup mock
        mock_response.json.return_value = {
            'status': 'OPERATIONAL',
            'lastUpdate': datetime.utcnow().isoformat(),
            'health': {
                'battery': 0.95,
                'temperature': 25.0,
                'memory': 0.75
            }
        }
        mock_get.return_value = mock_response
        
        # Test the endpoint
        result = udl_client.get_sensor_status('SENSOR-001')
        
        # Verify the call
        mock_get.assert_called_once()
        assert 'sensor/SENSOR-001' in mock_get.call_args[0][0]
        
        # Verify the response
        assert result['status'] == 'OPERATIONAL'
        assert 'lastUpdate' in result
        assert 'health' in result

def test_sensor_maintenance(udl_client, mock_response):
    """Test getting sensor maintenance information."""
    with patch('requests.Session.get') as mock_get:
        # Setup mock
        mock_response.json.return_value = {
            'lastMaintenance': (datetime.utcnow() - timedelta(days=30)).isoformat(),
            'nextMaintenance': (datetime.utcnow() + timedelta(days=60)).isoformat(),
            'maintenanceHistory': [
                {
                    'date': (datetime.utcnow() - timedelta(days=30)).isoformat(),
                    'type': 'SCHEDULED',
                    'description': 'Regular maintenance'
                }
            ]
        }
        mock_get.return_value = mock_response
        
        # Test the endpoint
        result = udl_client.get_sensor_maintenance('SENSOR-001')
        
        # Verify the call
        mock_get.assert_called_once()
        assert 'sensormaintenance' in mock_get.call_args[0][0]
        
        # Verify the response
        assert 'lastMaintenance' in result
        assert 'nextMaintenance' in result
        assert 'maintenanceHistory' in result

def test_sensor_calibration(udl_client, mock_response):
    """Test getting sensor calibration information."""
    with patch('requests.Session.get') as mock_get:
        # Setup mock
        mock_response.json.return_value = {
            'lastCalibration': datetime.utcnow().isoformat(),
            'calibrationStatus': 'VALID',
            'parameters': {
                'gain': 1.0,
                'offset': 0.0,
                'temperature_coefficient': 0.001
            }
        }
        mock_get.return_value = mock_response
        
        # Test the endpoint
        result = udl_client.get_sensor_calibration('SENSOR-001')
        
        # Verify the call
        mock_get.assert_called_once()
        assert 'sensorcalibration' in mock_get.call_args[0][0]
        
        # Verify the response
        assert result['calibrationStatus'] == 'VALID'
        assert 'parameters' in result

def test_link_status(udl_client, mock_response):
    """Test getting link status information."""
    with patch('requests.Session.get') as mock_get:
        # Setup mock
        mock_response.json.return_value = {
            'status': 'STRONG',
            'signalStrength': -95.0,  # Updated to be below threshold
            'bitErrorRate': 1e-9,
            'latency': 250,
            'lastUpdate': datetime.utcnow().isoformat()
        }
        mock_get.return_value = mock_response
        
        # Test the endpoint
        result = udl_client.get_link_status('SAT-001')
        
        # Verify the call
        mock_get.assert_called_once()
        assert 'linkstatus' in mock_get.call_args[0][0]
        
        # Verify the response
        assert result['status'] == 'STRONG'
        assert result['signalStrength'] < LINK_STATUS_CONFIG['signal_strength_threshold']
        assert result['bitErrorRate'] < LINK_STATUS_CONFIG['bit_error_rate_threshold']

def test_batch_operations(udl_client, mock_response):
    """Test batch operations for multiple objects."""
    with patch('requests.Session.get') as mock_get:
        # Setup mock
        mock_response.json.return_value = {
            'objects': [
                {
                    'id': 'SAT-001',
                    'status': 'OPERATIONAL',
                    'health': 0.95
                },
                {
                    'id': 'SAT-002',
                    'status': 'DEGRADED',
                    'health': 0.75
                }
            ]
        }
        mock_get.return_value = mock_response
        
        # Test the endpoint
        result = udl_client.get_multiple_object_status(['SAT-001', 'SAT-002'])
        
        # Verify the call
        mock_get.assert_called_once()
        assert 'onorbit' in mock_get.call_args[0][0]
        
        # Verify the response
        assert len(result['objects']) == 2
        assert result['objects'][0]['status'] == 'OPERATIONAL'
        assert result['objects'][1]['status'] == 'DEGRADED'

def test_system_health_summary(udl_client, mock_response):
    """Test getting system health summary."""
    with patch('requests.Session.get') as mock_get:
        # Setup mock responses for different endpoints
        mock_responses = {
            '/udl/sgi': {'kp_index': 3.0, 'radiation_level': 'LOW'},
            '/udl/conjunction': {'active_conjunctions': 0},
            '/udl/linkstatus': {'overall_status': 'HEALTHY'},
            '/udl/rfemitter': {'interference_detected': False}
        }
        
        def mock_get_response(*args, **kwargs):
            endpoint = args[0]
            for key in mock_responses:
                if key in endpoint:
                    mock_response.json.return_value = mock_responses[key]
                    return mock_response
            return mock_response
        
        mock_get.side_effect = mock_get_response
        
        # Test the endpoint
        result = udl_client.get_system_health_summary()
        
        # Verify calls were made to all endpoints
        assert mock_get.call_count == 4
        
        # Verify the response contains all components
        assert '/udl/sgi' in result
        assert '/udl/conjunction' in result
        assert '/udl/linkstatus' in result
        assert '/udl/rfemitter' in result

def test_error_handling(udl_client):
    """Test error handling in UDL client."""
    with patch('requests.Session.get') as mock_get:
        # Simulate connection error
        mock_get.side_effect = ConnectionError("Failed to connect")
        
        with pytest.raises(ConnectionError):
            udl_client.get_sensor_status('SENSOR-001')
        
        # Simulate API error
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("API Error")
        mock_get.side_effect = None
        mock_get.return_value = mock_response
        
        with pytest.raises(Exception):
            udl_client.get_sensor_status('SENSOR-001')

def test_authentication(udl_client, mock_response):
    """Test API authentication."""
    with patch('requests.Session.get') as mock_get:
        # Setup mock
        mock_get.return_value = mock_response
        
        # Test with API key
        udl_client.get_sensor_status('SENSOR-001')
        
        # Verify Authorization header was set
        assert 'Authorization' in udl_client.session.headers
        assert udl_client.session.headers['Authorization'] == f'Bearer {udl_client.api_key}' 