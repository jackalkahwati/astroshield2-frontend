"""
Test suite for Kill Chain Event Detection module
"""

import pytest
from datetime import datetime
from analysis.kill_chain_detector import KillChainDetector, KillChainEvent

@pytest.fixture
def detector():
    return KillChainDetector()

@pytest.fixture
def mock_telemetry_data():
    return {
        'launch_signatures': [{
            'object_id': 'SAT123',
            'confidence': 0.95,
            'type': 'THERMAL',
            'location': {'lat': 45.0, 'lon': -120.0},
            'velocity_profile': {'v0': 0, 'v1': 7.8}
        }],
        'reentry_candidates': [{
            'object_id': 'DEB456',
            'confidence': 0.88,
            'altitude': 85.0,
            'velocity': 7.2,
            'atmospheric_data': {'density': 1e-5, 'temperature': 200}
        }],
        'orbital_states': [{
            'object_id': 'SAT789',
            'delta_v': 0.15,
            'maneuver_type': 'INCLINATION_CHANGE',
            'orbital_change': {'delta_i': 0.5}
        }],
        'attitude_data': [{
            'object_id': 'SAT101',
            'rotation_delta': 10.5,
            'angular_velocity': 0.02,
            'stabilization': 'ACTIVE'
        }],
        'communication_links': [{
            'object_id': 'SAT202',
            'frequency_delta': 1500,
            'bandwidth_delta': 150,
            'modulation_change': True
        }],
        'proximity_data': [{
            'object_id': 'SAT303',
            'target_id': 'SAT404',
            'min_distance': 50.0,
            'rel_velocity': 0.1,
            'conjunction_prob': 0.02
        }],
        'separation_data': [{
            'parent_id': 'SAT505',
            'child_id': 'SAT606',
            'sep_velocity': 0.08,
            'mass_delta': -100,
            'trajectory_div': 2.5
        }]
    }

@pytest.fixture
def mock_historical_data():
    return {
        'orbital_states': [{
            'object_id': 'SAT789',
            'epoch': (datetime.now().timestamp() - 3600),
            'state_vector': [42164, 0, 0, 0, 3.075, 0]
        }]
    }

@pytest.fixture
def mock_space_environment():
    return {
        'radiation_belts': {
            'inner': {'flux': 1e5},
            'outer': {'flux': 1e6}
        },
        'debris_density': {
            'leo': 0.1,
            'meo': 0.01,
            'geo': 0.001
        }
    }

@pytest.mark.asyncio
async def test_detect_events_all_types(detector, mock_telemetry_data, mock_historical_data, mock_space_environment):
    """Test detection of all event types"""
    events = await detector.detect_events(mock_telemetry_data, mock_historical_data, mock_space_environment)
    
    # Verify we detect all event types
    event_types = {event.event_type for event in events}
    expected_types = {'LAUNCH', 'REENTRY', 'MANEUVER', 'ATTITUDE', 'LINK_MOD', 'PROXIMITY', 'SEPARATION'}
    assert event_types == expected_types
    
    # Verify event count
    assert len(events) == 7  # One of each type

@pytest.mark.asyncio
async def test_launch_detection(detector, mock_telemetry_data):
    """Test launch event detection"""
    events = detector._detect_launches(mock_telemetry_data)
    assert len(events) == 1
    
    event = events[0]
    assert event.event_type == 'LAUNCH'
    assert event.source_object == 'SAT123'
    assert event.confidence > 0.9
    assert 'velocity_profile' in event.evidence

@pytest.mark.asyncio
async def test_reentry_detection(detector, mock_telemetry_data):
    """Test reentry event detection"""
    events = detector._detect_reentries(mock_telemetry_data)
    assert len(events) == 1
    
    event = events[0]
    assert event.event_type == 'REENTRY'
    assert event.source_object == 'DEB456'
    assert 'altitude' in event.evidence
    assert 'atmospheric_interaction' in event.evidence

@pytest.mark.asyncio
async def test_maneuver_detection(detector, mock_telemetry_data, mock_historical_data):
    """Test maneuver event detection"""
    events = detector._detect_maneuvers(mock_telemetry_data, mock_historical_data)
    assert len(events) == 1
    
    event = events[0]
    assert event.event_type == 'MANEUVER'
    assert event.source_object == 'SAT789'
    assert event.evidence['delta_v'] > 0.1
    assert 'maneuver_type' in event.evidence

@pytest.mark.asyncio
async def test_attitude_change_detection(detector, mock_telemetry_data):
    """Test attitude change event detection"""
    events = detector._detect_attitude_changes(mock_telemetry_data)
    assert len(events) == 1
    
    event = events[0]
    assert event.event_type == 'ATTITUDE'
    assert event.source_object == 'SAT101'
    assert abs(event.evidence['rotation_change']) > 5.0

@pytest.mark.asyncio
async def test_link_modification_detection(detector, mock_telemetry_data):
    """Test link modification event detection"""
    events = detector._detect_link_modifications(mock_telemetry_data)
    assert len(events) == 1
    
    event = events[0]
    assert event.event_type == 'LINK_MOD'
    assert event.source_object == 'SAT202'
    assert abs(event.evidence['frequency_change']) > 1000

@pytest.mark.asyncio
async def test_proximity_event_detection(detector, mock_telemetry_data, mock_space_environment):
    """Test proximity event detection"""
    events = detector._detect_proximity_events(mock_telemetry_data, mock_space_environment)
    assert len(events) == 1
    
    event = events[0]
    assert event.event_type == 'PROXIMITY'
    assert event.source_object == 'SAT303'
    assert event.target_object == 'SAT404'
    assert event.evidence['minimum_distance'] < 100

@pytest.mark.asyncio
async def test_separation_detection(detector, mock_telemetry_data):
    """Test separation event detection"""
    events = detector._detect_separations(mock_telemetry_data)
    assert len(events) == 1
    
    event = events[0]
    assert event.event_type == 'SEPARATION'
    assert event.source_object == 'SAT505'
    assert event.target_object == 'SAT606'
    assert event.evidence['separation_velocity'] > 0.05

@pytest.mark.asyncio
async def test_validation_methods(detector):
    """Test validation methods for each event type"""
    # Launch validation
    assert detector._validate_launch_signature({
        'confidence': 0.9,
        'velocity_profile': {'v0': 0, 'v1': 7.8}
    }) == True
    
    # Reentry validation
    assert detector._validate_reentry_profile({
        'altitude': 90,
        'atmospheric_data': {'density': 1e-5}
    }) == True
    
    # Maneuver validation
    assert detector._validate_maneuver({
        'delta_v': 0.15
    }, []) == True
    
    # Attitude change validation
    assert detector._validate_attitude_change({
        'rotation_delta': 10.0
    }) == True
    
    # Link modification validation
    assert detector._validate_link_modification({
        'frequency_delta': 1500,
        'bandwidth_delta': 150,
        'modulation_change': True
    }) == True
    
    # Proximity event validation
    assert detector._validate_proximity_event({
        'min_distance': 50,
        'conjunction_prob': 0.02
    }, {}) == True
    
    # Separation validation
    assert detector._validate_separation({
        'sep_velocity': 0.08,
        'trajectory_div': 1.5
    }) == True
