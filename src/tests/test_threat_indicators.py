"""Test suite for threat indicator models using pytest."""

import pytest
from datetime import datetime, timedelta
import json
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock
import numpy as np

from src.models.threat_indicators import (
    StabilityIndicator,
    ManeuverIndicator,
    RFIndicator,
    SignatureAnalyzer,
    OrbitAnalyzer,
    LaunchAnalyzer,
    RegistryChecker,
    AMRAnalyzer,
    StimulationAnalyzer,
    ImagingManeuverAnalyzer,
    DebrisAnalyzer,
    UCTAnalyzer,
    BOGEYScorer
)

# Fixtures for test data
@pytest.fixture
def test_object_id():
    return 'TEST-SAT-001'

@pytest.fixture
def time_window():
    return {
        'start_time': (datetime.utcnow() - timedelta(hours=24)).isoformat(),
        'end_time': datetime.utcnow().isoformat()
    }

@pytest.fixture
def mock_state_history():
    return [
        {
            'epoch': (datetime.utcnow() - timedelta(minutes=i)).isoformat(),
            'position': {'x': 42164.0 + (0.1 if i == 5 else 0), 
                       'y': 0.0, 
                       'z': 0.0}
        }
        for i in range(10)
    ]

@pytest.fixture
def mock_maneuver_history():
    return [
        {
            'time': (datetime.utcnow() - timedelta(hours=i)).isoformat(),
            'delta_v': 0.15 if i == 2 else 0.05,
            'final_position': {
                'x': 42164.0 + (100.0 if i == 2 else 0),
                'y': 0.0,
                'z': 0.0
            }
        }
        for i in range(5)
    ]

@pytest.fixture
def mock_coverage_gaps():
    return [
        {
            'start': (datetime.utcnow() - timedelta(hours=2)).isoformat(),
            'end': (datetime.utcnow() - timedelta(hours=1)).isoformat()
        }
    ]

@pytest.fixture
def mock_rf_history():
    return [
        {
            'time': (datetime.utcnow() - timedelta(minutes=i)).isoformat(),
            'frequency': 6000 + (i * 100),
            'power_level': -75.0
        }
        for i in range(5)
    ]

@pytest.fixture
def mock_baseline_pol():
    return {
        'maneuver_windows': [
            {
                'start': (datetime.utcnow() - timedelta(hours=3)).isoformat(),
                'end': (datetime.utcnow() - timedelta(hours=2)).isoformat()
            }
        ],
        'rf_windows': [
            {
                'start': (datetime.utcnow() - timedelta(hours=1)).isoformat(),
                'end': datetime.utcnow().isoformat()
            }
        ]
    }

@pytest.fixture
def mock_udl_data():
    return {
        'track_data': {
            'time': datetime.utcnow().isoformat(),
            'magnitude': 16.5,
            'orbit_type': 'non_standard',
            'near_event': True
        },
        'illumination_data': {
            'eclipse_periods': []
        },
        'lunar_data': {
            'phase': 0.3,
            'elevation': 15.0
        },
        'space_weather': {
            'kp_index': 6.0,
            'dst_index': -100,
            'f10_7': 150
        },
        'radiation_belt': {
            'level': 4,
            'location': 'inner',
            'flux': 1e5
        },
        'rf_interference': {
            'interference_detected': True,
            'frequency_band': 'S',
            'power_level': -60
        },
        'state_history': [
            {
                'time': (datetime.utcnow() - timedelta(minutes=5)).isoformat(),
                'velocity': {'x': 0.1, 'y': 0.0, 'z': 0.0}
            },
            {
                'time': datetime.utcnow().isoformat(),
                'velocity': {'x': 0.3, 'y': 0.0, 'z': 0.0}
            }
        ],
        'sensor_data': [
            {
                'time': datetime.utcnow().isoformat(),
                'sensor_id': 'SENSOR_1',
                'quality': 0.9
            },
            {
                'time': (datetime.utcnow() + timedelta(seconds=30)).isoformat(),
                'sensor_id': 'SENSOR_2',
                'quality': 0.85
            }
        ]
    }

@pytest.fixture
def mock_visual_magnitude():
    return {
        'magnitude': 15.5,
        'uncertainty': 0.2,
        'normalized_range_km': 40000
    }

@pytest.fixture
def mock_state_accuracy():
    return {
        'rms': 0.5,  # km
        'radial': 0.3,
        'in_track': 0.4,
        'cross_track': 0.2
    }

# Test classes with pytest markers
@pytest.mark.stability
def test_stability_analysis(mock_state_history):
    """Test stability analysis model."""
    model = StabilityIndicator()
    results = model.analyze_stability(mock_state_history)
    
    assert isinstance(results['is_stable'], bool)
    assert isinstance(results['confidence'], float)
    assert isinstance(results['changes_detected'], list)
    assert 0 <= results['confidence'] <= 1.0

@pytest.mark.maneuvers
def test_maneuver_analysis(mock_maneuver_history, mock_coverage_gaps):
    """Test maneuver analysis model."""
    model = ManeuverIndicator()
    results = model.analyze_maneuvers(mock_maneuver_history, mock_coverage_gaps)
    
    assert isinstance(results['is_suspicious'], bool)
    assert isinstance(results['confidence'], float)
    assert isinstance(results['anomalies'], list)
    assert 0 <= results['confidence'] <= 1.0

@pytest.mark.rf
def test_rf_analysis(mock_rf_history, mock_baseline_pol):
    """Test RF pattern analysis model."""
    model = RFIndicator()
    results = model.analyze_rf_pattern(mock_rf_history, mock_baseline_pol)
    
    assert isinstance(results['is_anomalous'], bool)
    assert isinstance(results['confidence'], float)
    assert isinstance(results['anomalies'], list)
    assert 0 <= results['confidence'] <= 1.0

@pytest.mark.uct
@pytest.mark.parametrize("interference_level", [
    "none",
    "high_lunar",
    "high_solar",
    "high_radiation"
])
def test_uct_analysis_with_interference(mock_udl_data, interference_level):
    """Test UCT analysis with different interference levels."""
    uct = UCTAnalyzer()
    
    # Modify UDL data based on interference level
    if interference_level == "high_lunar":
        mock_udl_data['lunar_data']['phase'] = 0.8
        mock_udl_data['lunar_data']['elevation'] = 45.0
    elif interference_level == "high_solar":
        mock_udl_data['space_weather']['kp_index'] = 8.0
    elif interference_level == "high_radiation":
        mock_udl_data['radiation_belt']['level'] = 5
    
    result = uct.analyze_uct(
        mock_udl_data['track_data'],
        mock_udl_data['illumination_data'],
        mock_udl_data['lunar_data'],
        mock_udl_data['sensor_data'],
        mock_udl_data['space_weather'],
        mock_udl_data['radiation_belt'],
        mock_udl_data['rf_interference'],
        mock_udl_data['state_history']
    )
    
    assert 'environmental_factors' in result
    assert 'ccd_indicators' in result
    assert isinstance(result['confidence'], float)
    assert 0 <= result['confidence'] <= 1.0
    
    if interference_level == "high_lunar":
        assert result['environmental_factors']['lunar_interference']
    elif interference_level == "high_solar":
        assert result['environmental_factors']['solar_activity'] == 'high'
    elif interference_level == "high_radiation":
        assert result['environmental_factors']['radiation_belt_activity'] == 'high'

@pytest.mark.bogey
@pytest.mark.parametrize("object_type,expected_score_range", [
    ("typical_satellite", (8.0, 10.0)),
    ("debris", (0.0, 5.0))
])
def test_bogey_scoring(object_type, expected_score_range):
    """Test BOGEY scoring system with different object types."""
    scorer = BOGEYScorer()
    
    if object_type == "typical_satellite":
        test_data = {
            'object_data': {
                'orbit_class': 'GEO',
                'delta_v_to_geo': 5.0,
                'observation_quality': 0.9
            },
            'custody_duration': 30,
            'amr_data': {
                'amr': 0.05
            }
        }
    else:  # debris
        test_data = {
            'object_data': {
                'orbit_class': 'LEO',
                'observation_quality': 0.7
            },
            'custody_duration': 180,
            'amr_data': {
                'amr': 2.5
            }
        }
    
    result = scorer.calculate_bogey_score(
        test_data['object_data'],
        test_data['custody_duration'],
        test_data['amr_data']
    )
    
    assert isinstance(result['bogey_score'], float)
    assert expected_score_range[0] <= result['bogey_score'] <= expected_score_range[1]
    assert 0 <= result['confidence'] <= 1.0

@pytest.mark.debris
@pytest.mark.parametrize("event_type,debris_count,expected_indicators", [
    ("passivation", 30, ["excessive_debris"]),
    ("breakup", 1, ["unusual_amr"])
])
def test_debris_analysis(event_type, debris_count, expected_indicators):
    """Test debris analysis with different event types."""
    analyzer = DebrisAnalyzer()
    
    debris_data = [
        {
            'first_seen': datetime.utcnow().isoformat(),
            'controlled_motion': event_type == "breakup",
            'amr': 0.05 if event_type == "breakup" else 1.5,
            'position': {'x': 42164.0, 'y': 0.0, 'z': 0.0},
            'observation_quality': 0.9
        }
        for _ in range(debris_count)
    ]
    
    parent_data = {
        'id': 'PARENT-001',
        'type': 'ROCKET_BODY' if event_type == "passivation" else 'SATELLITE'
    }
    
    event_context = {
        'type': event_type,
        'time': datetime.utcnow().isoformat(),
        'expected_debris_count': 10 if event_type == "passivation" else 1
    }
    
    result = analyzer.analyze_debris_event(debris_data, parent_data, event_context)
    
    assert isinstance(result['ccd_likelihood'], float)
    assert isinstance(result['confidence'], float)
    assert isinstance(result['indicators'], list)
    
    for indicator in expected_indicators:
        assert any(i['type'] == indicator for i in result['indicators'])
    
    assert 0 <= result['ccd_likelihood'] <= 1.0
    assert 0 <= result['confidence'] <= 1.0

def test_bogey_scoring():
    """Test BOGEY score calculation according to DnD specification."""
    scorer = BOGEYScorer()
    
    # Test case 1: New object with typical satellite AMR
    result = scorer.calculate_bogey_score(
        custody_duration_days=1,
        amr_data={'amr': 0.02},  # Typical satellite AMR
        geo_data=None
    )
    assert 8.0 <= result['bogey_score'] <= 10.0
    assert result['components']['custody_score'] > 9.9  # Almost max score for new object
    
    # Test case 2: Long-tracked object with high AMR
    result = scorer.calculate_bogey_score(
        custody_duration_days=3650,  # 10 years
        amr_data={'amr': 2.0},  # Debris-like AMR
        geo_data=None
    )
    assert 1.0 <= result['bogey_score'] <= 3.0
    assert result['components']['custody_score'] == 1.0  # Min score for 10-year custody
    
    # Test case 3: Near-GEO object
    result = scorer.calculate_bogey_score(
        custody_duration_days=100,
        amr_data={'amr': 0.05},
        geo_data={'delta_v_to_geo': 20.0}  # Very close to GEO
    )
    assert result['geo_score'] > 9.0  # High GEO score for close proximity
    
    # Test case 4: Invalid AMR
    result = scorer.calculate_bogey_score(
        custody_duration_days=100,
        amr_data={'amr': -1.0},  # Invalid AMR
        geo_data=None
    )
    assert result['components']['amr_score'] == 1.0  # Minimum score for invalid AMR
    
    # Test case 5: Far from GEO
    result = scorer.calculate_bogey_score(
        custody_duration_days=100,
        amr_data={'amr': 0.05},
        geo_data={'delta_v_to_geo': 300.0}  # Far from GEO
    )
    assert result['geo_score'] is None  # No GEO score for objects far from GEO

@pytest.mark.uct
@pytest.mark.parametrize("test_case", [
    {
        "name": "geo_notification",
        "orbit_type": "GEO",
        "position": {"x": 42164.0, "y": 10.0, "z": 0.0},
        "expect_notification": True
    },
    {
        "name": "no_notification",
        "orbit_type": "LEO",
        "position": {"x": 7000.0, "y": 0.0, "z": 0.0},
        "expect_notification": False
    }
])
def test_geo_notifications(mock_udl_data, mock_visual_magnitude, mock_state_accuracy, test_case):
    """Test GEO proximity notifications."""
    uct = UCTAnalyzer()
    
    # Update mock data for test case
    mock_udl_data['track_data']['orbit_type'] = test_case['orbit_type']
    mock_udl_data['track_data']['position'] = test_case['position']
    
    result = uct.analyze_uct(
        mock_udl_data['track_data'],
        mock_udl_data['illumination_data'],
        mock_udl_data['lunar_data'],
        mock_udl_data['sensor_data'],
        mock_udl_data['space_weather'],
        mock_udl_data['radiation_belt'],
        mock_udl_data['rf_interference'],
        mock_udl_data['state_history'],
        mock_visual_magnitude,
        mock_state_accuracy,
        'test-orbit-determination-uuid'
    )
    
    # Verify notifications
    if test_case['expect_notification']:
        assert any(n['type'] == 'NEAR_GEO' for n in result['notifications'])
        notification = next(n for n in result['notifications'] if n['type'] == 'NEAR_GEO')
        assert 'distance_to_geo' in notification['data']
        assert notification['data']['distance_to_geo'] < notification['data']['threshold']
    else:
        assert not any(n['type'] == 'NEAR_GEO' for n in result['notifications'])
    
    # Verify new data fields
    assert result['visual_magnitude_40k'] == mock_visual_magnitude['magnitude']
    assert result['state_accuracy_km'] == mock_state_accuracy['rms']
    assert result['orbit_determination_ref'] == 'test-orbit-determination-uuid'

@pytest.mark.uct
def test_motion_analysis_with_accuracy(mock_udl_data, mock_state_accuracy):
    """Test motion analysis with state accuracy consideration."""
    uct = UCTAnalyzer()
    
    # Create state history with changes above and below accuracy threshold
    mock_udl_data['state_history'] = [
        {
            'time': (datetime.utcnow() - timedelta(minutes=i)).isoformat(),
            'velocity': {'x': vel, 'y': 0.0, 'z': 0.0}
        }
        for i, vel in enumerate([0.1, 0.2, 0.8, 0.9])  # 0.6 km/s change
    ]
    
    result = uct.analyze_uct(
        mock_udl_data['track_data'],
        mock_udl_data['illumination_data'],
        mock_udl_data['lunar_data'],
        mock_udl_data['sensor_data'],
        mock_udl_data['space_weather'],
        mock_udl_data['radiation_belt'],
        mock_udl_data['rf_interference'],
        mock_udl_data['state_history'],
        None,
        mock_state_accuracy
    )
    
    # Verify motion indicators
    motion_indicators = [i for i in result['ccd_indicators'] if i['type'] == 'unusual_maneuver']
    assert len(motion_indicators) > 0
    assert any('0.60 km/s' in i['detail'] for i in motion_indicators)

@pytest.mark.uct
def test_signature_analysis_with_magnitude(mock_udl_data, mock_visual_magnitude):
    """Test signature analysis with normalized visual magnitude."""
    uct = UCTAnalyzer()
    
    # Test with bright object
    bright_magnitude = {'magnitude': 12.0}
    result = uct.analyze_uct(
        mock_udl_data['track_data'],
        mock_udl_data['illumination_data'],
        mock_udl_data['lunar_data'],
        mock_udl_data['sensor_data'],
        mock_udl_data['space_weather'],
        mock_udl_data['radiation_belt'],
        mock_udl_data['rf_interference'],
        mock_udl_data['state_history'],
        bright_magnitude,
        None
    )
    assert not any(i['type'] == 'signature_management' for i in result['ccd_indicators'])
    
    # Test with dim object
    dim_magnitude = {'magnitude': 17.0}
    result = uct.analyze_uct(
        mock_udl_data['track_data'],
        mock_udl_data['illumination_data'],
        mock_udl_data['lunar_data'],
        mock_udl_data['sensor_data'],
        mock_udl_data['space_weather'],
        mock_udl_data['radiation_belt'],
        mock_udl_data['rf_interference'],
        mock_udl_data['state_history'],
        dim_magnitude,
        None
    )
    assert any(i['type'] == 'signature_management' for i in result['ccd_indicators'])
    dim_indicator = next(i for i in result['ccd_indicators'] if i['type'] == 'signature_management')
    assert '17.0' in dim_indicator['detail'] 