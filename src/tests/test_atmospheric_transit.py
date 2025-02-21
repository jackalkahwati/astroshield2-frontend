"""Test suite for atmospheric transit detection."""

import pytest
from datetime import datetime, timedelta
import numpy as np
from typing import List, Dict, Any

from models.atmospheric_transit import (
    AtmosphericTransitDetector,
    IonosphericAnalyzer,
    MagneticFieldAnalyzer,
    DopplerTracker,
    TrajectoryPredictor,
    GeophysicalData,
    SDRMeasurement,
    TransitObject,
    TransitType
)
from models.config.atmospheric_transit_config import DETECTOR_CONFIG

@pytest.fixture
def detector():
    """Create an atmospheric transit detector for testing."""
    return AtmosphericTransitDetector(DETECTOR_CONFIG)

@pytest.fixture
def mock_geophysical_data():
    """Create mock geophysical data."""
    base_time = datetime.utcnow()
    data = []
    
    for i in range(60):  # 1 minute of data at 1Hz
        time = base_time + timedelta(seconds=i)
        
        # Add a simulated perturbation around 30 seconds
        tec_anomaly = 0.0
        if 25 <= i <= 35:
            tec_anomaly = 0.8  # TECU
            
        magnetic_anomaly = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        if 28 <= i <= 32:
            magnetic_anomaly = {'x': 60.0, 'y': 40.0, 'z': -30.0}  # nT
            
        data.append(GeophysicalData(
            time=time,
            ionospheric_tec=1.0 + tec_anomaly,
            magnetic_field=magnetic_anomaly,
            location={
                'lat': 45.0,
                'lon': -75.0,
                'alt': 250.0
            },
            confidence=0.9
        ))
    
    return data

@pytest.fixture
def mock_sdr_data():
    """Create mock SDR data."""
    base_time = datetime.utcnow()
    data = []
    
    for i in range(300):  # 5 minutes of data at 1Hz
        time = base_time + timedelta(seconds=i)
        
        # Add a simulated Doppler track around 150 seconds
        doppler_shift = 0.0
        if 145 <= i <= 155:
            doppler_shift = 200.0 * np.sin((i - 145) * np.pi / 10)  # Hz
            
        # Add corresponding power variation
        power = -90.0  # dBm
        if 145 <= i <= 155:
            power = -60.0 + (i - 150) ** 2 * 0.2  # dBm
            
        data.append(SDRMeasurement(
            time=time,
            frequency=300e6,  # 300 MHz
            power=power,
            doppler_shift=doppler_shift,
            location={
                'lat': 45.0,
                'lon': -75.0,
                'alt': 250.0
            },
            confidence=0.85
        ))
    
    return data

def test_geophysical_processing(detector, mock_geophysical_data):
    """Test processing of geophysical data."""
    detections = detector.process_geophysical_data(mock_geophysical_data)
    
    assert len(detections) > 0
    for detection in detections:
        assert 'time' in detection
        assert 'location' in detection
        assert 'confidence' in detection
        assert detection['confidence'] >= DETECTOR_CONFIG['confidence_threshold']

def test_sdr_processing(detector, mock_sdr_data):
    """Test processing of SDR data."""
    detections = detector.process_sdr_data(mock_sdr_data)
    
    assert len(detections) > 0
    for detection in detections:
        assert 'time' in detection
        assert 'frequency' in detection
        assert 'doppler_shift' in detection
        assert 'confidence' in detection
        assert detection['confidence'] >= DETECTOR_CONFIG['confidence_threshold']

def test_transit_detection(detector, mock_geophysical_data, mock_sdr_data):
    """Test combined transit detection."""
    transit_objects = detector.detect_transits(
        mock_geophysical_data,
        mock_sdr_data
    )
    
    assert len(transit_objects) > 0
    for obj in transit_objects:
        assert isinstance(obj, TransitObject)
        assert obj.confidence >= DETECTOR_CONFIG['confidence_threshold']
        assert isinstance(obj.transit_type, TransitType)
        
        # Check velocity is above space-capable threshold
        velocity_magnitude = np.sqrt(
            obj.velocity['vx']**2 +
            obj.velocity['vy']**2 +
            obj.velocity['vz']**2
        )
        assert velocity_magnitude >= DETECTOR_CONFIG['velocity_threshold']
        
        # Check altitude is within range
        assert (DETECTOR_CONFIG['altitude_range']['min'] <=
                obj.location['alt'] <=
                DETECTOR_CONFIG['altitude_range']['max'])

def test_reentry_prediction(detector, mock_geophysical_data, mock_sdr_data):
    """Test prediction of reentry trajectories."""
    transit_objects = detector.detect_transits(
        mock_geophysical_data,
        mock_sdr_data
    )
    
    reentry_objects = [
        obj for obj in transit_objects
        if obj.transit_type == TransitType.REENTRY
    ]
    
    for obj in reentry_objects:
        assert obj.predicted_impact is not None
        assert 'time' in obj.predicted_impact
        assert 'location' in obj.predicted_impact
        assert 'uncertainty_radius_km' in obj.predicted_impact
        assert obj.predicted_impact['confidence'] >= DETECTOR_CONFIG['impact']['min_confidence']

def test_ionospheric_analyzer():
    """Test ionospheric perturbation detection."""
    analyzer = IonosphericAnalyzer(DETECTOR_CONFIG['ionospheric'])
    
    # Create test data with a simulated perturbation
    timestamps = [
        datetime.utcnow() + timedelta(seconds=i)
        for i in range(600)  # 10 minutes
    ]
    
    tec_data = [1.0] * 600  # Base TEC value
    # Add perturbation
    for i in range(290, 310):
        tec_data[i] += 0.8  # TECU
    
    perturbations = analyzer.detect_perturbations(tec_data, timestamps)
    
    assert len(perturbations) > 0
    for perturbation in perturbations:
        assert perturbation['magnitude'] >= DETECTOR_CONFIG['ionospheric']['tec_threshold']
        assert (DETECTOR_CONFIG['ionospheric']['min_perturbation_duration'] <=
                perturbation['duration'] <=
                DETECTOR_CONFIG['ionospheric']['max_perturbation_duration'])

def test_magnetic_analyzer():
    """Test magnetic field disturbance detection."""
    analyzer = MagneticFieldAnalyzer(DETECTOR_CONFIG['magnetic'])
    
    # Create test data with a simulated disturbance
    timestamps = [
        datetime.utcnow() + timedelta(seconds=i)
        for i in range(600)  # 10 minutes
    ]
    
    field_data = [
        {'x': 0.0, 'y': 0.0, 'z': 0.0}
        for _ in range(600)
    ]
    # Add disturbance
    for i in range(290, 310):
        field_data[i] = {'x': 60.0, 'y': 40.0, 'z': -30.0}  # nT
    
    disturbances = analyzer.detect_disturbances(field_data, timestamps)
    
    assert len(disturbances) > 0
    for disturbance in disturbances:
        assert disturbance['magnitude'] >= DETECTOR_CONFIG['magnetic']['field_threshold']
        assert (DETECTOR_CONFIG['magnetic']['min_disturbance_duration'] <=
                disturbance['duration'] <=
                DETECTOR_CONFIG['magnetic']['max_disturbance_duration'])

def test_doppler_tracker():
    """Test Doppler shift tracking."""
    tracker = DopplerTracker(DETECTOR_CONFIG['sdr'])
    
    # Create test data with a simulated Doppler track
    timestamps = [
        datetime.utcnow() + timedelta(seconds=i)
        for i in range(300)  # 5 minutes
    ]
    
    frequencies = [0.0] * 300
    # Add Doppler shift
    for i in range(145, 156):
        frequencies[i] = 200.0 * np.sin((i - 145) * np.pi / 10)  # Hz
    
    tracks = tracker.track_signals(frequencies, timestamps)
    
    assert len(tracks) > 0
    for track in tracks:
        assert abs(track['max_shift']) >= DETECTOR_CONFIG['sdr']['doppler_threshold']
        assert (DETECTOR_CONFIG['sdr']['min_track_duration'] <=
                track['duration'] <=
                DETECTOR_CONFIG['sdr']['max_track_duration'])

def test_trajectory_predictor():
    """Test trajectory and impact prediction."""
    predictor = TrajectoryPredictor(DETECTOR_CONFIG['impact'])
    
    # Create test initial state
    initial_state = {
        'time': datetime.utcnow(),
        'position': {'x': 0.0, 'y': 0.0, 'z': 100.0},  # km
        'velocity': {'vx': 1.0, 'vy': 1.0, 'vz': -2.0}  # km/s
    }
    
    trajectory = predictor.predict_trajectory(initial_state)
    assert len(trajectory) > 0
    
    impact = predictor.estimate_impact(trajectory)
    assert impact is not None
    assert 'time' in impact
    assert 'location' in impact
    assert 'uncertainty_radius_km' in impact
``` 