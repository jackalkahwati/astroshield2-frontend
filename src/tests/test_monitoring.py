"""Test suite for trajectory prediction monitoring systems."""

import pytest
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import time
import numpy as np

from models.monitoring.prediction_metrics import PerformanceTracker
from models.monitoring.real_time_monitor import RealTimeMonitor

@pytest.fixture
def performance_tracker():
    """Create a performance tracker for testing."""
    tracker = PerformanceTracker(log_dir="test_logs/predictions")
    yield tracker
    # Cleanup
    if os.path.exists("test_logs"):
        import shutil
        shutil.rmtree("test_logs")

@pytest.fixture
def real_time_monitor():
    """Create a real-time monitor for testing."""
    monitor = RealTimeMonitor(metrics_interval=0.1)
    yield monitor
    monitor.stop()
    # Cleanup
    if os.path.exists("logs/real_time"):
        import shutil
        shutil.rmtree("logs/real_time")

@pytest.fixture
def sample_prediction():
    """Create a sample prediction result."""
    return {
        'prediction_id': 'test_001',
        'predicted_impact': {
            'lat': 28.5,
            'lon': -80.5,
            'time': 1200.0
        },
        'initial_state': {
            'position': {'x': 6771000.0, 'y': 0.0, 'z': 0.0},
            'velocity': {'vx': -7.8, 'vy': 0.0, 'vz': 0.0}
        },
        'confidence': 0.95,
        'environmental_conditions': {
            'density': 1.225,
            'temperature': 288.15,
            'wind_speed': 10.0
        },
        'computation_time': 0.5
    }

def test_performance_tracking(performance_tracker, sample_prediction):
    """Test basic performance tracking functionality."""
    # Record prediction
    performance_tracker.record_prediction(
        prediction_id=sample_prediction['prediction_id'],
        predicted_impact=sample_prediction['predicted_impact'],
        initial_state=sample_prediction['initial_state'],
        confidence=sample_prediction['confidence'],
        environmental_conditions=sample_prediction['environmental_conditions'],
        computation_time=sample_prediction['computation_time']
    )
    
    # Verify log file creation
    log_file = Path("test_logs/predictions") / f"prediction_{sample_prediction['prediction_id']}.json"
    assert log_file.exists()
    
    # Check log contents
    with open(log_file, 'r') as f:
        logged_data = json.load(f)
        assert logged_data['prediction_id'] == sample_prediction['prediction_id']
        assert logged_data['confidence'] == sample_prediction['confidence']
        assert logged_data['computation_time'] == sample_prediction['computation_time']

def test_validation_update(performance_tracker, sample_prediction):
    """Test updating predictions with actual results."""
    # Record prediction
    performance_tracker.record_prediction(
        prediction_id=sample_prediction['prediction_id'],
        predicted_impact=sample_prediction['predicted_impact'],
        initial_state=sample_prediction['initial_state'],
        confidence=sample_prediction['confidence'],
        environmental_conditions=sample_prediction['environmental_conditions'],
        computation_time=sample_prediction['computation_time']
    )
    
    # Add actual impact data
    actual_impact = {
        'lat': 28.6,
        'lon': -80.4,
        'time': 1195.0
    }
    
    performance_tracker.update_with_actual(
        sample_prediction['prediction_id'],
        actual_impact
    )
    
    # Verify error metrics
    log_file = Path("test_logs/predictions") / f"prediction_{sample_prediction['prediction_id']}.json"
    with open(log_file, 'r') as f:
        logged_data = json.load(f)
        assert 'error_metrics' in logged_data
        assert 'distance_error_km' in logged_data['error_metrics']
        assert 'time_error_seconds' in logged_data['error_metrics']
        
        # Check error magnitudes
        assert logged_data['error_metrics']['distance_error_km'] < 50.0  # Should be small for this test case
        assert logged_data['error_metrics']['time_error_seconds'] == 5.0  # |1195 - 1200|

def test_real_time_monitoring(real_time_monitor):
    """Test real-time monitoring system."""
    # Start monitoring
    real_time_monitor.start()
    
    # Record some prediction times
    for _ in range(5):
        real_time_monitor.record_prediction_time(0.5)
        time.sleep(0.1)
    
    # Get performance summary
    summary = real_time_monitor.get_performance_summary(window_minutes=1)
    
    assert 'avg_prediction_latency' in summary
    assert abs(summary['avg_prediction_latency'] - 0.5) < 0.1
    assert 'avg_cpu_usage' in summary
    assert 'avg_memory_usage' in summary
    
    # Test alert triggering
    real_time_monitor.update_thresholds({
        'prediction_latency': 0.1  # Set low to trigger alert
    })
    
    # Record high latency
    real_time_monitor.record_prediction_time(1.0)
    time.sleep(0.2)  # Allow time for processing
    
    # Check alert log
    alert_log = Path("logs/real_time/alerts.log")
    assert alert_log.exists()
    with open(alert_log, 'r') as f:
        alerts = f.readlines()
        assert any("High prediction latency" in line for line in alerts)

def test_validation_against_known_cases(performance_tracker, sample_prediction):
    """Test validation against known test cases."""
    # Create test validation case
    test_case = {
        'initial_state': sample_prediction['initial_state'],
        'actual_impact': {
            'lat': 28.5,
            'lon': -80.5,
            'time': 1200.0
        },
        'max_distance_error': 100.0,
        'max_time_error': 60.0
    }
    
    # Inject test case
    performance_tracker.validation_cases = {'test_case': test_case}
    
    # Record prediction
    performance_tracker.record_prediction(
        prediction_id=sample_prediction['prediction_id'],
        predicted_impact=sample_prediction['predicted_impact'],
        initial_state=sample_prediction['initial_state'],
        confidence=sample_prediction['confidence'],
        environmental_conditions=sample_prediction['environmental_conditions'],
        computation_time=sample_prediction['computation_time']
    )
    
    # Run validation
    results = performance_tracker.validate_against_known_cases()
    
    assert results['total_cases'] == 1
    assert results['passed_cases'] == 1
    assert results['failed_cases'] == 0
    assert len(results['details']) == 1
    assert results['details'][0]['passed']

def test_performance_summary_generation(performance_tracker):
    """Test generation of performance summaries."""
    # Record multiple predictions with varying parameters
    for i in range(5):
        performance_tracker.record_prediction(
            prediction_id=f'test_{i}',
            predicted_impact={
                'lat': 28.5 + i*0.1,
                'lon': -80.5 + i*0.1,
                'time': 1200.0 + i*10
            },
            initial_state={
                'position': {'x': 6771000.0, 'y': 0.0, 'z': 0.0},
                'velocity': {'vx': -7.8, 'vy': 0.0, 'vz': 0.0}
            },
            confidence=0.9 + i*0.02,
            environmental_conditions={
                'density': 1.225,
                'temperature': 288.15,
                'wind_speed': 10.0 + i
            },
            computation_time=0.5 + i*0.1
        )
        
        # Add actual impacts with varying errors
        performance_tracker.update_with_actual(
            f'test_{i}',
            {
                'lat': 28.5 + i*0.1 + 0.1,
                'lon': -80.5 + i*0.1 - 0.1,
                'time': 1200.0 + i*10 + 5
            }
        )
    
    # Get summary
    summary = performance_tracker.get_performance_summary()
    
    assert 'mean_distance_error_km' in summary
    assert 'mean_time_error_seconds' in summary
    assert 'mean_confidence' in summary
    assert 'total_predictions' in summary
    assert summary['total_predictions'] == 5
    
    # Verify reasonable error ranges
    assert 0 < summary['mean_distance_error_km'] < 50
    assert 0 < summary['mean_time_error_seconds'] < 10
    assert 0.9 < summary['mean_confidence'] < 1.0 