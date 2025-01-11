import pytest
import numpy as np
from unittest.mock import Mock, patch
from analysis.ml_evaluators import MLManeuverEvaluator, MLSignatureEvaluator, MLAMREvaluator

@pytest.fixture
def sample_trajectory_data():
    """Generate sample trajectory data for testing."""
    return {
        'timestamps': np.linspace(0, 3600, 100),
        'positions': np.random.rand(100, 3),
        'velocities': np.random.rand(100, 3)
    }

@pytest.fixture
def sample_signature_data():
    """Generate sample signature data for testing."""
    return {
        'optical': np.random.rand(32, 32),
        'radar': np.random.rand(64, 64)
    }

@pytest.fixture
def sample_amr_data():
    """Generate sample Area-to-Mass Ratio data for testing."""
    return {
        'timestamps': np.linspace(0, 3600, 50),
        'amr_values': np.random.rand(50),
        'solar_pressure': np.random.rand(50)
    }

def test_maneuver_detection(sample_trajectory_data):
    """Test maneuver detection model."""
    evaluator = MLManeuverEvaluator()
    
    # Test normal trajectory
    result = evaluator.analyze_maneuvers(sample_trajectory_data)
    assert isinstance(result, list)
    assert all(0 <= indicator.confidence_level <= 1 for indicator in result)

    # Test trajectory with simulated maneuver
    modified_data = sample_trajectory_data.copy()
    modified_data['velocities'][50:] += 1.0  # Simulate sudden velocity change
    result_with_maneuver = evaluator.analyze_maneuvers(modified_data)
    assert any(indicator.indicator_name == 'maneuver_detected' 
              for indicator in result_with_maneuver)

def test_signature_analysis(sample_signature_data):
    """Test signature analysis model."""
    evaluator = MLSignatureEvaluator()
    
    # Test normal signature
    result = evaluator.analyze_signatures(sample_signature_data)
    assert isinstance(result, list)
    assert all(0 <= indicator.confidence_level <= 1 for indicator in result)

    # Test anomalous signature
    modified_data = sample_signature_data.copy()
    modified_data['optical'] *= 2.0  # Simulate signature anomaly
    result_with_anomaly = evaluator.analyze_signatures(modified_data)
    assert any(indicator.indicator_name == 'signature_anomaly' 
              for indicator in result_with_anomaly)

def test_amr_analysis(sample_amr_data):
    """Test AMR analysis model."""
    evaluator = MLAMREvaluator()
    
    # Test normal AMR data
    result = evaluator.analyze_amr(sample_amr_data)
    assert isinstance(result, list)
    assert all(0 <= indicator.confidence_level <= 1 for indicator in result)

    # Test anomalous AMR
    modified_data = sample_amr_data.copy()
    modified_data['amr_values'] *= 1.5  # Simulate AMR change
    result_with_change = evaluator.analyze_amr(modified_data)
    assert any(indicator.indicator_name == 'amr_change' 
              for indicator in result_with_change)

def test_model_error_handling():
    """Test error handling in ML models."""
    evaluators = [
        MLManeuverEvaluator(),
        MLSignatureEvaluator(),
        MLAMREvaluator()
    ]
    
    for evaluator in evaluators:
        # Test with invalid input
        with pytest.raises(ValueError):
            evaluator.analyze(None)
        
        # Test with malformed data
        with pytest.raises(ValueError):
            evaluator.analyze({'invalid': 'data'})

def test_confidence_thresholds():
    """Test confidence threshold configurations."""
    evaluator = MLManeuverEvaluator(
        confidence_threshold=0.8,
        noise_threshold=0.1
    )
    
    # Test that low confidence predictions are filtered
    with patch('analysis.ml_evaluators.MLManeuverEvaluator._predict', 
              return_value=np.array([0.7])):  # Below threshold
        result = evaluator.analyze_maneuvers(sample_trajectory_data())
        assert len(result) == 0

    # Test that high confidence predictions are included
    with patch('analysis.ml_evaluators.MLManeuverEvaluator._predict', 
              return_value=np.array([0.9])):  # Above threshold
        result = evaluator.analyze_maneuvers(sample_trajectory_data())
        assert len(result) > 0 