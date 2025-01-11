"""Tests for the enhanced CCDM service with ML capabilities."""
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from services.ccdm_service import CCDMService
from analysis.ml_evaluators import MLManeuverEvaluator, MLSignatureEvaluator, MLAMREvaluator

@pytest.fixture
def mock_evaluators():
    """Mock ML evaluators."""
    with patch('services.ccdm_service.MLManeuverEvaluator') as mock_maneuver, \
         patch('services.ccdm_service.MLSignatureEvaluator') as mock_signature, \
         patch('services.ccdm_service.MLAMREvaluator') as mock_amr:
        
        # Configure mock returns
        mock_maneuver.return_value.analyze_maneuvers.return_value = []
        mock_signature.return_value.analyze_signatures.return_value = []
        mock_amr.return_value.analyze_amr.return_value = []
        
        yield {
            'maneuver': mock_maneuver,
            'signature': mock_signature,
            'amr': mock_amr
        }

@pytest.fixture
def ccdm_service():
    """Create CCDM service instance."""
    return CCDMService()

def test_analyze_conjunction_success(ccdm_service, mock_evaluators):
    """Test successful conjunction analysis."""
    # Setup
    spacecraft_id = "test_spacecraft_1"
    other_spacecraft_id = "test_spacecraft_2"
    
    # Execute
    result = ccdm_service.analyze_conjunction(spacecraft_id, other_spacecraft_id)
    
    # Verify
    assert result['status'] == 'operational'
    assert 'indicators' in result
    assert 'analysis_timestamp' in result
    assert 'risk_assessment' in result

def test_analyze_conjunction_with_indicators(ccdm_service):
    """Test conjunction analysis with mock indicators."""
    # Setup
    spacecraft_id = "test_spacecraft_1"
    other_spacecraft_id = "test_spacecraft_2"
    
    # Mock indicators
    mock_indicator = Mock()
    mock_indicator.indicator_name = "test_maneuver"
    mock_indicator.confidence_level = 0.9
    mock_indicator.dict.return_value = {
        'indicator_name': 'test_maneuver',
        'confidence_level': 0.9
    }
    
    with patch.object(ccdm_service.maneuver_evaluator, 'analyze_maneuvers', return_value=[mock_indicator]):
        # Execute
        result = ccdm_service.analyze_conjunction(spacecraft_id, other_spacecraft_id)
    
    # Verify
    assert result['status'] == 'operational'
    assert len(result['indicators']) == 1
    assert result['risk_assessment']['risk_level'] == 'critical'

def test_get_active_conjunctions(ccdm_service):
    """Test getting active conjunctions."""
    # Setup
    spacecraft_id = "test_spacecraft"
    mock_nearby = ["nearby_1", "nearby_2"]
    
    with patch.object(ccdm_service, '_get_nearby_spacecraft', return_value=mock_nearby):
        # Execute
        result = ccdm_service.get_active_conjunctions(spacecraft_id)
    
    # Verify
    assert isinstance(result, list)
    assert len(result) == len(mock_nearby)

def test_analyze_conjunction_trends(ccdm_service):
    """Test analyzing conjunction trends."""
    # Setup
    spacecraft_id = "test_spacecraft"
    hours = 24
    
    # Execute
    result = ccdm_service.analyze_conjunction_trends(spacecraft_id, hours)
    
    # Verify
    assert 'total_conjunctions' in result
    assert 'risk_levels' in result
    assert 'temporal_metrics' in result
    assert 'velocity_metrics' in result
    assert 'ml_insights' in result

def test_risk_calculation():
    """Test risk calculation logic."""
    service = CCDMService()
    
    # Create mock indicators
    indicators = [
        Mock(indicator_name='maneuver_detected', confidence_level=0.9),
        Mock(indicator_name='signature_anomaly', confidence_level=0.7),
        Mock(indicator_name='amr_change', confidence_level=0.5)
    ]
    
    # Calculate risk
    risk = service._calculate_risk(indicators)
    
    # Verify
    assert risk['overall_risk'] == 0.9
    assert risk['risk_level'] == 'critical'
    assert all(factor in risk['risk_factors'] for factor in ['maneuver', 'signature', 'amr'])

def test_error_handling(ccdm_service):
    """Test error handling in main methods."""
    # Setup
    spacecraft_id = "test_spacecraft"
    other_spacecraft_id = "other_spacecraft"
    
    # Test conjunction analysis error
    with patch.object(ccdm_service, '_get_trajectory_data', side_effect=Exception("Test error")):
        result = ccdm_service.analyze_conjunction(spacecraft_id, other_spacecraft_id)
        assert result['status'] == 'error'
        assert 'message' in result
    
    # Test trends analysis error
    with patch.object(ccdm_service, '_get_historical_conjunctions', side_effect=Exception("Test error")):
        result = ccdm_service.analyze_conjunction_trends(spacecraft_id)
        assert result['status'] == 'error'
        assert 'message' in result
