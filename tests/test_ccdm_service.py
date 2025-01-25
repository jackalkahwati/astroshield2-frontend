"""Tests for the enhanced CCDM service with ML capabilities."""
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from services.ccdm_service import CCDMService
from analysis.ml_evaluators import MLManeuverEvaluator, MLSignatureEvaluator, MLAMREvaluator
from fastapi.testclient import TestClient
from app.main import app
from app.models.ccdm import (
    ObservationData,
    ObjectAnalysisRequest,
    ShapeChangeResponse,
    ThermalSignatureResponse,
    PropulsiveCapabilityResponse,
    ConfidenceLevel,
    CCDMUpdate,
    CCDMAssessment
)

# Test client setup
client = TestClient(app)

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

@pytest.fixture
def test_observation_data():
    return ObservationData(
        timestamp=datetime.utcnow(),
        sensor_id="test_sensor_1",
        measurements={
            "position_x": 100.0,
            "position_y": 200.0,
            "position_z": 300.0,
            "velocity_x": 1.0,
            "velocity_y": 2.0,
            "velocity_z": 3.0
        },
        metadata={"source": "test"}
    )

@pytest.fixture
def test_object_id():
    return "TEST_OBJ_001"

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

class TestCCDMService:
    @pytest.mark.asyncio
    async def test_analyze_object(self, test_object_id, test_observation_data):
        request = ObjectAnalysisRequest(
            object_id=test_object_id,
            observation_data=test_observation_data
        )
        response = client.post("/api/v1/ccdm/analyze_object", json=request.dict())
        assert response.status_code == 200
        data = response.json()
        assert data["object_id"] == test_object_id
        assert 0.0 <= data["confidence_level"] <= 1.0

    @pytest.mark.asyncio
    async def test_detect_shape_changes(self, test_object_id):
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=24)
        response = client.post(
            "/api/v1/ccdm/detect_shape_changes",
            json={
                "object_id": test_object_id,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["object_id"] == test_object_id
        assert isinstance(data["detected_changes"], list)
        assert data["analysis_confidence"] in [level.value for level in ConfidenceLevel]

    @pytest.mark.asyncio
    async def test_assess_thermal_signature(self, test_object_id):
        timestamp = datetime.utcnow()
        response = client.post(
            "/api/v1/ccdm/assess_thermal_signature",
            json={
                "object_id": test_object_id,
                "timestamp": timestamp.isoformat()
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["object_id"] == test_object_id
        assert "temperature_kelvin" in data["metrics"]
        assert 0.0 <= data["metrics"]["anomaly_score"] <= 1.0

    @pytest.mark.asyncio
    async def test_evaluate_propulsive_capabilities(self, test_object_id):
        response = client.post(
            "/api/v1/ccdm/evaluate_propulsive_capabilities",
            json={
                "object_id": test_object_id,
                "analysis_period": 24
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["object_id"] == test_object_id
        assert "estimated_thrust" in data["metrics"]
        assert data["metrics"]["maneuver_capability_score"] >= 0.0

    @pytest.mark.asyncio
    async def test_get_historical_analysis(self, test_object_id):
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=7)
        response = client.get(
            f"/api/v1/ccdm/historical_analysis/{test_object_id}",
            params={
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

@pytest.mark.performance
class TestCCDMPerformance:
    @pytest.mark.asyncio
    async def test_concurrent_analysis(self, test_object_id, test_observation_data):
        import asyncio
        import time

        request = ObjectAnalysisRequest(
            object_id=test_object_id,
            observation_data=test_observation_data
        )

        async def make_request():
            response = client.post("/api/v1/ccdm/analyze_object", json=request.dict())
            return response.status_code

        start_time = time.time()
        num_requests = 100
        tasks = [make_request() for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)
        end_time = time.time()

        success_count = sum(1 for status in results if status == 200)
        assert success_count >= num_requests * 0.95  # 95% success rate
        assert end_time - start_time < 30  # Complete within 30 seconds

    @pytest.mark.asyncio
    async def test_shape_change_detection_performance(self, test_object_id):
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=30)
        
        start_perf_time = time.time()
        response = client.post(
            "/api/v1/ccdm/detect_shape_changes",
            json={
                "object_id": test_object_id,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            }
        )
        end_perf_time = time.time()
        
        assert response.status_code == 200
        assert end_perf_time - start_perf_time < 5  # Complete within 5 seconds

def test_process_ccdm_update():
    update = CCDMUpdate(
        object_id="test-sat-001",
        timestamp=time.time(),
        update_type="position",
        data={"x": 100, "y": 200, "z": 300},
        confidence=ConfidenceLevel.HIGH,
        severity="low"
    )
    response = client.post("/ccdm/update", json=update.dict())
    assert response.status_code == 200
    assert response.json()["status"] == "success"

def test_get_assessment():
    object_id = "test-sat-001"
    response = client.get(f"/ccdm/assessment/{object_id}")
    assert response.status_code == 200
    assessment = CCDMAssessment(**response.json())
    assert assessment.object_id == object_id
