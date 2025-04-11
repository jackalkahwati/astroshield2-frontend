"""Tests for the enhanced CCDM service with ML capabilities."""
import pytest
import time
import uuid
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import asyncio

# Import the CCDM service and models
from backend.app.services.ccdm import CCDMService
from backend.app.models.ccdm import (
    ObjectAnalysisResponse,
    ShapeChangeResponse,
    ThermalSignatureResponse,
    PropulsiveCapabilityResponse,
    HistoricalAnalysis,
    ShapeChangeMetrics,
    ThermalSignatureMetrics,
    PropulsionMetrics,
    PropulsionType,
    CCDMAssessment,
    AnomalyDetection
)

# Import threat analyzer classes
from src.asttroshield.analysis.threat_analyzer import (
    StabilityIndicator,
    ManeuverIndicator,
    RFIndicator,
    SubSatelliteAnalyzer,
    ITUComplianceChecker,
    AnalystDisagreementChecker,
    OrbitAnalyzer,
    SignatureAnalyzer,
    StimulationAnalyzer,
    AMRAnalyzer,
    ImagingManeuverAnalyzer,
    LaunchAnalyzer,
    EclipseAnalyzer,
    RegistryChecker
)

# Import ML models
from ml.models.anomaly_detector import SpaceObjectAnomalyDetector
from ml.models.track_evaluator import TrackEvaluator

# Import needed clients
from src.asttroshield.api_client.udl_client import UDLClient
from src.asttroshield.udl_integration import USSFUDLIntegrator
from src.kafka_client.kafka_consume import KafkaConsumer

@pytest.fixture
def mock_space_object_data():
    """
    Fixture that provides mock space object data for testing.
    """
    return {
        "object_id": "12345",
        "state_vector": {
            "position": {"x": 1000.0, "y": 2000.0, "z": 3000.0},
            "velocity": {"x": 1.0, "y": 2.0, "z": 3.0},
            "epoch": datetime.utcnow().isoformat()
        },
        "state_history": [
            {
                "time": (datetime.utcnow() - timedelta(hours=1)).isoformat(),
                "position": {"x": 900.0, "y": 1900.0, "z": 2900.0},
                "velocity": {"x": 0.9, "y": 1.9, "z": 2.9}
            },
            {
                "time": (datetime.utcnow() - timedelta(hours=2)).isoformat(),
                "position": {"x": 800.0, "y": 1800.0, "z": 2800.0},
                "velocity": {"x": 0.8, "y": 1.8, "z": 2.8}
            }
        ],
        "maneuver_history": [
            {
                "time": (datetime.utcnow() - timedelta(days=1)).isoformat(),
                "delta_v": 0.5,
                "thrust_vector": {"x": 0.1, "y": 0.2, "z": 0.3},
                "duration": 60.0,
                "confidence": 0.85
            },
            {
                "time": (datetime.utcnow() - timedelta(days=7)).isoformat(),
                "delta_v": 0.8,
                "thrust_vector": {"x": -0.1, "y": 0.3, "z": 0.2},
                "duration": 90.0,
                "confidence": 0.9
            }
        ],
        "rf_history": [
            {
                "time": (datetime.utcnow() - timedelta(hours=6)).isoformat(),
                "frequency": 2200.0,
                "power": -90.0,
                "duration": 300.0,
                "bandwidth": 5.0,
                "confidence": 0.75
            }
        ],
        "radar_signature": {
            "rcs": 1.5,
            "timestamp": datetime.utcnow().isoformat(),
            "sensor_id": "radar-001",
            "confidence": 0.9
        },
        "optical_signature": {
            "magnitude": 6.5,
            "timestamp": datetime.utcnow().isoformat(),
            "sensor_id": "optical-002",
            "confidence": 0.85
        },
        "baseline_signatures": {
            "radar": {
                "rcs_mean": 1.2,
                "rcs_std": 0.3,
                "rcs_min": 0.5,
                "rcs_max": 2.0
            },
            "optical": {
                "magnitude_mean": 7.0,
                "magnitude_std": 0.5,
                "magnitude_min": 6.0,
                "magnitude_max": 8.0
            }
        },
        "baseline_pol": {
            "rf": {
                "max_power": -95.0,
                "frequencies": [2200.0, 8400.0],
                "duty_cycles": [0.1, 0.05]
            },
            "maneuvers": {
                "typical_delta_v": 0.3,
                "typical_intervals": [14, 30]
            }
        },
        "orbit_data": {
            "semi_major_axis": 7000.0,
            "eccentricity": 0.001,
            "inclination": 51.6,
            "raan": 120.0,
            "arg_perigee": 180.0,
            "mean_anomaly": 0.0,
            "mean_motion": 15.5
        },
        "parent_orbit_data": {
            "parent_object_id": "12340",
            "semi_major_axis": 7000.0,
            "inclination": 51.6,
            "eccentricity": 0.001
        },
        "population_data": {
            "orbit_regime": "LEO",
            "density": 15.0,
            "mean_amr": 0.01,
            "std_amr": 0.003
        },
        "anomaly_baseline": {
            "thermal_profile": [270, 280, 275, 265],
            "maneuver_frequency": 0.06
        },
        "object_events": [
            {
                "type": "maneuver",
                "time": (datetime.utcnow() - timedelta(days=1)).isoformat(),
                "data": {"delta_v": 0.5}
            },
            {
                "type": "rf_emission",
                "time": (datetime.utcnow() - timedelta(hours=6)).isoformat(),
                "data": {"frequency": 2200.0, "power": -90.0}
            },
            {
                "type": "conjunction",
                "time": (datetime.utcnow() - timedelta(days=2)).isoformat(),
                "data": {"miss_distance": 45.0, "secondary_object": "23456"}
            }
        ],
        "filing_data": {
            "itu_filing": "ABC123",
            "authorized_frequencies": [2200.0, 8400.0]
        },
        "registry_data": {
            "registered_ids": ["12345"],
            "registry_authority": "UNOOSA"
        }
    }

@pytest.fixture
def mock_udl_client():
    """
    Fixture that provides a mock UDL client.
    """
    client = MagicMock()
    
    # Set up mock return values for common methods
    client.get_state_vector = AsyncMock(return_value={
        "position": {"x": 1000.0, "y": 2000.0, "z": 3000.0},
        "velocity": {"x": 1.0, "y": 2.0, "z": 3.0},
        "epoch": datetime.utcnow().isoformat()
    })
    
    client.get_state_vector_history = AsyncMock(return_value=[
        {
            "time": (datetime.utcnow() - timedelta(hours=i)).isoformat(),
            "position": {"x": 1000.0 - i*100, "y": 2000.0 - i*100, "z": 3000.0 - i*100},
            "velocity": {"x": 1.0 - i*0.1, "y": 2.0 - i*0.1, "z": 3.0 - i*0.1}
        } for i in range(1, 24)
    ])
    
    client.get_elset_data = AsyncMock(return_value={
        "semi_major_axis": 7000.0,
        "eccentricity": 0.001,
        "inclination": 51.6,
        "raan": 120.0,
        "arg_perigee": 180.0,
        "mean_anomaly": 0.0,
        "mean_motion": 15.5
    })
    
    client.get_maneuver_data = AsyncMock(return_value=[
        {
            "time": (datetime.utcnow() - timedelta(days=i*7)).isoformat(),
            "delta_v": 0.5 + (i*0.1),
            "thrust_vector": {"x": 0.1, "y": 0.2, "z": 0.3},
            "duration": 60.0 + (i*10),
            "confidence": 0.85
        } for i in range(0, 3)
    ])
    
    client.get_conjunction_data = AsyncMock(return_value={
        "events": [
            {
                "timestamp": (datetime.utcnow() - timedelta(days=i)).isoformat(),
                "miss_distance": 40.0 + (i*5),
                "probability": 1e-6 * (i+1)
            } for i in range(0, 5)
        ]
    })
    
    client.get_rf_interference = AsyncMock(return_value={
        "measurements": [
            {
                "timestamp": (datetime.utcnow() - timedelta(hours=i*6)).isoformat(),
                "frequency": 2200.0,
                "power_level": -90.0 - (i*2),
                "duration": 300.0,
                "bandwidth": 5.0
            } for i in range(0, 4)
        ]
    })
    
    return client

@pytest.fixture
def mock_analyzers():
    """
    Fixture that provides mock analyzer instances.
    """
    mock_stability = MagicMock(spec=StabilityIndicator)
    mock_maneuver = MagicMock(spec=ManeuverIndicator)
    mock_rf = MagicMock(spec=RFIndicator)
    mock_subsatellite = MagicMock(spec=SubSatelliteAnalyzer)
    mock_itu = MagicMock(spec=ITUComplianceChecker)
    mock_disagreement = MagicMock(spec=AnalystDisagreementChecker)
    mock_orbit = MagicMock(spec=OrbitAnalyzer)
    mock_signature = MagicMock(spec=SignatureAnalyzer)
    mock_stimulation = MagicMock(spec=StimulationAnalyzer)
    mock_amr = MagicMock(spec=AMRAnalyzer)
    mock_imaging = MagicMock(spec=ImagingManeuverAnalyzer)
    mock_launch = MagicMock(spec=LaunchAnalyzer)
    mock_eclipse = MagicMock(spec=EclipseAnalyzer)
    mock_registry = MagicMock(spec=RegistryChecker)
    
    # Configure standard returns for analyzer methods
    mock_stability.analyze_stability.return_value = {"stability_changed": False, "confidence": 0.8}
    mock_maneuver.analyze_maneuvers.return_value = {"maneuvers_detected": True, "pol_violation": False, "confidence": 0.85}
    mock_rf.analyze_rf_pattern.return_value = {"rf_detected": True, "pol_violation": False, "confidence": 0.7}
    mock_subsatellite.detect_sub_satellites.return_value = {"subsatellites_detected": False, "confidence": 0.8}
    mock_itu.check_itu_compliance.return_value = {"violates_filing": False, "confidence": 0.95}
    mock_disagreement.check_disagreements.return_value = {"class_disagreement": False, "confidence": 0.9}
    mock_orbit.analyze_orbit.return_value = {"orbit_out_of_family": False, "confidence": 0.85}
    mock_signature.analyze_signatures.return_value = {"optical_out_of_family": False, "confidence": 0.8}
    mock_stimulation.analyze_stimulation.return_value = {"stimulation_detected": False, "confidence": 0.7}
    mock_amr.analyze_amr.return_value = {"amr_out_of_family": False, "confidence": 0.85}
    mock_imaging.analyze_imaging_maneuvers.return_value = {"imaging_maneuver_detected": False, "confidence": 0.75}
    mock_launch.analyze_launch.return_value = {"suspicious_source": False, "confidence": 0.9}
    mock_eclipse.analyze_eclipse_behavior.return_value = {"uct_during_eclipse": False, "confidence": 0.8}
    mock_registry.check_registry.return_value = {"registered": True, "confidence": 0.95}
    
    return {
        "stability": mock_stability,
        "maneuver": mock_maneuver,
        "rf": mock_rf,
        "subsatellite": mock_subsatellite,
        "itu": mock_itu,
        "disagreement": mock_disagreement,
        "orbit": mock_orbit,
        "signature": mock_signature,
        "stimulation": mock_stimulation,
        "amr": mock_amr,
        "imaging": mock_imaging,
        "launch": mock_launch,
        "eclipse": mock_eclipse,
        "registry": mock_registry
    }

@pytest.fixture
def mock_ml_models():
    """
    Fixture that provides mock ML models.
    """
    mock_anomaly_detector = MagicMock(spec=SpaceObjectAnomalyDetector)
    mock_track_evaluator = MagicMock(spec=TrackEvaluator)
    
    return {
        "anomaly_detector": mock_anomaly_detector,
        "track_evaluator": mock_track_evaluator
    }

@pytest.fixture
def ccdm_service_instance(mock_udl_client, mock_analyzers, mock_ml_models):
    """
    Create a CCDM service instance with mocked dependencies.
    """
    # Create test config
    config = {
        "data_sources": {
            "udl": {
                "base_url": "https://test-udl.example.com/api",
                "api_key_env": "TEST_UDL_API_KEY",
                "timeout": 5
            },
            "space_track": {
                "base_url": "https://test-space-track.example.com/query",
                "timeout": 5
            },
            "tmdb": {
                "base_url": "https://test-tmdb.example.com/api",
                "timeout": 5
            }
        },
        "kafka": {
            "bootstrap_servers": "test-kafka:9092",
            "group_id": "test-group",
            "topics": {
                "subscribe": ["test-topic-1", "test-topic-2"],
                "publish": ["test-output-topic"]
            },
            "heartbeat_interval_ms": 1000
        },
        "ccdm": {
            "indicators": {
                "stability": {"threshold": 0.7, "weight": 1.0},
                "maneuvers": {"threshold": 0.7, "weight": 1.0},
                "rf": {"threshold": 0.7, "weight": 1.0},
            },
            "sensor_types": ["RADAR", "EO"],
            "ml_filter_threshold": 0.7,
            "assessment_interval_seconds": 60,
            "anomaly_detection": {
                "lookback_days": 7,
                "anomaly_threshold": 0.7
            }
        },
        "security": {
            "data_classification": "TEST",
            "access_control": {
                "required_roles": ["test_analyst"]
            }
        }
    }
    
    # Create service with patched dependencies
    with patch('src.asttroshield.api_client.udl_client.UDLClient', return_value=mock_udl_client), \
         patch('src.asttroshield.udl_integration.USSFUDLIntegrator'), \
         patch('src.kafka_client.kafka_consume.KafkaConsumer'), \
         patch('ml.models.anomaly_detector.SpaceObjectAnomalyDetector', 
               return_value=mock_ml_models["anomaly_detector"]), \
         patch('ml.models.track_evaluator.TrackEvaluator', 
               return_value=mock_ml_models["track_evaluator"]):
        
        service = CCDMService(config=config)
        
        # Replace analyzer instances with mocks
        service.stability_analyzer = mock_analyzers["stability"]
        service.maneuver_analyzer = mock_analyzers["maneuver"]
        service.rf_analyzer = mock_analyzers["rf"]
        service.subsatellite_analyzer = mock_analyzers["subsatellite"]
        service.itu_checker = mock_analyzers["itu"]
        service.disagreement_checker = mock_analyzers["disagreement"]
        service.orbit_analyzer = mock_analyzers["orbit"]
        service.signature_analyzer = mock_analyzers["signature"]
        service.stimulation_analyzer = mock_analyzers["stimulation"]
        service.amr_analyzer = mock_analyzers["amr"]
        service.imaging_analyzer = mock_analyzers["imaging"]
        service.launch_analyzer = mock_analyzers["launch"]
        service.eclipse_analyzer = mock_analyzers["eclipse"]
        service.registry_checker = mock_analyzers["registry"]
        
        return service


class TestCCDMServiceInit:
    """Test CCDM service initialization."""
    
    def test_service_init(self, ccdm_service_instance):
        """Test that the service initializes correctly with all components."""
        assert ccdm_service_instance is not None
        assert ccdm_service_instance._config is not None
        
        # Check that critical components are available
        assert hasattr(ccdm_service_instance, 'udl_client')
        assert hasattr(ccdm_service_instance, 'udl_integrator')
        assert hasattr(ccdm_service_instance, 'anomaly_detector')
        assert hasattr(ccdm_service_instance, 'track_evaluator')
        assert hasattr(ccdm_service_instance, 'kafka_consumer')
        
        # Check that all analyzers are present
        assert hasattr(ccdm_service_instance, 'stability_analyzer')
        assert hasattr(ccdm_service_instance, 'maneuver_analyzer')
        assert hasattr(ccdm_service_instance, 'rf_analyzer')
        assert hasattr(ccdm_service_instance, 'subsatellite_analyzer')
        assert hasattr(ccdm_service_instance, 'itu_checker')
        assert hasattr(ccdm_service_instance, 'orbit_analyzer')
        
        # Check that cache is initialized
        assert isinstance(ccdm_service_instance._object_data_cache, dict)
        assert isinstance(ccdm_service_instance._cache_ttl, int)
        assert isinstance(ccdm_service_instance._cache_timestamps, dict)


@pytest.mark.asyncio
class TestCCDMServiceDataFetching:
    """Test data fetching methods of CCDM service."""
    
    async def test_fetch_udl_data(self, ccdm_service_instance, mock_udl_client):
        """Test fetching data from UDL."""
        # Configure mock return values
        object_id = "test-sat-001"
        
        # Call the method
        result = await ccdm_service_instance._fetch_udl_data(object_id)
        
        # Verify the result structure
        assert "state_vector" in result
        assert "state_history" in result
        assert "maneuver_history" in result
        assert "rf_history" in result
        assert "conjunction_history" in result
        assert "orbit_data" in result
        
        # Verify that UDL client methods were called with correct parameters
        mock_udl_client.get_state_vector.assert_called_once_with(object_id)
        mock_udl_client.get_elset_data.assert_called_once_with(object_id)
        mock_udl_client.get_maneuver_data.assert_called_once_with(object_id)
        mock_udl_client.get_conjunction_data.assert_called_once_with(object_id)
    
    async def test_fetch_space_track_data(self, ccdm_service_instance):
        """Test fetching data from Space Track."""
        object_id = "test-sat-001"
        
        # Call the method
        result = await ccdm_service_instance._fetch_space_track_data(object_id)
        
        # Verify the result structure
        assert "space_track_data" in result
        assert "registry_data" in result
        assert result["space_track_data"]["NORAD_CAT_ID"] == object_id if object_id.isdigit() else "99999"
    
    async def test_fetch_tmdb_data(self, ccdm_service_instance):
        """Test fetching data from TMDB."""
        object_id = "test-sat-001"
        
        # Call the method
        result = await ccdm_service_instance._fetch_tmdb_data(object_id)
        
        # Verify the result structure
        assert "baseline_signatures" in result
        assert "baseline_pol" in result
        assert "population_data" in result
        assert "anomaly_baseline" in result
    
    async def test_fetch_kafka_data(self, ccdm_service_instance):
        """Test fetching data from Kafka."""
        object_id = "test-sat-001"
        
        # Call the method
        result = await ccdm_service_instance._fetch_kafka_data(object_id)
        
        # Verify the result structure
        assert "recent_observations" in result
        assert "recent_conjunctions" in result
        assert "recent_rf_events" in result
        assert "last_update" in result
    
    async def test_get_object_data(self, ccdm_service_instance):
        """Test getting comprehensive object data."""
        object_id = "test-sat-001"
        
        # Patch the individual fetch methods
        with patch.object(ccdm_service_instance, '_fetch_udl_data', 
                        return_value={"state_vector": {}, "maneuver_history": []}), \
             patch.object(ccdm_service_instance, '_fetch_space_track_data',
                        return_value={"space_track_data": {}}), \
             patch.object(ccdm_service_instance, '_fetch_tmdb_data',
                        return_value={"baseline_signatures": {}}), \
             patch.object(ccdm_service_instance, '_fetch_kafka_data',
                        return_value={"recent_observations": []}):
            
            # Call the method
            result = await ccdm_service_instance._get_object_data(object_id)
            
            # Verify the result
            assert result["object_id"] == object_id
            assert "state_vector" in result
            assert "maneuver_history" in result
            assert "rf_history" in result
            
            # Check that the result was cached
            assert object_id in ccdm_service_instance._object_data_cache
            assert object_id in ccdm_service_instance._cache_timestamps
    
    async def test_safe_fetch(self, ccdm_service_instance):
        """Test safe fetch method for handling exceptions."""
        # Test successful synchronous function
        def success_func():
            return {"success": True}
        
        result = await ccdm_service_instance._safe_fetch(success_func)
        assert result == {"success": True}
        
        # Test successful async function
        async def async_success_func():
            return {"async_success": True}
        
        result = await ccdm_service_instance._safe_fetch(async_success_func)
        assert result == {"async_success": True}
        
        # Test function that raises exception
        def error_func():
            raise ValueError("Test error")
        
        result = await ccdm_service_instance._safe_fetch(error_func)
        assert isinstance(result, ValueError)
        assert str(result) == "Test error"
        
        # Test async function that raises exception
        async def async_error_func():
            raise ValueError("Async test error")
        
        result = await ccdm_service_instance._safe_fetch(async_error_func)
        assert isinstance(result, ValueError)
        assert str(result) == "Async test error"


@pytest.mark.asyncio
class TestCCDMServiceAnalysis:
    """Test CCDM analysis methods."""
    
    async def test_analyze_object(self, ccdm_service_instance, mock_space_object_data):
        """Test analyzing an object."""
        object_id = "test-sat-001"
        
        # Patch the data fetching and analysis methods
        with patch.object(ccdm_service_instance, '_get_object_data', 
                        return_value=mock_space_object_data), \
             patch.object(ccdm_service_instance, '_analyze_shape_changes',
                        return_value={"detected": True, "confidence": 0.85}), \
             patch.object(ccdm_service_instance, '_analyze_thermal_signature',
                        return_value={"detected": True, "confidence": 0.8}), \
             patch.object(ccdm_service_instance, '_analyze_propulsive_capabilities',
                        return_value={"detected": True, "confidence": 0.9, 
                                      "propulsion_type": PropulsionType.CHEMICAL}):
            
            # Call the method
            result = await ccdm_service_instance.analyze_object(object_id)
            
            # Verify the result
            assert isinstance(result, ObjectAnalysisResponse)
            assert result.object_id == object_id
            assert result.analysis_complete is True
            assert result.confidence_score > 0.8
            assert result.shape_change.detected is True
            assert result.thermal_signature.detected is True
            assert result.propulsive_capability.detected is True
    
    async def test_analyze_shape_changes(self, ccdm_service_instance, mock_space_object_data):
        """Test analyzing shape changes."""
        object_id = "test-sat-001"
        
        # Make the radar signature significantly different from baseline
        test_data = mock_space_object_data.copy()
        test_data["radar_signature"]["rcs"] = 2.5  # Baseline mean is 1.2, std is 0.3
        
        # Call the method
        result = await ccdm_service_instance._analyze_shape_changes(object_id, test_data)
        
        # Verify the result
        assert "detected" in result
        assert "confidence" in result
        assert "radar_change" in result
        assert result["detected"] is True
        assert result["radar_change"] is True
        assert result["confidence"] > 0.7
    
    async def test_analyze_thermal_signature(self, ccdm_service_instance, mock_space_object_data):
        """Test analyzing thermal signature."""
        object_id = "test-sat-001"
        
        # Add a recent maneuver event
        test_data = mock_space_object_data.copy()
        test_data["object_events"].insert(0, {
            "type": "maneuver",
            "time": datetime.utcnow().isoformat(),
            "data": {"delta_v": 1.2}
        })
        
        # Call the method
        result = await ccdm_service_instance._analyze_thermal_signature(object_id, test_data)
        
        # Verify the result
        assert "detected" in result
        assert "confidence" in result
        assert "temperature_kelvin" in result
        assert "anomaly_score" in result
        assert result["detected"] is True
        assert result["confidence"] > 0.7
        assert result["anomaly_score"] > 0.5
    
    async def test_analyze_propulsive_capabilities(self, ccdm_service_instance, mock_space_object_data):
        """Test analyzing propulsive capabilities."""
        object_id = "test-sat-001"
        
        # Add high delta-v maneuvers indicating chemical propulsion
        test_data = mock_space_object_data.copy()
        test_data["maneuver_history"] = [
            {
                "time": (datetime.utcnow() - timedelta(days=i*7)).isoformat(),
                "delta_v": 0.6,
                "thrust_vector": {"x": 0.1, "y": 0.2, "z": 0.3},
                "duration": 60.0,
                "confidence": 0.85
            } for i in range(0, 3)
        ]
        
        # Call the method
        result = await ccdm_service_instance._analyze_propulsive_capabilities(object_id, test_data)
        
        # Verify the result
        assert "detected" in result
        assert "confidence" in result
        assert "propulsion_type" in result
        assert result["detected"] is True
        assert result["propulsion_type"] == PropulsionType.CHEMICAL
        assert result["confidence"] > 0.7
    
    async def test_detect_shape_changes(self, ccdm_service_instance, mock_space_object_data):
        """Test the detect_shape_changes method."""
        object_id = "test-sat-001"
        start_time = datetime.utcnow() - timedelta(days=7)
        end_time = datetime.utcnow()
        
        # Patch the data fetching and calculation methods
        with patch.object(ccdm_service_instance, '_get_object_data', 
                        return_value=mock_space_object_data), \
             patch.object(ccdm_service_instance, '_calculate_shape_metrics',
                        return_value=ShapeChangeMetrics(
                            volume_change=0.3,
                            surface_area_change=0.4,
                            aspect_ratio_change=0.15,
                            confidence=0.85
                        )):
            
            # Call the method
            result = await ccdm_service_instance.detect_shape_changes(object_id, start_time, end_time)
            
            # Verify the result
            assert isinstance(result, ShapeChangeResponse)
            assert result.detected is True
            assert result.confidence > 0.7
            assert isinstance(result.timestamp, datetime)
            assert hasattr(result, 'metrics')
            assert result.metrics.volume_change == 0.3
            assert result.metrics.surface_area_change == 0.4
            assert result.metrics.aspect_ratio_change == 0.15
    
    async def test_assess_thermal_signature(self, ccdm_service_instance, mock_space_object_data):
        """Test the assess_thermal_signature method."""
        object_id = "test-sat-001"
        timestamp = datetime.utcnow()
        
        # Patch the data fetching and calculation methods
        with patch.object(ccdm_service_instance, '_get_object_data', 
                        return_value=mock_space_object_data), \
             patch.object(ccdm_service_instance, '_calculate_thermal_metrics',
                        return_value=ThermalSignatureMetrics(
                            temperature_kelvin=285.0,
                            anomaly_score=0.75
                        )):
            
            # Call the method
            result = await ccdm_service_instance.assess_thermal_signature(object_id, timestamp)
            
            # Verify the result
            assert isinstance(result, ThermalSignatureResponse)
            assert result.detected is True
            assert result.confidence > 0.0
            assert isinstance(result.timestamp, datetime)
            assert hasattr(result, 'metrics')
            assert result.metrics.temperature_kelvin == 285.0
            assert result.metrics.anomaly_score == 0.75
    
    async def test_evaluate_propulsive_capabilities(self, ccdm_service_instance, mock_space_object_data):
        """Test the evaluate_propulsive_capabilities method."""
        object_id = "test-sat-001"
        analysis_period = 30
        
        # Patch the data fetching and analysis methods
        with patch.object(ccdm_service_instance, '_get_object_data', 
                        return_value=mock_space_object_data), \
             patch.object(ccdm_service_instance, '_analyze_propulsive_capabilities',
                        return_value={
                            "detected": True,
                            "confidence": 0.8,
                            "propulsion_type": PropulsionType.CHEMICAL,
                            "thrust_estimate": 10.5,
                            "fuel_reserve_estimate": 65.0
                        }):
            
            # Call the method
            result = await ccdm_service_instance.evaluate_propulsive_capabilities(object_id, analysis_period)
            
            # Verify the result
            assert isinstance(result, PropulsiveCapabilityResponse)
            assert result.detected is True
            assert result.confidence > 0.7
            assert isinstance(result.timestamp, datetime)
            assert hasattr(result, 'metrics')
            assert result.metrics.type == PropulsionType.CHEMICAL
            assert result.metrics.thrust_estimate == 10.5
            assert result.metrics.fuel_reserve_estimate == 65.0
    
    async def test_get_historical_analysis(self, ccdm_service_instance, mock_space_object_data):
        """Test the get_historical_analysis method."""
        object_id = "test-sat-001"
        
        # Patch the data fetching and analysis methods
        with patch.object(ccdm_service_instance, '_get_object_data', 
                        return_value=mock_space_object_data), \
             patch.object(ccdm_service_instance, '_generate_historical_analyses',
                        return_value=[
                            HistoricalAnalysis(
                                object_id=object_id,
                                time_range={
                                    "start": datetime.utcnow() - timedelta(days=5),
                                    "end": datetime.utcnow()
                                },
                                patterns=[{"type": "orbital", "confidence": 0.8}],
                                trend_analysis={"stability": 0.9},
                                anomalies=[{"type": "maneuver", "confidence": 0.7}]
                            )
                        ]):
            
            # Call the method
            result = await ccdm_service_instance.get_historical_analysis(object_id)
            
            # Verify the result
            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], HistoricalAnalysis)
            assert result[0].object_id == object_id
            assert len(result[0].patterns) > 0
            assert len(result[0].anomalies) > 0
            assert len(result[0].trend_analysis) > 0
    
    async def test_get_ccdm_assessment(self, ccdm_service_instance, mock_space_object_data, mock_analyzers):
        """Test the get_ccdm_assessment method."""
        object_id = "test-sat-001"
        
        # Configure mock analyzers with test return values
        mock_analyzers["stability"].analyze_stability.return_value = {"stability_changed": True, "confidence": 0.8}
        mock_analyzers["maneuver"].analyze_maneuvers.return_value = {"maneuvers_detected": True, "pol_violation": False, "confidence": 0.85}
        
        # Patch the data fetching method
        with patch.object(ccdm_service_instance, '_get_object_data', 
                        return_value=mock_space_object_data):
            
            # Call the method
            result = await ccdm_service_instance.get_ccdm_assessment(object_id)
            
            # Verify the result
            assert isinstance(result, CCDMAssessment)
            assert result.object_id == object_id
            assert result.assessment_type == "automated_ccdm_indicators"
            assert isinstance(result.timestamp, datetime)
            assert isinstance(result.results, dict)
            assert isinstance(result.confidence_level, float)
            assert isinstance(result.triggered_indicators, list)
            assert isinstance(result.recommendations, list)
            assert "stability" in result.triggered_indicators
    
    async def test_get_anomaly_detections(self, ccdm_service_instance):
        """Test the get_anomaly_detections method."""
        object_id = "test-sat-001"
        days = 30
        
        # Call the method
        result = await ccdm_service_instance.get_anomaly_detections(object_id, days)
        
        # Verify the result
        assert isinstance(result, list)
        for anomaly in result:
            assert isinstance(anomaly, AnomalyDetection)
            assert anomaly.object_id == object_id
            assert isinstance(anomaly.timestamp, datetime)
            assert isinstance(anomaly.anomaly_type, str)
            assert isinstance(anomaly.details, dict)
            assert isinstance(anomaly.confidence, float)
            assert isinstance(anomaly.recommended_actions, list)


@pytest.mark.asyncio
class TestCCDMServiceErrorHandling:
    """Test error handling in CCDM service."""
    
    async def test_analyze_object_with_error(self, ccdm_service_instance):
        """Test error handling in analyze_object method."""
        object_id = "test-sat-001"
        
        # Make _get_object_data raise an exception
        with patch.object(ccdm_service_instance, '_get_object_data', 
                        side_effect=Exception("Test error")):
            
            # Call the method
            result = await ccdm_service_instance.analyze_object(object_id)
            
            # Verify error response
            assert isinstance(result, ObjectAnalysisResponse)
            assert result.object_id == object_id
            assert result.analysis_complete is False
            assert result.confidence_score == 0.0
            assert hasattr(result, 'error')
            assert "Test error" in result.error
    
    async def test_detect_shape_changes_with_error(self, ccdm_service_instance):
        """Test error handling in detect_shape_changes method."""
        object_id = "test-sat-001"
        start_time = datetime.utcnow() - timedelta(days=7)
        end_time = datetime.utcnow()
        
        # Make _get_object_data raise an exception
        with patch.object(ccdm_service_instance, '_get_object_data', 
                        side_effect=Exception("Test error")):
            
            # Call the method
            result = await ccdm_service_instance.detect_shape_changes(object_id, start_time, end_time)
            
            # Verify error response
            assert isinstance(result, ShapeChangeResponse)
            assert result.detected is False
            assert result.confidence == 0.0
            assert hasattr(result, 'error')
            assert "Test error" in result.error
    
    async def test_assess_thermal_signature_with_error(self, ccdm_service_instance):
        """Test error handling in assess_thermal_signature method."""
        object_id = "test-sat-001"
        timestamp = datetime.utcnow()
        
        # Make _get_object_data raise an exception
        with patch.object(ccdm_service_instance, '_get_object_data', 
                        side_effect=Exception("Test error")):
            
            # Call the method
            result = await ccdm_service_instance.assess_thermal_signature(object_id, timestamp)
            
            # Verify error response
            assert isinstance(result, ThermalSignatureResponse)
            assert result.detected is False
            assert result.confidence == 0.0
            assert hasattr(result, 'error')
            assert "Test error" in result.error
    
    async def test_evaluate_propulsive_capabilities_with_error(self, ccdm_service_instance):
        """Test error handling in evaluate_propulsive_capabilities method."""
        object_id = "test-sat-001"
        analysis_period = 30
        
        # Make _get_object_data raise an exception
        with patch.object(ccdm_service_instance, '_get_object_data', 
                        side_effect=Exception("Test error")):
            
            # Call the method
            result = await ccdm_service_instance.evaluate_propulsive_capabilities(object_id, analysis_period)
            
            # Verify error response
            assert isinstance(result, PropulsiveCapabilityResponse)
            assert result.detected is False
            assert result.confidence == 0.0
            assert hasattr(result, 'error')
            assert "Test error" in result.error
    
    async def test_get_historical_analysis_with_error(self, ccdm_service_instance):
        """Test error handling in get_historical_analysis method."""
        object_id = "test-sat-001"
        
        # Make _get_object_data raise an exception
        with patch.object(ccdm_service_instance, '_get_object_data', 
                        side_effect=Exception("Test error")):
            
            # Call the method
            result = await ccdm_service_instance.get_historical_analysis(object_id)
            
            # Verify error response
            assert isinstance(result, list)
            assert len(result) == 0


class TestCCDMServiceDataFormatting:
    """Test data formatting methods in CCDM service."""
    
    def test_format_maneuver_data(self, ccdm_service_instance):
        """Test formatting maneuver data."""
        raw_maneuvers = [
            {"time": "2023-01-01T00:00:00Z", "delta_v": 0.5},
            {"epoch": "2023-01-02T00:00:00Z", "delta_v": 0.6, "thrust_vector": {"x": 0.1, "y": 0.2, "z": 0.3}},
            {"timestamp": "2023-01-03T00:00:00Z", "delta_v": 0.7}
        ]
        
        # Call the method
        result = ccdm_service_instance._format_maneuver_data(raw_maneuvers)
        
        # Verify the result
        assert len(result) == 3
        assert "time" in result[0]
        assert "delta_v" in result[0]
        assert "thrust_vector" in result[0]
        assert "duration" in result[0]
        assert "confidence" in result[0]
        
        # Check that different time fields are handled correctly
        assert result[0]["time"] == "2023-01-01T00:00:00Z"
        assert result[1]["time"] == "2023-01-02T00:00:00Z"
        assert result[2]["time"] == "2023-01-03T00:00:00Z"
    
    def test_format_rf_data(self, ccdm_service_instance):
        """Test formatting RF data."""
        raw_rf = {
            "measurements": [
                {"time": "2023-01-01T00:00:00Z", "frequency": 2200.0, "power_level": -90.0},
                {"timestamp": "2023-01-02T00:00:00Z", "frequency": 8400.0, "power_level": -95.0, "bandwidth": 10.0},
                {"time": "2023-01-03T00:00:00Z", "frequency": 4500.0, "power_level": -85.0, "duration": 60.0}
            ]
        }
        
        # Call the method
        result = ccdm_service_instance._format_rf_data(raw_rf)
        
        # Verify the result
        assert len(result) == 3
        assert "time" in result[0]
        assert "frequency" in result[0]
        assert "power" in result[0]
        assert "duration" in result[0]
        assert "bandwidth" in result[0]
        assert "confidence" in result[0]
        
        # Check that different time fields are handled correctly
        assert result[0]["time"] == "2023-01-01T00:00:00Z"
        assert result[1]["time"] == "2023-01-02T00:00:00Z"
        assert result[2]["time"] == "2023-01-03T00:00:00Z"
        
        # Check that power_level is mapped to power
        assert result[0]["power"] == -90.0
        assert result[1]["power"] == -95.0
        assert result[2]["power"] == -85.0
    
    def test_extract_orbit_data(self, ccdm_service_instance):
        """Test extracting orbit data."""
        elset_data = {
            "semi_major_axis": 7000.0,
            "eccentricity": 0.001,
            "inclination": 51.6,
            "raan": 120.0,
            "arg_perigee": 180.0,
            "mean_anomaly": 0.0,
            "mean_motion": 15.5
        }
        
        state_vector = {
            "position": {"x": 1000.0, "y": 2000.0, "z": 3000.0},
            "velocity": {"x": 1.0, "y": 2.0, "z": 3.0},
            "epoch": "2023-01-01T00:00:00Z"
        }
        
        # Call the method
        result = ccdm_service_instance._extract_orbit_data(elset_data, state_vector)
        
        # Verify the result
        assert "semi_major_axis" in result
        assert "eccentricity" in result
        assert "inclination" in result
        assert "raan" in result
        assert "arg_perigee" in result
        assert "mean_anomaly" in result
        assert "mean_motion" in result
        assert "position_vector" in result
        assert "velocity_vector" in result
        assert "epoch" in result
        
        # Check values
        assert result["semi_major_axis"] == 7000.0
        assert result["eccentricity"] == 0.001
        assert result["inclination"] == 51.6
        assert result["position_vector"] == {"x": 1000.0, "y": 2000.0, "z": 3000.0}
        assert result["velocity_vector"] == {"x": 1.0, "y": 2.0, "z": 3.0}
        assert result["epoch"] == "2023-01-01T00:00:00Z"