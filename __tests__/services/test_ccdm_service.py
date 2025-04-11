import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import pytest
import asyncio
from datetime import datetime, timedelta
import json
import os
import sys

# Add the parent directory to sys.path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.app.services.ccdm import CCDMService
from backend.app.models.ccdm import (
    ObjectAnalysisResponse,
    ShapeChangeResponse,
    ThermalSignatureResponse,
    PropulsiveCapabilityResponse,
    HistoricalAnalysis,
    PropulsionType,
    PropulsionMetrics,
    ShapeChangeMetrics,
    ThermalSignatureMetrics,
    CCDMAssessment,
    AnomalyDetection
)

@pytest.mark.asyncio
class TestCCDMService:
    """Test class for CCDMService"""

    async def setup_method(self):
        """Setup method that runs before each test"""
        # Create a mock config with test values
        self.test_config = {
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

        # Create the CCDMService with our test config
        with patch('src.asttroshield.api_client.udl_client.UDLClient'), \
             patch('src.asttroshield.udl_integration.USSFUDLIntegrator'), \
             patch('src.kafka_client.kafka_consume.KafkaConsumer'), \
             patch('ml.models.anomaly_detector.SpaceObjectAnomalyDetector'), \
             patch('ml.models.track_evaluator.TrackEvaluator'):
            self.service = CCDMService(config=self.test_config)
            
            # Mock all analyzer classes
            for analyzer_attr in [
                'stability_analyzer', 'maneuver_analyzer', 'rf_analyzer',
                'subsatellite_analyzer', 'itu_checker', 'disagreement_checker',
                'orbit_analyzer', 'signature_analyzer', 'stimulation_analyzer',
                'amr_analyzer', 'imaging_analyzer', 'launch_analyzer',
                'eclipse_analyzer', 'registry_checker'
            ]:
                if hasattr(self.service, analyzer_attr):
                    setattr(self.service, analyzer_attr, MagicMock())

    async def test_initialization(self):
        """Test that the service initializes correctly"""
        assert self.service._config is not None
        assert hasattr(self.service, 'udl_client')
        assert hasattr(self.service, 'udl_integrator')
        assert hasattr(self.service, 'space_track_client')
        assert hasattr(self.service, 'tmdb_client')
        assert hasattr(self.service, 'kafka_consumer')
        assert hasattr(self.service, 'stability_analyzer')
        assert len(self.service._object_data_cache) == 0

    @patch('backend.app.services.ccdm.CCDMService._get_object_data')
    async def test_analyze_object(self, mock_get_object_data):
        """Test the analyze_object method"""
        # Setup mock object data
        mock_object_data = {
            "object_id": "12345",
            "state_vector": {"position": {"x": 1, "y": 2, "z": 3}, "velocity": {"x": 0.1, "y": 0.2, "z": 0.3}},
            "state_history": [{"time": "2023-01-01T00:00:00Z", "position": {}, "velocity": {}}],
            "maneuver_history": [{"time": "2023-01-01T01:00:00Z", "delta_v": 0.5}],
            "rf_history": [{"time": "2023-01-01T02:00:00Z", "frequency": 2200.0, "power": -90.0}],
            "radar_signature": {"rcs": 1.5, "confidence": 0.8},
            "optical_signature": {"magnitude": 6.5, "confidence": 0.7},
            "baseline_signatures": {
                "radar": {"rcs_mean": 1.0, "rcs_std": 0.2},
                "optical": {"magnitude_mean": 7.0, "magnitude_std": 0.5}
            },
            "orbit_data": {"semi_major_axis": 7000.0, "inclination": 51.6, "eccentricity": 0.001},
            "anomaly_baseline": {"thermal_profile": [270, 280, 275, 265]}
        }
        mock_get_object_data.return_value = mock_object_data
        
        # Mock the analysis methods
        with patch.object(self.service, '_analyze_shape_changes', return_value={"detected": True, "confidence": 0.8}), \
             patch.object(self.service, '_analyze_thermal_signature', return_value={"detected": True, "confidence": 0.75}), \
             patch.object(self.service, '_analyze_propulsive_capabilities', return_value={"detected": True, "confidence": 0.9, "propulsion_type": PropulsionType.CHEMICAL}):
            
            # Call the analyze_object method
            response = await self.service.analyze_object("12345")
            
            # Verify the response
            assert isinstance(response, ObjectAnalysisResponse)
            assert response.object_id == "12345"
            assert response.analysis_complete is True
            assert response.confidence_score > 0
            assert response.shape_change.detected is True
            assert response.thermal_signature.detected is True
            assert response.propulsive_capability.detected is True

    @patch('backend.app.services.ccdm.CCDMService._get_object_data')
    async def test_analyze_object_with_error(self, mock_get_object_data):
        """Test error handling in analyze_object method"""
        # Make _get_object_data raise an exception
        mock_get_object_data.side_effect = Exception("Test error")
        
        # Call the analyze_object method
        response = await self.service.analyze_object("12345")
        
        # Verify error response
        assert isinstance(response, ObjectAnalysisResponse)
        assert response.object_id == "12345"
        assert response.analysis_complete is False
        assert response.confidence_score == 0.0
        assert hasattr(response, 'error')

    @patch('backend.app.services.ccdm.CCDMService._get_object_data')
    async def test_detect_shape_changes(self, mock_get_object_data):
        """Test the detect_shape_changes method"""
        # Setup mock object data with significant shape changes
        mock_object_data = {
            "object_id": "12345",
            "radar_signature": {"rcs": 2.0, "confidence": 0.9},
            "optical_signature": {"magnitude": 5.0, "confidence": 0.8},
            "baseline_signatures": {
                "radar": {"rcs_mean": 1.0, "rcs_std": 0.2},
                "optical": {"magnitude_mean": 7.0, "magnitude_std": 0.5}
            },
            "object_events": [
                {"type": "deployment", "time": "2023-01-01T01:00:00Z", "data": {"component": "solar_panel"}}
            ]
        }
        mock_get_object_data.return_value = mock_object_data
        
        # Call the detect_shape_changes method
        start_time = datetime.utcnow() - timedelta(days=7)
        end_time = datetime.utcnow()
        response = await self.service.detect_shape_changes("12345", start_time, end_time)
        
        # Verify the response
        assert isinstance(response, ShapeChangeResponse)
        assert response.detected is True
        assert response.confidence > 0.5
        assert isinstance(response.timestamp, datetime)

    @patch('backend.app.services.ccdm.CCDMService._get_object_data')
    async def test_assess_thermal_signature(self, mock_get_object_data):
        """Test the assess_thermal_signature method"""
        # Setup mock object data with thermal anomalies
        mock_object_data = {
            "object_id": "12345",
            "anomaly_baseline": {"thermal_profile": [270, 280, 275, 265]},
            "object_events": [
                {"type": "maneuver", "time": (datetime.utcnow() - timedelta(hours=1)).isoformat(),
                 "data": {"delta_v": 0.8}}
            ]
        }
        mock_get_object_data.return_value = mock_object_data
        
        # Call the assess_thermal_signature method
        response = await self.service.assess_thermal_signature("12345", datetime.utcnow())
        
        # Verify the response
        assert isinstance(response, ThermalSignatureResponse)
        assert isinstance(response.confidence, float)
        assert isinstance(response.timestamp, datetime)

    @patch('backend.app.services.ccdm.CCDMService._get_object_data')
    async def test_evaluate_propulsive_capabilities(self, mock_get_object_data):
        """Test the evaluate_propulsive_capabilities method"""
        # Setup mock object data with propulsive evidence
        mock_object_data = {
            "object_id": "12345",
            "maneuver_history": [
                {"time": "2023-01-01T01:00:00Z", "delta_v": 0.6},
                {"time": "2023-01-05T02:00:00Z", "delta_v": 0.7},
                {"time": "2023-01-10T03:00:00Z", "delta_v": 0.5}
            ]
        }
        mock_get_object_data.return_value = mock_object_data
        
        # Call the evaluate_propulsive_capabilities method
        response = await self.service.evaluate_propulsive_capabilities("12345", 30)
        
        # Verify the response
        assert isinstance(response, PropulsiveCapabilityResponse)
        assert response.detected is True
        assert response.confidence > 0.5
        assert isinstance(response.timestamp, datetime)
        if hasattr(response, 'metrics') and response.metrics:
            assert isinstance(response.metrics.type, PropulsionType)

    @patch('backend.app.services.ccdm.CCDMService._get_object_data')
    async def test_get_historical_analysis(self, mock_get_object_data):
        """Test the get_historical_analysis method"""
        # Setup mock object data
        mock_object_data = {
            "object_id": "12345",
            "object_events": [
                {"type": "maneuver", "time": "2023-01-01T01:00:00Z", "data": {"delta_v": 0.5}},
                {"type": "rf_emission", "time": "2023-01-02T02:00:00Z", "data": {"power": -85.0}}
            ]
        }
        mock_get_object_data.return_value = mock_object_data
        
        # Call the get_historical_analysis method
        response = await self.service.get_historical_analysis("12345")
        
        # Verify the response
        assert isinstance(response, list)
        assert len(response) > 0
        for analysis in response:
            assert isinstance(analysis, HistoricalAnalysis)
            assert analysis.object_id == "12345"
            assert "start" in analysis.time_range
            assert "end" in analysis.time_range
            assert isinstance(analysis.patterns, list)
            assert isinstance(analysis.trend_analysis, dict)
            assert isinstance(analysis.anomalies, list)

    @patch('backend.app.services.ccdm.CCDMService._get_object_data')
    async def test_get_ccdm_assessment(self, mock_get_object_data):
        """Test the get_ccdm_assessment method"""
        # Setup mock object data
        mock_object_data = {
            "object_id": "12345",
            "state_history": [{"time": "2023-01-01T00:00:00Z", "position": {}, "velocity": {}}],
            "maneuver_history": [{"time": "2023-01-01T01:00:00Z", "delta_v": 0.5}],
            "rf_history": [{"time": "2023-01-01T02:00:00Z", "frequency": 2200.0, "power": -90.0}],
            "baseline_pol": {"rf": {"max_power": -95.0}, "maneuvers": {"typical_delta_v": 0.2}}
        }
        mock_get_object_data.return_value = mock_object_data
        
        # Mock analyzer methods to return test results
        for analyzer_attr in [
            'stability_analyzer', 'maneuver_analyzer', 'rf_analyzer',
            'subsatellite_analyzer', 'itu_checker', 'disagreement_checker',
            'orbit_analyzer', 'signature_analyzer', 'stimulation_analyzer', 
            'amr_analyzer', 'imaging_analyzer', 'launch_analyzer',
            'eclipse_analyzer', 'registry_checker'
        ]:
            if hasattr(self.service, analyzer_attr):
                analyzer = getattr(self.service, analyzer_attr)
                # Create a mock result with a random detection flag and confidence
                analyzer.analyze_stability.return_value = {"stability_changed": True, "confidence": 0.8}
                analyzer.analyze_maneuvers.return_value = {"maneuvers_detected": True, "pol_violation": False, "confidence": 0.85}
                analyzer.analyze_rf_pattern.return_value = {"rf_detected": True, "pol_violation": False, "confidence": 0.75}
                analyzer.detect_sub_satellites.return_value = {"subsatellites_detected": False, "confidence": 0.8}
                analyzer.check_itu_compliance.return_value = {"violates_filing": False, "confidence": 0.9}
                analyzer.check_disagreements.return_value = {"class_disagreement": False, "confidence": 0.85}
                analyzer.analyze_orbit.return_value = {"orbit_out_of_family": False, "confidence": 0.8}
                analyzer.analyze_signatures.return_value = {"optical_out_of_family": False, "confidence": 0.7}
                analyzer.analyze_stimulation.return_value = {"stimulation_detected": False, "confidence": 0.85}
                analyzer.analyze_amr.return_value = {"amr_out_of_family": False, "confidence": 0.9}
                analyzer.analyze_imaging_maneuvers.return_value = {"imaging_maneuver_detected": False, "confidence": 0.8}
                analyzer.analyze_launch.return_value = {"suspicious_source": False, "confidence": 0.9}
                analyzer.analyze_eclipse_behavior.return_value = {"uct_during_eclipse": False, "confidence": 0.85}
                analyzer.check_registry.return_value = {"registered": True, "confidence": 0.95}
        
        # Call the get_ccdm_assessment method
        response = await self.service.get_ccdm_assessment("12345")
        
        # Verify the response
        assert isinstance(response, CCDMAssessment)
        assert response.object_id == "12345"
        assert response.assessment_type == "automated_ccdm_indicators"
        assert isinstance(response.timestamp, datetime)
        assert isinstance(response.results, dict)
        assert isinstance(response.triggered_indicators, list)
        assert isinstance(response.confidence_level, float)
        assert isinstance(response.recommendations, list)

    @patch('backend.app.services.ccdm.CCDMService._get_object_data')
    async def test_get_anomaly_detections(self, mock_get_object_data):
        """Test the get_anomaly_detections method"""
        # Setup mock object data
        mock_object_data = {
            "object_id": "12345"
        }
        mock_get_object_data.return_value = mock_object_data
        
        # Call the get_anomaly_detections method
        response = await self.service.get_anomaly_detections("12345", 30)
        
        # Verify the response
        assert isinstance(response, list)
        for anomaly in response:
            assert isinstance(anomaly, AnomalyDetection)
            assert anomaly.object_id == "12345"
            assert isinstance(anomaly.timestamp, datetime)
            assert isinstance(anomaly.anomaly_type, str)
            assert isinstance(anomaly.details, dict)
            assert isinstance(anomaly.confidence, float)
            assert isinstance(anomaly.recommended_actions, list)

    async def test_data_fetch_methods(self):
        """Test the data fetching methods"""
        test_object_id = "12345"
        
        # Test UDL data fetching
        with patch.object(self.service.udl_client, 'get_state_vector', return_value={"position": {}, "velocity": {}}), \
             patch.object(self.service.udl_client, 'get_state_vector_history', return_value=[]), \
             patch.object(self.service.udl_client, 'get_elset_data', return_value={}), \
             patch.object(self.service.udl_client, 'get_maneuver_data', return_value=[]), \
             patch.object(self.service.udl_client, 'get_conjunction_data', return_value={}), \
             patch.object(self.service.udl_client, 'get_rf_interference', return_value={}):
            
            udl_data = await self.service._fetch_udl_data(test_object_id)
            assert isinstance(udl_data, dict)
            assert "state_vector" in udl_data
            assert "state_history" in udl_data
            assert "maneuver_history" in udl_data
            assert "rf_history" in udl_data
        
        # Test Space Track data fetching
        space_track_data = await self.service._fetch_space_track_data(test_object_id)
        assert isinstance(space_track_data, dict)
        assert "space_track_data" in space_track_data
        
        # Test TMDB data fetching
        tmdb_data = await self.service._fetch_tmdb_data(test_object_id)
        assert isinstance(tmdb_data, dict)
        assert "baseline_signatures" in tmdb_data
        
        # Test Kafka data fetching
        kafka_data = await self.service._fetch_kafka_data(test_object_id)
        assert isinstance(kafka_data, dict)
        assert "recent_observations" in kafka_data

    async def test_safe_fetch(self):
        """Test the _safe_fetch method"""
        # Test successful fetch
        test_func = lambda: {"test": "data"}
        result = await self.service._safe_fetch(test_func)
        assert result == {"test": "data"}
        
        # Test failed fetch
        error_func = lambda: {}
        error_func.__name__ = "error_func"
        error_func = lambda: exec('raise ValueError("Test error")')
        result = await self.service._safe_fetch(error_func)
        assert isinstance(result, Exception)
        
        # Test async fetch
        async def async_func():
            return {"test": "async data"}
        
        result = await self.service._safe_fetch(async_func)
        assert result == {"test": "async data"}

    async def test_data_formatting_methods(self):
        """Test data formatting methods"""
        # Test maneuver data formatting
        raw_maneuvers = [
            {"time": "2023-01-01T00:00:00Z", "delta_v": 0.5},
            {"epoch": "2023-01-02T00:00:00Z", "delta_v": 0.6},
        ]
        formatted_maneuvers = self.service._format_maneuver_data(raw_maneuvers)
        assert len(formatted_maneuvers) == 2
        assert "time" in formatted_maneuvers[0]
        assert "delta_v" in formatted_maneuvers[0]
        
        # Test RF data formatting
        raw_rf = {
            "measurements": [
                {"time": "2023-01-01T00:00:00Z", "frequency": 2200.0, "power_level": -90.0},
                {"timestamp": "2023-01-02T00:00:00Z", "frequency": 8400.0, "power_level": -95.0},
            ]
        }
        formatted_rf = self.service._format_rf_data(raw_rf)
        assert len(formatted_rf) == 2
        assert "time" in formatted_rf[0]
        assert "frequency" in formatted_rf[0]
        assert "power" in formatted_rf[0]

    async def test_analyze_shape_changes(self):
        """Test the _analyze_shape_changes method directly"""
        test_object_id = "12345"
        
        # Test with significant RCS change
        test_data_rcs_change = {
            "radar_signature": {"rcs": 2.0, "confidence": 0.9},
            "baseline_signatures": {
                "radar": {"rcs_mean": 1.0, "rcs_std": 0.2}
            }
        }
        result = await self.service._analyze_shape_changes(test_object_id, test_data_rcs_change)
        assert result["detected"] is True
        assert result["confidence"] > 0.5
        assert result["radar_change"] is True
        
        # Test with no significant changes
        test_data_no_change = {
            "radar_signature": {"rcs": 1.1, "confidence": 0.9},
            "optical_signature": {"magnitude": 7.1, "confidence": 0.8},
            "baseline_signatures": {
                "radar": {"rcs_mean": 1.0, "rcs_std": 0.2},
                "optical": {"magnitude_mean": 7.0, "magnitude_std": 0.5}
            }
        }
        result = await self.service._analyze_shape_changes(test_object_id, test_data_no_change)
        assert result["detected"] is False
        assert result["confidence"] > 0.5

    async def test_analyze_thermal_signature(self):
        """Test the _analyze_thermal_signature method directly"""
        test_object_id = "12345"
        
        # Test with recent maneuver (should cause thermal anomaly)
        test_data_maneuver = {
            "anomaly_baseline": {"thermal_profile": [270, 280, 275, 265]},
            "object_events": [
                {"type": "maneuver", "time": datetime.utcnow().isoformat(),
                 "data": {"delta_v": 0.8}}
            ]
        }
        result = await self.service._analyze_thermal_signature(test_object_id, test_data_maneuver)
        assert result["detected"] is True
        assert result["confidence"] > 0.5
        assert result["temperature_kelvin"] > 270
        
        # Test with no recent events
        test_data_no_events = {
            "anomaly_baseline": {"thermal_profile": [270, 280, 275, 265]},
            "object_events": []
        }
        result = await self.service._analyze_thermal_signature(test_object_id, test_data_no_events)
        assert result["detected"] is False
        assert result["confidence"] > 0.0

    async def test_analyze_propulsive_capabilities(self):
        """Test the _analyze_propulsive_capabilities method directly"""
        test_object_id = "12345"
        
        # Test with multiple large maneuvers (should indicate chemical propulsion)
        test_data_chemical = {
            "maneuver_history": [
                {"time": "2023-01-01T01:00:00Z", "delta_v": 0.6},
                {"time": "2023-01-05T02:00:00Z", "delta_v": 0.7},
                {"time": "2023-01-10T03:00:00Z", "delta_v": 0.5}
            ]
        }
        result = await self.service._analyze_propulsive_capabilities(test_object_id, test_data_chemical)
        assert result["detected"] is True
        assert result["propulsion_type"] == PropulsionType.CHEMICAL
        assert result["confidence"] > 0.5
        
        # Test with small maneuvers (should indicate electric propulsion)
        test_data_electric = {
            "maneuver_history": [
                {"time": "2023-01-01T01:00:00Z", "delta_v": 0.02},
                {"time": "2023-01-05T02:00:00Z", "delta_v": 0.03},
                {"time": "2023-01-10T03:00:00Z", "delta_v": 0.04}
            ]
        }
        result = await self.service._analyze_propulsive_capabilities(test_object_id, test_data_electric)
        assert result["detected"] is True
        assert result["propulsion_type"] == PropulsionType.ELECTRIC
        assert result["confidence"] > 0.5
        
        # Test with no maneuvers
        test_data_no_maneuvers = {
            "maneuver_history": []
        }
        result = await self.service._analyze_propulsive_capabilities(test_object_id, test_data_no_maneuvers)
        assert result["detected"] is False
        assert result["propulsion_type"] == PropulsionType.UNKNOWN