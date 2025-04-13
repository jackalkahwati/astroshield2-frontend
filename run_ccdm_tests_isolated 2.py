"""
Script to run CCDM tests in an isolated environment with mocked dependencies.
This approach allows testing the service logic without requiring all actual dependencies.
"""
import sys
import os
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timedelta
import asyncio
import json
import pytest

# Create mock classes to replace actual dependencies
class MockCCDMModels:
    """Mock CCDM models."""
    class ObjectAnalysisResponse:
        def __init__(self, object_id, timestamp, analysis_complete, confidence_score, 
                    shape_change, thermal_signature, propulsive_capability, error=None):
            self.object_id = object_id
            self.timestamp = timestamp
            self.analysis_complete = analysis_complete
            self.confidence_score = confidence_score
            self.shape_change = shape_change
            self.thermal_signature = thermal_signature
            self.propulsive_capability = propulsive_capability
            self.error = error
    
    class ShapeChangeResponse:
        def __init__(self, detected, confidence, timestamp, metrics=None, error=None):
            self.detected = detected
            self.confidence = confidence
            self.timestamp = timestamp
            self.metrics = metrics
            self.error = error
    
    class ThermalSignatureResponse:
        def __init__(self, detected, confidence, timestamp, metrics=None, error=None):
            self.detected = detected
            self.confidence = confidence
            self.timestamp = timestamp
            self.metrics = metrics
            self.error = error
    
    class PropulsiveCapabilityResponse:
        def __init__(self, detected, confidence, timestamp, metrics=None, error=None):
            self.detected = detected
            self.confidence = confidence
            self.timestamp = timestamp
            self.metrics = metrics
            self.error = error
    
    class ShapeChangeMetrics:
        def __init__(self, volume_change, surface_area_change, aspect_ratio_change, confidence):
            self.volume_change = volume_change
            self.surface_area_change = surface_area_change
            self.aspect_ratio_change = aspect_ratio_change
            self.confidence = confidence
    
    class ThermalSignatureMetrics:
        def __init__(self, temperature_kelvin, anomaly_score):
            self.temperature_kelvin = temperature_kelvin
            self.anomaly_score = anomaly_score
    
    class PropulsionType:
        CHEMICAL = "chemical"
        ELECTRIC = "electric"
        NUCLEAR = "nuclear"
        UNKNOWN = "unknown"
    
    class PropulsionMetrics:
        def __init__(self, type, thrust_estimate=None, fuel_reserve_estimate=None):
            self.type = type
            self.thrust_estimate = thrust_estimate
            self.fuel_reserve_estimate = fuel_reserve_estimate
    
    class HistoricalAnalysis:
        def __init__(self, object_id, time_range, patterns, trend_analysis, anomalies):
            self.object_id = object_id
            self.time_range = time_range
            self.patterns = patterns
            self.trend_analysis = trend_analysis
            self.anomalies = anomalies
    
    class CCDMAssessment:
        def __init__(self, object_id, assessment_type, timestamp, results, confidence_level, 
                     triggered_indicators, recommendations, summary=None):
            self.object_id = object_id
            self.assessment_type = assessment_type
            self.timestamp = timestamp
            self.results = results
            self.confidence_level = confidence_level
            self.triggered_indicators = triggered_indicators
            self.recommendations = recommendations
            self.summary = summary
    
    class AnomalyDetection:
        def __init__(self, object_id, timestamp, anomaly_type, details, confidence, recommended_actions):
            self.object_id = object_id
            self.timestamp = timestamp
            self.anomaly_type = anomaly_type
            self.details = details
            self.confidence = confidence
            self.recommended_actions = recommended_actions

# Create mock classes for analyzers
class MockAnalyzers:
    """Mock analyzer classes."""
    class StabilityIndicator:
        def analyze_stability(self, data):
            return {"stability_changed": False, "confidence": 0.8}
    
    class ManeuverIndicator:
        def analyze_maneuvers(self, data, baseline=None):
            return {"maneuvers_detected": True, "pol_violation": False, "confidence": 0.85}
    
    class RFIndicator:
        def analyze_rf_pattern(self, data, baseline=None):
            return {"rf_detected": True, "pol_violation": False, "confidence": 0.7}
    
    class SubSatelliteAnalyzer:
        def detect_sub_satellites(self, object_id, associated_objects):
            return {"subsatellites_detected": False, "confidence": 0.8}
    
    class ITUComplianceChecker:
        def check_itu_compliance(self, object_id, rf_data, filing_data):
            return {"violates_filing": False, "confidence": 0.95}
    
    class AnalystDisagreementChecker:
        def check_disagreements(self, object_id, analysis_history):
            return {"class_disagreement": False, "confidence": 0.9}
    
    class OrbitAnalyzer:
        def analyze_orbit(self, orbit_data, parent_orbit_data, population_data, radiation_data):
            return {"orbit_out_of_family": False, "confidence": 0.85}
    
    class SignatureAnalyzer:
        def analyze_signatures(self, optical_sig, radar_sig, baseline_sigs):
            return {"optical_out_of_family": False, "confidence": 0.8}
    
    class StimulationAnalyzer:
        def analyze_stimulation(self, events, system_locations):
            return {"stimulation_detected": False, "confidence": 0.7}
    
    class AMRAnalyzer:
        def analyze_amr(self, amr_history, population_data):
            return {"amr_out_of_family": False, "confidence": 0.85}
    
    class ImagingManeuverAnalyzer:
        def analyze_imaging_maneuvers(self, maneuvers, coverage_gaps, proximity_events):
            return {"imaging_maneuver_detected": False, "confidence": 0.75}
    
    class LaunchAnalyzer:
        def analyze_launch(self, launch_site, expected_objects, tracked_objects, known_threats):
            return {"suspicious_source": False, "confidence": 0.9}
    
    class EclipseAnalyzer:
        def analyze_eclipse_behavior(self, object_id, events, eclipse_times):
            return {"uct_during_eclipse": False, "confidence": 0.8}
    
    class RegistryChecker:
        def check_registry(self, object_id, registry_data):
            return {"registered": True, "confidence": 0.95}

# Create a mock UDL client
class MockUDLClient:
    """Mock UDL client."""
    async def get_state_vector(self, object_id):
        return {
            "position": {"x": 1000.0, "y": 2000.0, "z": 3000.0},
            "velocity": {"x": 1.0, "y": 2.0, "z": 3.0},
            "epoch": datetime.utcnow().isoformat()
        }
    
    async def get_state_vector_history(self, object_id, start_time, end_time):
        return [
            {
                "time": (datetime.utcnow() - timedelta(hours=i)).isoformat(),
                "position": {"x": 1000.0 - i*100, "y": 2000.0 - i*100, "z": 3000.0 - i*100},
                "velocity": {"x": 1.0 - i*0.1, "y": 2.0 - i*0.1, "z": 3.0 - i*0.1}
            } for i in range(1, 24)
        ]
    
    async def get_elset_data(self, object_id):
        return {
            "semi_major_axis": 7000.0,
            "eccentricity": 0.001,
            "inclination": 51.6,
            "raan": 120.0,
            "arg_perigee": 180.0,
            "mean_anomaly": 0.0,
            "mean_motion": 15.5
        }
    
    async def get_maneuver_data(self, object_id):
        return [
            {
                "time": (datetime.utcnow() - timedelta(days=i*7)).isoformat(),
                "delta_v": 0.5 + (i*0.1),
                "thrust_vector": {"x": 0.1, "y": 0.2, "z": 0.3},
                "duration": 60.0 + (i*10),
                "confidence": 0.85
            } for i in range(0, 3)
        ]
    
    async def get_conjunction_data(self, object_id):
        return {
            "events": [
                {
                    "timestamp": (datetime.utcnow() - timedelta(days=i)).isoformat(),
                    "miss_distance": 40.0 + (i*5),
                    "probability": 1e-6 * (i+1)
                } for i in range(0, 5)
            ]
        }
    
    async def get_rf_interference(self, freq_range):
        return {
            "measurements": [
                {
                    "timestamp": (datetime.utcnow() - timedelta(hours=i*6)).isoformat(),
                    "frequency": 2200.0,
                    "power_level": -90.0 - (i*2),
                    "duration": 300.0,
                    "bandwidth": 5.0
                } for i in range(0, 4)
            ]
        }

# Create a CCDM service definition with all dependencies mocked
class MockCCDMService:
    """CCDM service implementation with dependencies mocked."""
    def __init__(self, config=None):
        """Initialize the CCDM service with configuration and analyzers."""
        self._config = config or self._get_default_config()
        
        # Initialize data clients
        self._initialize_data_clients()
        
        # Initialize analyzer classes
        self._initialize_analyzers()
        
        # Initialize ML models for advanced detection
        self._initialize_ml_models()
        
        # Initialize Kafka message handling for real-time data
        self._initialize_kafka_messaging()
        
        # Cache for object data to reduce redundant fetches
        self._object_data_cache = {}
        self._cache_ttl = 300  # seconds
        self._cache_timestamps = {}
    
    def _get_default_config(self):
        """Get default configuration for CCDM service."""
        return {
            "data_sources": {
                "udl": {
                    "base_url": "https://udl.sda.mil/api",
                    "api_key_env": "UDL_API_KEY",
                    "timeout": 30
                },
                "space_track": {
                    "base_url": "https://www.space-track.org/basicspacedata/query",
                    "timeout": 30
                },
                "tmdb": {
                    "base_url": "https://tmdb.sda.mil/api",
                    "timeout": 30
                }
            },
            "kafka": {
                "bootstrap_servers": "kafka.sda.mil:9092",
                "group_id": "astroshield-ccdm-service",
                "topics": {
                    "subscribe": [
                        "SS2.data.state-vector",
                        "SS5.conjunction.events", 
                        "SS5.cyber.threats",
                        "SS5.launch.prediction", 
                        "SS5.telemetry.data"
                    ],
                    "publish": [
                        "SS4.ccdm.detection",
                        "SS4.ccdm.assessment"
                    ]
                },
                "heartbeat_interval_ms": 60000
            },
            "ccdm": {
                "indicators": {
                    "stability": {"threshold": 0.85, "weight": 1.0},
                    "maneuvers": {"threshold": 0.8, "weight": 1.2},
                    "rf": {"threshold": 0.85, "weight": 1.0},
                    "subsatellites": {"threshold": 0.9, "weight": 1.5},
                    "itu_compliance": {"threshold": 0.95, "weight": 0.8},
                    "analyst_disagreement": {"threshold": 0.8, "weight": 0.7},
                    "orbit": {"threshold": 0.9, "weight": 1.1},
                    "signature": {"threshold": 0.85, "weight": 1.2},
                    "stimulation": {"threshold": 0.9, "weight": 1.3},
                    "amr": {"threshold": 0.85, "weight": 1.0},
                    "imaging": {"threshold": 0.85, "weight": 1.2},
                    "launch": {"threshold": 0.9, "weight": 1.1},
                    "eclipse": {"threshold": 0.9, "weight": 1.4},
                    "registry": {"threshold": 0.95, "weight": 0.9},
                },
                "sensor_types": ["RADAR", "EO", "RF", "IR", "SPACE"],
                "ml_filter_threshold": 0.7,
                "assessment_interval_seconds": 300,
                "anomaly_detection": {
                    "lookback_days": 30,
                    "anomaly_threshold": 0.85
                }
            },
            "security": {
                "data_classification": "UNCLASSIFIED",
                "access_control": {
                    "required_roles": ["ccdm_analyst", "active"]
                }
            }
        }
    
    def _initialize_data_clients(self):
        """Initialize data source clients."""
        # UDL client for accessing the Unified Data Library
        self.udl_client = MockUDLClient()
        
        # UDL integration layer for enhanced data operations
        self.udl_integrator = MagicMock()
        
        # Space Track client would be initialized here
        self.space_track_client = MagicMock()
        
        # TMDB client would be initialized here
        self.tmdb_client = MagicMock()
    
    def _initialize_analyzers(self):
        """Initialize all CCDM analyzer components."""
        # Core CCDM Indicators
        analyzers = MockAnalyzers()
        self.stability_analyzer = analyzers.StabilityIndicator()
        self.maneuver_analyzer = analyzers.ManeuverIndicator()
        self.rf_analyzer = analyzers.RFIndicator()
        self.subsatellite_analyzer = analyzers.SubSatelliteAnalyzer()
        self.itu_checker = analyzers.ITUComplianceChecker()
        self.disagreement_checker = analyzers.AnalystDisagreementChecker()
        self.orbit_analyzer = analyzers.OrbitAnalyzer()
        self.signature_analyzer = analyzers.SignatureAnalyzer()
        self.stimulation_analyzer = analyzers.StimulationAnalyzer()
        self.amr_analyzer = analyzers.AMRAnalyzer()
        self.imaging_analyzer = analyzers.ImagingManeuverAnalyzer()
        self.launch_analyzer = analyzers.LaunchAnalyzer()
        self.eclipse_analyzer = analyzers.EclipseAnalyzer()
        self.registry_checker = analyzers.RegistryChecker()
    
    def _initialize_ml_models(self):
        """Initialize ML models for enhanced CCDM detection."""
        # Anomaly detection model for filtering false alarms
        self.anomaly_detector = MagicMock()
        
        # Track evaluation for trajectory analysis
        self.track_evaluator = MagicMock()
    
    def _initialize_kafka_messaging(self):
        """Initialize Kafka messaging for real-time data streaming."""
        # Initialize consumer for relevant topics
        self.kafka_consumer = MagicMock()
    
    def format_maneuver_data(self, raw_maneuvers):
        """Format raw maneuver data for CCDM analysis."""
        formatted = []
        for maneuver in raw_maneuvers:
            formatted.append({
                "time": maneuver.get("time") or maneuver.get("epoch") or maneuver.get("timestamp"),
                "delta_v": maneuver.get("delta_v", 0.0),
                "thrust_vector": maneuver.get("thrust_vector", {"x": 0, "y": 0, "z": 0}),
                "duration": maneuver.get("duration", 0.0),
                "confidence": maneuver.get("confidence", 0.8)
            })
        return formatted
    
    def format_rf_data(self, raw_rf):
        """Format raw RF data for CCDM analysis."""
        formatted = []
        for measurement in raw_rf.get("measurements", []):
            formatted.append({
                "time": measurement.get("time") or measurement.get("timestamp"),
                "frequency": measurement.get("frequency", 0.0),
                "power": measurement.get("power_level", -120.0),
                "duration": measurement.get("duration", 0.0),
                "bandwidth": measurement.get("bandwidth", 0.0),
                "confidence": measurement.get("confidence", 0.7)
            })
        return formatted
    
    def extract_orbit_data(self, elset_data, state_vector):
        """Extract and combine orbit data from ELSET and state vector."""
        # Start with data from ELSET if available
        orbit_data = {
            "semi_major_axis": elset_data.get("semi_major_axis", 0.0),
            "eccentricity": elset_data.get("eccentricity", 0.0),
            "inclination": elset_data.get("inclination", 0.0),
            "raan": elset_data.get("raan", 0.0),
            "arg_perigee": elset_data.get("arg_perigee", 0.0),
            "mean_anomaly": elset_data.get("mean_anomaly", 0.0),
            "mean_motion": elset_data.get("mean_motion", 0.0)
        }
        
        # Add data from state vector if available
        if state_vector:
            position = state_vector.get("position", {})
            velocity = state_vector.get("velocity", {})
            
            if position and velocity:
                # Calculate orbital elements from state vector (if needed)
                orbit_data.update({
                    "position_vector": position,
                    "velocity_vector": velocity,
                    "epoch": state_vector.get("epoch")
                })
        
        return orbit_data
    
    async def safe_fetch(self, fetch_func):
        """Safely execute a fetch function and handle exceptions."""
        try:
            return await fetch_func() if asyncio.iscoroutinefunction(fetch_func) else fetch_func()
        except Exception as e:
            return e
    
    async def analyze_shape_changes(self, object_id, object_data):
        """Analyze shape changes in an object."""
        # Get baseline signatures
        baseline = object_data.get("baseline_signatures", {})
        baseline_radar = baseline.get("radar", {})
        baseline_optical = baseline.get("optical", {})
        
        # Get current signatures
        current_radar = object_data.get("radar_signature", {})
        current_optical = object_data.get("optical_signature", {})
        
        # Check for shape changes
        radar_change = False
        optical_change = False
        
        # Radar-based detection (RCS change)
        if current_radar and baseline_radar:
            current_rcs = current_radar.get("rcs", 0.0)
            mean_rcs = baseline_radar.get("rcs_mean", 1.0)
            std_rcs = baseline_radar.get("rcs_std", 0.3)
            
            # Check if current RCS is more than 2 standard deviations from the mean
            if abs(current_rcs - mean_rcs) > 2 * std_rcs:
                radar_change = True
        
        # Optical-based detection (magnitude change)
        if current_optical and baseline_optical:
            current_mag = current_optical.get("magnitude", 10.0)
            mean_mag = baseline_optical.get("magnitude_mean", 10.0)
            std_mag = baseline_optical.get("magnitude_std", 0.5)
            
            # Check if current magnitude is more than 2 standard deviations from the mean
            if abs(current_mag - mean_mag) > 2 * std_mag:
                optical_change = True
        
        # Combine detections - true if either sensor detects a change
        detected = radar_change or optical_change
        
        # Calculate confidence based on the quality of the data
        radar_confidence = current_radar.get("confidence", 0.0) if current_radar else 0.0
        optical_confidence = current_optical.get("confidence", 0.0) if current_optical else 0.0
        
        # Weight radar higher for shape analysis (more reliable for this purpose)
        if radar_confidence and optical_confidence:
            confidence = (0.7 * radar_confidence + 0.3 * optical_confidence)
        else:
            confidence = radar_confidence or optical_confidence or 0.5
        
        return {
            "detected": detected,
            "confidence": confidence,
            "radar_change": radar_change,
            "optical_change": optical_change
        }

# Now define our tests
class TestCCDMServiceIsolated:
    """Test CCDM service functionality in an isolated environment."""
    
    def test_format_maneuver_data(self):
        """Test the maneuver data formatting logic."""
        # Sample input data
        raw_maneuvers = [
            {"time": "2023-01-01T00:00:00Z", "delta_v": 0.5},
            {"epoch": "2023-01-02T00:00:00Z", "delta_v": 0.6, "thrust_vector": {"x": 0.1, "y": 0.2, "z": 0.3}},
            {"timestamp": "2023-01-03T00:00:00Z", "delta_v": 0.7}
        ]
        
        # Create service instance
        service = MockCCDMService()
        
        # Call the function
        result = service.format_maneuver_data(raw_maneuvers)
        
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
        
        # Check delta-v values
        assert result[0]["delta_v"] == 0.5
        assert result[1]["delta_v"] == 0.6
        assert result[2]["delta_v"] == 0.7
        
        # Check that thrust vector is properly handled
        assert result[1]["thrust_vector"] == {"x": 0.1, "y": 0.2, "z": 0.3}
        assert result[0]["thrust_vector"] == {"x": 0, "y": 0, "z": 0}  # Default
    
    def test_format_rf_data(self):
        """Test the RF data formatting logic."""
        # Sample input data
        raw_rf = {
            "measurements": [
                {"time": "2023-01-01T00:00:00Z", "frequency": 2200.0, "power_level": -90.0},
                {"timestamp": "2023-01-02T00:00:00Z", "frequency": 8400.0, "power_level": -95.0, "bandwidth": 10.0},
                {"time": "2023-01-03T00:00:00Z", "frequency": 4500.0, "power_level": -85.0, "duration": 60.0}
            ]
        }
        
        # Create service instance
        service = MockCCDMService()
        
        # Call the function
        result = service.format_rf_data(raw_rf)
        
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
        
        # Check that bandwidth is properly handled
        assert result[1]["bandwidth"] == 10.0
        assert result[0]["bandwidth"] == 0.0  # Default
        
    def test_extract_orbit_data(self):
        """Test extracting orbit data from ELSET and state vector."""
        # Sample input data
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
        
        # Create service instance
        service = MockCCDMService()
        
        # Call the function
        result = service.extract_orbit_data(elset_data, state_vector)
        
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
    
    @pytest.mark.asyncio
    async def test_safe_fetch(self):
        """Test safe fetch method for handling exceptions."""
        # Create service instance
        service = MockCCDMService()
        
        # Test successful synchronous function
        def success_func():
            return {"success": True}
        
        result = await service.safe_fetch(success_func)
        assert result == {"success": True}
        
        # Test successful async function
        async def async_success_func():
            return {"async_success": True}
        
        result = await service.safe_fetch(async_success_func)
        assert result == {"async_success": True}
        
        # Test function that raises exception
        def error_func():
            raise ValueError("Test error")
        
        result = await service.safe_fetch(error_func)
        assert isinstance(result, ValueError)
        assert str(result) == "Test error"
        
        # Test async function that raises exception
        async def async_error_func():
            raise ValueError("Async test error")
        
        result = await service.safe_fetch(async_error_func)
        assert isinstance(result, ValueError)
        assert str(result) == "Async test error"
    
    @pytest.mark.asyncio
    async def test_analyze_shape_changes(self):
        """Test analyzing shape changes."""
        # Sample input data with significant shape changes
        object_id = "test-sat-001"
        test_data = {
            "radar_signature": {
                "rcs": 2.5,  # Significantly different from baseline
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
            }
        }
        
        # Create service instance
        service = MockCCDMService()
        
        # Call the method
        result = await service.analyze_shape_changes(object_id, test_data)
        
        # Verify the result
        assert "detected" in result
        assert "confidence" in result
        assert "radar_change" in result
        assert "optical_change" in result
        assert result["detected"] is True
        assert result["radar_change"] is True  # RCS 2.5 is more than 2 std devs from mean 1.2, std 0.3
        assert result["optical_change"] is False  # Magnitude 6.5 is not far enough from mean 7.0, std 0.5
        assert result["confidence"] > 0.7
    
    @pytest.mark.asyncio
    async def test_object_analysis_response(self):
        """Test creating a complete object analysis response."""
        # Create sample input data
        object_id = "test-sat-001"
        
        # Create mock shape change data
        shape_change_data = {
            "detected": True,
            "confidence": 0.85,
            "radar_change": True,
            "optical_change": False
        }
        
        # Create mock thermal signature data
        thermal_data = {
            "detected": True,
            "confidence": 0.8,
            "temperature_kelvin": 285.0,
            "anomaly_score": 0.75
        }
        
        # Create mock propulsive capabilities data
        propulsive_data = {
            "detected": True,
            "confidence": 0.9,
            "propulsion_type": MockCCDMModels.PropulsionType.CHEMICAL,
            "thrust_estimate": 10.5,
            "fuel_reserve_estimate": 65.0
        }
        
        # Function to create the response
        def create_object_analysis_response(object_id, shape_data, thermal_data, propulsive_data):
            """Create an object analysis response from component results."""
            # Calculate overall confidence
            confidences = [
                shape_data.get("confidence", 0.0),
                thermal_data.get("confidence", 0.0),
                propulsive_data.get("confidence", 0.0)
            ]
            
            # Filter out zero confidences
            valid_confidences = [c for c in confidences if c > 0]
            overall_confidence = sum(valid_confidences) / len(valid_confidences) if valid_confidences else 0.5
            
            # Create response object
            models = MockCCDMModels()
            
            # Create component responses
            shape_change = models.ShapeChangeResponse(
                detected=shape_data.get("detected", False),
                confidence=shape_data.get("confidence", 0.0),
                timestamp=datetime.utcnow()
            )
            
            thermal_signature = models.ThermalSignatureResponse(
                detected=thermal_data.get("detected", False),
                confidence=thermal_data.get("confidence", 0.0),
                timestamp=datetime.utcnow()
            )
            
            # Create metrics if propulsion is detected
            if propulsive_data.get("detected", False):
                prop_metrics = models.PropulsionMetrics(
                    type=propulsive_data.get("propulsion_type", models.PropulsionType.UNKNOWN),
                    thrust_estimate=propulsive_data.get("thrust_estimate"),
                    fuel_reserve_estimate=propulsive_data.get("fuel_reserve_estimate")
                )
            else:
                prop_metrics = None
            
            propulsive_capability = models.PropulsiveCapabilityResponse(
                detected=propulsive_data.get("detected", False),
                confidence=propulsive_data.get("confidence", 0.0),
                timestamp=datetime.utcnow(),
                metrics=prop_metrics
            )
            
            # Create the full response
            return models.ObjectAnalysisResponse(
                object_id=object_id,
                timestamp=datetime.utcnow(),
                analysis_complete=True,
                confidence_score=overall_confidence,
                shape_change=shape_change,
                thermal_signature=thermal_signature,
                propulsive_capability=propulsive_capability
            )
        
        # Call the function
        response = create_object_analysis_response(
            object_id, 
            shape_change_data, 
            thermal_data, 
            propulsive_data
        )
        
        # Verify the response
        assert response.object_id == object_id
        assert response.analysis_complete is True
        assert 0.8 < response.confidence_score < 0.9  # Should be around 0.85
        
        # Verify shape change component
        assert response.shape_change.detected is True
        assert response.shape_change.confidence == 0.85
        
        # Verify thermal signature component
        assert response.thermal_signature.detected is True
        assert response.thermal_signature.confidence == 0.8
        
        # Verify propulsive capability component
        assert response.propulsive_capability.detected is True
        assert response.propulsive_capability.confidence == 0.9
        assert response.propulsive_capability.metrics.type == MockCCDMModels.PropulsionType.CHEMICAL
        assert response.propulsive_capability.metrics.thrust_estimate == 10.5
        assert response.propulsive_capability.metrics.fuel_reserve_estimate == 65.0

if __name__ == "__main__":
    # Run the tests directly
    pytest.main(["-xvs", __file__])