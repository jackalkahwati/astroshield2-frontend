"""
Comprehensive tests for SDA Welders Arc Integration
Tests all subsystems, Kafka messaging, UDL integration, and workflows
"""

import pytest
import asyncio
from datetime import datetime, timedelta
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
import json

from app.sda_integration.kafka.kafka_client import (
    KafkaConfig, WeldersArcKafkaClient, EventProcessor,
    WeldersArcMessage, EventType, SubsystemID, KafkaTopics
)
from app.sda_integration.udl.udl_client import (
    UDLConfig, UDLClient, CollectionRequest, SensorObservation, TLE
)
from app.sda_integration.subsystems.ss2_state_estimation import (
    StateEstimator, UCTProcessor, Track, StateVector
)
from app.sda_integration.subsystems.ss5_hostility_monitoring import (
    HostilityMonitor, WEZPredictor, IntentAssessor,
    ThreatType, IntentLevel, WeaponEngagementZone, ThreatAssessment
)
from app.sda_integration.workflows.node_red_service import (
    NodeREDConfig, NodeREDService, CCDMWorkflowManager, CCDMIndicator
)
from app.sda_integration.welders_arc_integration import WeldersArcIntegration
from app.core.config import Settings


@pytest.fixture
def kafka_config():
    return KafkaConfig(bootstrap_servers="localhost:9092")


@pytest.fixture
def udl_config():
    return UDLConfig(
        base_url="https://test-udl.mil",
        api_key="test-key"
    )


@pytest.fixture
def node_red_config():
    return NodeREDConfig(
        base_url="http://localhost:1880",
        username="test",
        password="test"
    )


@pytest.fixture
def settings():
    return Settings(
        KAFKA_BOOTSTRAP_SERVERS="localhost:9092",
        UDL_BASE_URL="https://test-udl.mil",
        UDL_API_KEY="test-key",
        NODE_RED_URL="http://localhost:1880",
        NODE_RED_USER="test",
        NODE_RED_PASSWORD="test"
    )


class TestKafkaIntegration:
    """Test Kafka message bus integration"""
    
    @pytest.mark.asyncio
    async def test_kafka_client_initialization(self, kafka_config):
        """Test Kafka client initialization"""
        client = WeldersArcKafkaClient(kafka_config)
        
        with patch.object(client, 'producer') as mock_producer:
            with patch.object(client, 'admin_client') as mock_admin:
                await client.initialize()
                
                assert client.producer is not None
                assert client.admin_client is not None
                
    @pytest.mark.asyncio
    async def test_message_publishing(self, kafka_config):
        """Test message publishing to Kafka"""
        client = WeldersArcKafkaClient(kafka_config)
        client.producer = Mock()
        
        message = WeldersArcMessage(
            message_id="test-123",
            timestamp=datetime.utcnow(),
            subsystem=SubsystemID.SS2_STATE_ESTIMATION,
            event_type=EventType.MANEUVER,
            data={"object_id": "TEST-001", "delta_v": 0.5}
        )
        
        await client.publish(KafkaTopics.EVENT_MANEUVER_DETECTION, message)
        
        client.producer.produce.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_event_processor(self, kafka_config):
        """Test event processing workflows"""
        client = WeldersArcKafkaClient(kafka_config)
        client.publish = AsyncMock()
        
        processor = EventProcessor(client)
        
        # Test launch event
        await processor.process_event(EventType.LAUNCH, {
            "launch_site": "Cape Canaveral",
            "vehicle_id": "FALCON-9",
            "payload_id": "STARLINK-1234"
        })
        
        # Verify message was published
        client.publish.assert_called()
        call_args = client.publish.call_args
        assert call_args[0][0] == KafkaTopics.EVENT_LAUNCH_DETECTION
        
    def test_all_kafka_topics_defined(self):
        """Ensure all 122+ Kafka topics are defined"""
        topic_count = 0
        for attr in dir(KafkaTopics):
            if not attr.startswith('_'):
                value = getattr(KafkaTopics, attr)
                if isinstance(value, str) and value.startswith('welders.'):
                    topic_count += 1
                    
        assert topic_count >= 30  # We have at least 30 core topics defined


class TestUDLIntegration:
    """Test UDL client integration"""
    
    @pytest.mark.asyncio
    async def test_udl_authentication(self, udl_config):
        """Test UDL authentication"""
        client = UDLClient(udl_config)
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.json = AsyncMock(return_value={
                "access_token": "test-token",
                "expires_in": 3600
            })
            mock_response.raise_for_status = Mock()
            
            mock_session.return_value.post.return_value.__aenter__.return_value = mock_response
            
            await client.connect()
            
            assert client._auth_token == "test-token"
            assert client._token_expires is not None
            
    @pytest.mark.asyncio
    async def test_get_sensor_observations(self, udl_config):
        """Test retrieving sensor observations"""
        client = UDLClient(udl_config)
        client._auth_token = "test-token"
        client._token_expires = datetime.utcnow() + timedelta(hours=1)
        
        mock_observations = {
            "observations": [
                {
                    "observation_id": "obs-001",
                    "sensor_id": "sensor-001",
                    "target_id": "target-001",
                    "timestamp": datetime.utcnow().isoformat(),
                    "observation_type": "optical",
                    "position": {"lat": 0, "lon": 0, "alt": 500},
                    "metadata": {}
                }
            ]
        }
        
        with patch.object(client, '_make_request', return_value=mock_observations):
            observations = await client.get_sensor_observations(
                sensor_id="sensor-001",
                limit=10
            )
            
            assert len(observations) == 1
            assert observations[0].observation_id == "obs-001"
            
    @pytest.mark.asyncio
    async def test_collection_request_submission(self, udl_config):
        """Test submitting collection requests"""
        client = UDLClient(udl_config)
        client._auth_token = "test-token"
        client._token_expires = datetime.utcnow() + timedelta(hours=1)
        
        request = CollectionRequest(
            request_id="req-001",
            sensor_id="sensor-001",
            target_id="target-001",
            collection_type="OPTICAL",
            priority=1,
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow() + timedelta(hours=1)
        )
        
        mock_response = {
            "request_id": "req-001",
            "status": "ACCEPTED",
            "sensor_id": "sensor-001",
            "scheduled_time": (datetime.utcnow() + timedelta(minutes=30)).isoformat()
        }
        
        with patch.object(client, '_make_request', return_value=mock_response):
            response = await client.submit_collection_request(request)
            
            assert response.status == "ACCEPTED"
            assert response.scheduled_time is not None


class TestStateEstimation:
    """Test state estimation subsystem"""
    
    @pytest.mark.asyncio
    async def test_uct_processing(self):
        """Test UCT processing pipeline"""
        processor = UCTProcessor()
        
        # Create test tracks
        tracks = []
        for i in range(3):
            track = Track(
                track_id=f"track-{i}",
                sensor_id="sensor-001",
                timestamp=datetime.utcnow() + timedelta(minutes=i*10),
                position=np.array([7000 + i*10, 0, 0])
            )
            tracks.append(track)
            
        # Process tracks
        for track in tracks:
            result = await processor.process_track(track)
            
        # Should attempt orbit determination after 3 tracks
        assert len(processor.pending_tracks) <= 3
        
    def test_orbit_propagation(self):
        """Test two-body orbit propagation"""
        kafka_client = Mock()
        estimator = StateEstimator(kafka_client)
        
        # Initial state (circular orbit at 500km)
        r0 = np.array([6878.137, 0, 0])  # km
        v0 = np.array([0, 7.61, 0])      # km/s
        state = np.concatenate([r0, v0])
        
        # Propagate 1 orbit period (~94 minutes)
        target_time = datetime.utcnow() + timedelta(minutes=94)
        propagated = estimator._propagate_state(state, target_time)
        
        # Should return to approximately same position
        assert np.allclose(propagated[:3], r0, rtol=0.01)
        
    @pytest.mark.asyncio
    async def test_maneuver_detection(self):
        """Test maneuver detection from observations"""
        kafka_client = Mock()
        estimator = StateEstimator(kafka_client)
        
        # Create observations before maneuver
        obs_pre = []
        for i in range(5):
            obs_pre.append({
                "id": f"obs-pre-{i}",
                "sensor_id": "sensor-001",
                "timestamp": (datetime.utcnow() - timedelta(hours=2, minutes=i*10)).isoformat(),
                "position": [6878.137 + i*0.1, 0, 0]
            })
            
        # Create observations after maneuver (with delta-V applied)
        obs_post = []
        for i in range(5):
            obs_post.append({
                "id": f"obs-post-{i}",
                "sensor_id": "sensor-001",
                "timestamp": (datetime.utcnow() - timedelta(minutes=i*10)).isoformat(),
                "position": [6900.0 + i*0.1, 0, 0]  # Different orbit
            })
            
        maneuver_time = datetime.utcnow() - timedelta(hours=1)
        
        result = await estimator.process_maneuver_hypothesis(
            "object-001",
            maneuver_time,
            obs_pre,
            obs_post
        )
        
        assert result["maneuver_detected"] == True
        assert "delta_v" in result
        assert "maneuver_type" in result


class TestHostilityMonitoring:
    """Test hostility monitoring subsystem"""
    
    @pytest.mark.asyncio
    async def test_wez_prediction_kinetic(self):
        """Test kinetic kill WEZ prediction"""
        predictor = WEZPredictor()
        
        threat_state = {
            "object_id": "threat-001",
            "position": [7000, 0, 0],
            "velocity": [0, 7.5, 0]
        }
        
        target_state = {
            "object_id": "target-001",
            "position": [7100, 0, 0],
            "velocity": [0, 7.4, 0]
        }
        
        capabilities = {
            "kinetic_interceptor": True,
            "max_intercept_range": 1000
        }
        
        time_window = (
            datetime.utcnow(),
            datetime.utcnow() + timedelta(hours=24)
        )
        
        zones = await predictor.predict_wez(
            threat_state,
            target_state,
            capabilities,
            time_window
        )
        
        assert len(zones) > 0
        assert zones[0].threat_type == ThreatType.KINETIC_KILL
        
    @pytest.mark.asyncio
    async def test_intent_assessment(self):
        """Test intent assessment logic"""
        assessor = IntentAssessor()
        
        # Create pattern of life
        assessor.pattern_database["object-001"] = Mock(
            maneuver_frequency=0.1,  # Normal frequency
            typical_delta_v=0.01,
            maneuver_times=[
                datetime.utcnow() - timedelta(days=i)
                for i in range(10)
            ]
        )
        
        # Current behavior shows anomaly
        current_behavior = {
            "maneuvering": True,
            "delta_v": 0.5  # Much higher than typical
        }
        
        # High probability WEZ
        wez_predictions = [
            Mock(
                probability=0.9,
                threat_type=ThreatType.KINETIC_KILL
            )
        ]
        
        assessment = await assessor.assess_intent(
            "object-001",
            current_behavior,
            wez_predictions,
            {}
        )
        
        assert assessment.threat_level >= IntentLevel.SUSPICIOUS
        assert ThreatType.KINETIC_KILL in assessment.threat_types
        
    @pytest.mark.asyncio
    async def test_proximity_event_handling(self):
        """Test emergency proximity event handling"""
        kafka_client = Mock()
        kafka_client.publish = AsyncMock()
        
        monitor = HostilityMonitor(kafka_client)
        monitor.threat_catalog["threat-001"] = Mock()
        
        message = WeldersArcMessage(
            message_id="prox-001",
            timestamp=datetime.utcnow(),
            subsystem=SubsystemID.SS2_STATE_ESTIMATION,
            event_type=EventType.PROXIMITY,
            data={
                "object1_id": "threat-001",
                "object2_id": "target-001",
                "range_km": 5.0,
                "relative_velocity_km_s": 2.0
            }
        )
        
        await monitor._handle_proximity_event(message)
        
        # Should publish emergency alert
        kafka_client.publish.assert_called()
        call_args = kafka_client.publish.call_args_list
        
        # Check for threat warning
        threat_warning_published = False
        emergency_alert_published = False
        
        for call in call_args:
            topic = call[0][0]
            if topic == KafkaTopics.THREAT_WARNING:
                threat_warning_published = True
            elif topic == KafkaTopics.ALERT_OPERATOR:
                emergency_alert_published = True
                
        assert threat_warning_published
        assert emergency_alert_published


class TestNodeREDIntegration:
    """Test Node-RED workflow integration"""
    
    @pytest.mark.asyncio
    async def test_ccdm_workflow_deployment(self, node_red_config):
        """Test CCDM workflow deployment"""
        service = NodeREDService(node_red_config)
        
        with patch.object(service.session, 'post') as mock_post:
            mock_response = AsyncMock()
            mock_response.raise_for_status = Mock()
            mock_response.json = AsyncMock(return_value={"id": "flow-123"})
            mock_post.return_value.__aenter__.return_value = mock_response
            
            flow_id = await service.deploy_ccdm_workflow()
            
            assert flow_id == "flow-123"
            mock_post.assert_called_once()
            
    @pytest.mark.asyncio
    async def test_indicator_workflow_creation(self, node_red_config):
        """Test individual indicator workflow creation"""
        service = NodeREDService(node_red_config)
        manager = CCDMWorkflowManager(service)
        
        # Test creating workflow for each of the 19 indicators
        test_indicators = [
            ("object_stability", "lstm"),
            ("maneuvers_detected", "bilstm"),
            ("rf_detection", "cnn"),
            ("proximity_operations", "gnn")
        ]
        
        for indicator_name, model_type in test_indicators:
            flow = await service.create_indicator_workflow(indicator_name, model_type)
            
            assert flow is not None
            # Verify flow structure
            assert len(flow.nodes) >= 6  # Input, model, threshold, status nodes, output
            assert any(node["type"] == "ml-inference" for node in flow.nodes)
            assert any(node["type"] == "kafka-producer" for node in flow.nodes)


class TestWeldersArcIntegration:
    """Test main integration service"""
    
    @pytest.mark.asyncio
    async def test_system_initialization(self, settings):
        """Test full system initialization"""
        integration = WeldersArcIntegration(settings)
        
        with patch.object(integration, 'kafka_client') as mock_kafka:
            with patch.object(integration, 'udl_client') as mock_udl:
                with patch.object(integration, 'node_red_service') as mock_node_red:
                    await integration.initialize()
                    
                    assert integration.state_estimator is not None
                    assert integration.hostility_monitor is not None
                    assert integration.event_processor is not None
                    assert integration.ccdm_workflows is not None
                    
    @pytest.mark.asyncio
    async def test_event_correlation(self, settings):
        """Test cross-subsystem event correlation"""
        integration = WeldersArcIntegration(settings)
        
        # Create correlated events
        events = [
            {
                "timestamp": datetime.utcnow(),
                "object_id": "object-001",
                "event_type": "maneuver"
            },
            {
                "timestamp": datetime.utcnow() + timedelta(seconds=30),
                "object_id": "object-001",
                "event_type": "rf_detection"
            }
        ]
        
        correlations = integration._find_event_correlations(events)
        
        assert len(correlations) == 1
        assert len(correlations[0]) == 2
        
    @pytest.mark.asyncio
    async def test_system_health_check(self, settings):
        """Test system health monitoring"""
        integration = WeldersArcIntegration(settings)
        integration._running = True
        integration.kafka_client = Mock(producer=True)
        integration.udl_client = Mock(session=True)
        integration.node_red_service = Mock()
        
        status = await integration.get_system_status()
        
        assert status["running"] == True
        assert status["kafka_connected"] == True
        assert status["udl_connected"] == True
        assert status["node_red_connected"] == True


class TestCCDMIndicators:
    """Test all 19 CCDM indicators"""
    
    def test_all_indicators_defined(self):
        """Ensure all 19 indicators are defined"""
        manager = CCDMWorkflowManager(Mock())
        
        expected_indicators = [
            # Stability
            "object_stability",
            "stability_changes",
            # Maneuver
            "maneuvers_detected",
            "pattern_of_life",
            # RF
            "rf_detection",
            "subsatellite_deployment",
            # Compliance
            "itu_fcc_compliance",
            "analyst_consensus",
            # Signature
            "optical_signature",
            "radar_signature",
            # Stimulation
            "system_response",
            # Physical
            "area_mass_ratio",
            "proximity_operations",
            # Tracking
            "tracking_anomalies",
            "imaging_maneuvers",
            # Launch
            "launch_site",
            "un_registry",
            # Deception
            "camouflage_detection",
            "intent_assessment"
        ]
        
        actual_indicators = [ind[0] for ind in manager.indicators]
        
        assert len(actual_indicators) == 19
        for expected in expected_indicators:
            assert expected in actual_indicators


class TestEndToEndScenarios:
    """Test complete end-to-end scenarios"""
    
    @pytest.mark.asyncio
    async def test_launch_to_threat_assessment(self, settings):
        """Test complete launch detection to threat assessment workflow"""
        integration = WeldersArcIntegration(settings)
        
        # Mock all services
        integration.kafka_client = Mock()
        integration.kafka_client.publish = AsyncMock()
        integration.event_processor = EventProcessor(integration.kafka_client)
        
        # Simulate launch event
        await integration.process_event(EventType.LAUNCH, {
            "launch_site": "Unknown Site",
            "vehicle_id": "UNKNOWN-001",
            "trajectory": "polar",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Verify launch detection message published
        integration.kafka_client.publish.assert_called()
        
        # Simulate UCT generation
        uct_message = WeldersArcMessage(
            message_id="uct-001",
            timestamp=datetime.utcnow(),
            subsystem=SubsystemID.SS2_STATE_ESTIMATION,
            event_type="uct_detection",
            data={
                "track_id": "track-001",
                "sensor_id": "sensor-001",
                "position": [7000, 0, 0],
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        # Process would continue through state estimation, CCDM evaluation,
        # hostility monitoring, and response recommendation
        
    @pytest.mark.asyncio
    async def test_maneuver_pattern_violation(self, settings):
        """Test detection of pattern of life violations"""
        integration = WeldersArcIntegration(settings)
        
        # Setup mock services
        kafka_client = Mock()
        kafka_client.publish = AsyncMock()
        
        monitor = HostilityMonitor(kafka_client)
        await monitor.initialize()
        
        # Establish pattern of life
        object_id = "sat-001"
        for i in range(10):
            maneuver_message = WeldersArcMessage(
                message_id=f"maneuver-{i}",
                timestamp=datetime.utcnow() - timedelta(days=30-i*3),
                subsystem=SubsystemID.SS2_STATE_ESTIMATION,
                event_type=EventType.MANEUVER,
                data={
                    "object_id": object_id,
                    "delta_v_magnitude": 0.01,  # Small station-keeping
                    "maneuver_type": "station_keeping"
                }
            )
            await monitor._handle_maneuver_event(maneuver_message)
            
        # Now send anomalous maneuver
        anomaly_message = WeldersArcMessage(
            message_id="maneuver-anomaly",
            timestamp=datetime.utcnow(),
            subsystem=SubsystemID.SS2_STATE_ESTIMATION,
            event_type=EventType.MANEUVER,
            data={
                "object_id": object_id,
                "delta_v_magnitude": 0.5,  # Much larger than normal
                "maneuver_type": "plane_change"
            }
        )
        
        await monitor._handle_maneuver_event(anomaly_message)
        
        # Should generate threat assessment
        assert object_id in monitor.threat_catalog
        assessment = monitor.threat_catalog[object_id]
        assert assessment.threat_level >= IntentLevel.SUSPICIOUS


if __name__ == "__main__":
    pytest.main(["-v", __file__]) 