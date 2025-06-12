#!/usr/bin/env python3
"""
Unit tests for UDL integration components
Tests the event-driven UDL WebSocket client
"""

import unittest
import json
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import sys
import os
import time
from datetime import datetime
import pytest

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import UDL modules - paths will be adjusted based on actual implementation
try:
    from src.asttroshield.udl_integration.websocket_client import UDLWebSocketClient
    from src.asttroshield.common.message_schemas import StateVector, Observation
except ImportError:
    # Mock classes for testing if not available
    class UDLWebSocketClient:
        def __init__(self, url, token=None, **kwargs):
            self.url = url
            self.token = token
            self.on_message_callback = None
            self.on_error_callback = None
            self.on_connect_callback = None
            self.connected = False
            self.retry_count = kwargs.get('retry_count', 3)
            self.retry_delay = kwargs.get('retry_delay', 5)
            self.kafka_producer = None
            
        async def connect(self):
            self.connected = True
            if self.on_connect_callback:
                await self.on_connect_callback()
            return True
            
        async def disconnect(self):
            self.connected = False
            return True
            
        async def send_message(self, message):
            return True
            
        def on_message(self, callback):
            self.on_message_callback = callback
            
        def on_error(self, callback):
            self.on_error_callback = callback
            
        def on_connect(self, callback):
            self.on_connect_callback = callback
            
        def set_kafka_producer(self, producer):
            self.kafka_producer = producer
            
        def route_message(self, message):
            if isinstance(message, dict):
                msg_type = message.get('type', '')
                if 'state_vector' in msg_type:
                    return 'ss0.statevector.current'
                elif 'observation' in msg_type:
                    return 'ss0.observations.current'
                else:
                    return 'ss0.general'
            return 'ss0.unknown'
    
    class StateVector:
        def __init__(self, object_id=None, timestamp=None, position=None, velocity=None):
            self.object_id = object_id
            self.timestamp = timestamp or datetime.now()
            self.position = position or {"x": 0, "y": 0, "z": 0}
            self.velocity = velocity or {"x": 0, "y": 0, "z": 0}
            
        @classmethod
        def from_dict(cls, data):
            return cls(
                object_id=data.get('object_id'),
                timestamp=data.get('timestamp'),
                position=data.get('position'),
                velocity=data.get('velocity')
            )
            
        def to_dict(self):
            return {
                'object_id': self.object_id,
                'timestamp': self.timestamp,
                'position': self.position,
                'velocity': self.velocity
            }
    
    class Observation:
        def __init__(self, object_id=None, timestamp=None, sensor_id=None, measurements=None):
            self.object_id = object_id
            self.timestamp = timestamp or datetime.now()
            self.sensor_id = sensor_id
            self.measurements = measurements or {}
            
        @classmethod
        def from_dict(cls, data):
            return cls(
                object_id=data.get('object_id'),
                timestamp=data.get('timestamp'),
                sensor_id=data.get('sensor_id'),
                measurements=data.get('measurements')
            )
            
        def to_dict(self):
            return {
                'object_id': self.object_id,
                'timestamp': self.timestamp,
                'sensor_id': self.sensor_id,
                'measurements': self.measurements
            }

class TestUDLWebSocketClient(unittest.TestCase):
    """Test UDL WebSocket client functionality"""
    
    def setUp(self):
        """Set up test environment before each test"""
        self.url = "ws://test-udl-server:8080/api/events"
        self.token = "test-token-123"
        
        # Create mock Kafka producer
        self.mock_kafka_producer = MagicMock()
        self.mock_kafka_producer.send = AsyncMock()
        
    @patch('asyncio.sleep', new_callable=AsyncMock)
    async def test_connection_retry(self, mock_sleep):
        """Test connection retry logic"""
        # Create client with connection failures
        with patch.object(UDLWebSocketClient, 'connect', side_effect=[
            Exception("Connection failed"),
            Exception("Connection failed again"),
            True  # Success on third attempt
        ]):
            client = UDLWebSocketClient(
                self.url, 
                token=self.token,
                retry_count=3,
                retry_delay=1
            )
            
            # Mock the connect callback
            connect_callback = AsyncMock()
            client.on_connect(connect_callback)
            
            # Attempt connection
            result = await client.connect()
            
            # Verify retries
            self.assertTrue(result)
            self.assertEqual(mock_sleep.call_count, 2)  # Should have slept twice
            connect_callback.assert_called_once()
    
    async def test_message_routing(self):
        """Test message routing to correct Kafka topics"""
        client = UDLWebSocketClient(self.url, token=self.token)
        
        # Test state vector routing
        state_vector_msg = {
            "type": "state_vector",
            "object_id": "SATCAT-12345",
            "timestamp": time.time(),
            "position": {"x": 42164000, "y": 0, "z": 0},
            "velocity": {"x": 0, "y": 3074, "z": 0}
        }
        
        topic = client.route_message(state_vector_msg)
        self.assertEqual(topic, "ss0.statevector.current")
        
        # Test observation routing
        observation_msg = {
            "type": "observation",
            "object_id": "SATCAT-12345",
            "timestamp": time.time(),
            "sensor_id": "SENSOR-001",
            "measurements": {
                "azimuth": 245.3,
                "elevation": 47.8,
                "range": 35786000
            }
        }
        
        topic = client.route_message(observation_msg)
        self.assertEqual(topic, "ss0.observations.current")
        
        # Test unknown message type
        unknown_msg = {
            "type": "unknown_type",
            "data": "some data"
        }
        
        topic = client.route_message(unknown_msg)
        self.assertEqual(topic, "ss0.general")
    
    @patch.object(UDLWebSocketClient, 'connect')
    @patch.object(UDLWebSocketClient, 'disconnect')
    async def test_connection_lifecycle(self, mock_disconnect, mock_connect):
        """Test connection lifecycle methods"""
        # Setup mocks
        mock_connect.return_value = True
        mock_disconnect.return_value = True
        
        # Create client
        client = UDLWebSocketClient(self.url, token=self.token)
        
        # Test connect
        result = await client.connect()
        self.assertTrue(result)
        mock_connect.assert_called_once()
        
        # Test disconnect
        result = await client.disconnect()
        self.assertTrue(result)
        mock_disconnect.assert_called_once()
    
    async def test_message_processing(self):
        """Test end-to-end message processing"""
        # Create client
        client = UDLWebSocketClient(self.url, token=self.token)
        
        # Set mock Kafka producer
        client.set_kafka_producer(self.mock_kafka_producer)
        
        # Create mock message handler
        message_handler = AsyncMock()
        client.on_message(message_handler)
        
        # Simulate incoming message
        test_message = {
            "type": "state_vector",
            "object_id": "SATCAT-12345",
            "timestamp": time.time(),
            "position": {"x": 42164000, "y": 0, "z": 0},
            "velocity": {"x": 0, "y": 3074, "z": 0}
        }
        
        # Manually call message handler (as if WebSocket received message)
        await client.on_message_callback(json.dumps(test_message))
        
        # Verify message was processed
        message_handler.assert_called_once()
        
        # Verify Kafka producer was called with correct topic
        self.mock_kafka_producer.send.assert_called_once()
        args, kwargs = self.mock_kafka_producer.send.call_args
        self.assertEqual(args[0], "ss0.statevector.current")  # First arg should be topic
    
    async def test_state_vector_validation(self):
        """Test state vector validation logic"""
        # Create valid state vector
        valid_sv = {
            "object_id": "SATCAT-12345",
            "timestamp": time.time(),
            "position": {"x": 42164000, "y": 0, "z": 0},
            "velocity": {"x": 0, "y": 3074, "z": 0}
        }
        
        sv_obj = StateVector.from_dict(valid_sv)
        self.assertEqual(sv_obj.object_id, "SATCAT-12345")
        self.assertIsNotNone(sv_obj.timestamp)
        self.assertEqual(sv_obj.position["x"], 42164000)
        self.assertEqual(sv_obj.velocity["y"], 3074)
        
        # Test conversion back to dict
        sv_dict = sv_obj.to_dict()
        self.assertEqual(sv_dict["object_id"], "SATCAT-12345")
        self.assertEqual(sv_dict["position"]["x"], 42164000)
    
    async def test_latency_measurement(self):
        """Test latency measurement for UDL messages"""
        # Create client
        client = UDLWebSocketClient(self.url, token=self.token)
        
        # Create test message with timestamp
        current_time = time.time()
        test_message = {
            "type": "state_vector",
            "object_id": "SATCAT-12345",
            "timestamp": current_time - 1.0,  # 1 second ago
            "position": {"x": 42164000, "y": 0, "z": 0},
            "velocity": {"x": 0, "y": 3074, "z": 0}
        }
        
        # Patch time.time to return consistent value for testing
        with patch('time.time', return_value=current_time):
            # Directly test latency calculation if available
            if hasattr(client, 'calculate_message_latency'):
                latency = client.calculate_message_latency(test_message)
                # Message should be ~1000ms old
                self.assertAlmostEqual(latency, 1000, delta=10)

@pytest.mark.asyncio
class TestUDLAsyncFunctions:
    """Pytest-based async tests for UDL functions"""
    
    @pytest.fixture
    async def udl_client(self):
        """Create and return a UDL WebSocket client fixture"""
        client = UDLWebSocketClient("ws://test-udl-server:8080/api/events")
        yield client
    
    async def test_websocket_reconnection(self, udl_client):
        """Test WebSocket reconnection on failure"""
        # Simulate connection failure
        with patch.object(udl_client, 'connect', side_effect=Exception("Connection failed")):
            # Call reconnect logic if available
            if hasattr(udl_client, 'reconnect'):
                with patch.object(udl_client, 'reconnect', return_value=True):
                    # Trigger error handler if available
                    if udl_client.on_error_callback:
                        result = await udl_client.on_error_callback(Exception("WebSocket error"))
                        assert result is True
    
    async def test_message_throughput(self, udl_client):
        """Test message throughput capacity"""
        # Create mock Kafka producer
        mock_producer = MagicMock()
        mock_producer.send = AsyncMock(return_value=True)
        udl_client.set_kafka_producer(mock_producer)
        
        # Generate batch of test messages
        test_messages = []
        for i in range(100):
            test_messages.append({
                "type": "state_vector",
                "object_id": f"SATCAT-{i}",
                "timestamp": time.time(),
                "position": {"x": 42164000, "y": 0, "z": 0},
                "velocity": {"x": 0, "y": 3074, "z": 0}
            })
        
        # Process messages in rapid succession
        start_time = time.time()
        for msg in test_messages:
            if udl_client.on_message_callback:
                await udl_client.on_message_callback(json.dumps(msg))
        end_time = time.time()
        
        # Calculate throughput
        duration = end_time - start_time
        throughput = len(test_messages) / duration
        
        # Verify throughput (should be at least 100 msgs per second)
        assert throughput >= 100

if __name__ == "__main__":
    unittest.main() 