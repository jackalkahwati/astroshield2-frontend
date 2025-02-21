"""Test suite for Kafka integration and message flow validation."""

import pytest
import json
from datetime import datetime, timezone
from unittest.mock import Mock, patch
import os
from pathlib import Path
import numpy as np

from models.monitoring.prediction_metrics import PerformanceTracker
from models.monitoring.real_time_monitor import RealTimeMonitor

@pytest.fixture
def mock_kafka_producer():
    """Create a mock Kafka producer for testing."""
    with patch('kafka.KafkaProducer') as mock_producer:
        producer_instance = Mock()
        mock_producer.return_value = producer_instance
        yield producer_instance

@pytest.fixture
def mock_kafka_consumer():
    """Create a mock Kafka consumer for testing."""
    with patch('kafka.KafkaConsumer') as mock_consumer:
        consumer_instance = Mock()
        mock_consumer.return_value = consumer_instance
        yield consumer_instance

@pytest.fixture
def sample_trajectory_message():
    """Create a sample trajectory prediction message."""
    return {
        "messageId": "traj_pred_001",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "predictionType": "REENTRY",
        "objectId": "ISS_RESUPPLY_27",
        "initialState": {
            "position": {"x": 6771000.0, "y": 0.0, "z": 0.0},
            "velocity": {"vx": -7.8, "vy": 0.0, "vz": 0.0},
            "epoch": datetime.now(timezone.utc).isoformat()
        },
        "predictedImpact": {
            "latitude": 28.5,
            "longitude": -80.5,
            "time": datetime.now(timezone.utc).isoformat(),
            "confidence": 0.95
        },
        "environmentalConditions": {
            "atmosphericDensity": 1.225,
            "temperature": 288.15,
            "windSpeed": 10.0,
            "windDirection": 270.0
        },
        "computationMetrics": {
            "processingTime": 0.5,
            "iterationCount": 1000,
            "convergenceStatus": "CONVERGED"
        }
    }

def test_kafka_producer_connection(mock_kafka_producer, sample_trajectory_message):
    """Test Kafka producer connection and message sending."""
    # Setup mock for successful message delivery
    future = Mock()
    future.get.return_value = Mock(
        topic='trajectory.predictions',
        partition=0,
        offset=1
    )
    mock_kafka_producer.send.return_value = future
    
    # Attempt to send message
    mock_kafka_producer.send('trajectory.predictions', value=sample_trajectory_message)
    
    # Verify producer was called correctly
    mock_kafka_producer.send.assert_called_once_with(
        'trajectory.predictions',
        value=sample_trajectory_message
    )
    
    # Verify message delivery was checked
    future.get.assert_called_once()

def test_kafka_consumer_message_processing(mock_kafka_consumer, sample_trajectory_message):
    """Test processing of incoming Kafka messages."""
    # Setup mock consumer messages
    mock_message = Mock()
    mock_message.value = json.dumps(sample_trajectory_message).encode('utf-8')
    mock_kafka_consumer.__iter__.return_value = [mock_message]
    
    # Process message
    messages = list(mock_kafka_consumer)
    assert len(messages) == 1
    
    # Verify message content
    received_message = json.loads(messages[0].value.decode('utf-8'))
    assert received_message['messageId'] == sample_trajectory_message['messageId']
    assert received_message['predictionType'] == 'REENTRY'

def test_trigger_processing(mock_kafka_consumer, performance_tracker):
    """Test processing of trigger messages."""
    # Create a sample trigger message
    trigger_message = {
        "triggerType": "REENTRY_DETECTION",
        "objectId": "ISS_RESUPPLY_27",
        "confidence": 0.95,
        "detectionTime": datetime.now(timezone.utc).isoformat(),
        "sensorId": "SENSOR_001",
        "measurements": {
            "altitude": 120000.0,  # meters
            "velocity": 7800.0,    # m/s
            "dynamic_pressure": 45000.0  # Pa
        }
    }
    
    # Setup mock consumer
    mock_message = Mock()
    mock_message.value = json.dumps(trigger_message).encode('utf-8')
    mock_kafka_consumer.__iter__.return_value = [mock_message]
    
    # Process trigger
    messages = list(mock_kafka_consumer)
    received_trigger = json.loads(messages[0].value.decode('utf-8'))
    
    # Verify trigger conditions
    assert received_trigger['triggerType'] == 'REENTRY_DETECTION'
    assert received_trigger['confidence'] >= 0.9  # Minimum confidence threshold
    assert received_trigger['measurements']['altitude'] <= 300000.0  # Max altitude threshold
    assert received_trigger['measurements']['dynamic_pressure'] >= 40000.0  # Min dynamic pressure

def test_message_format_validation(sample_trajectory_message):
    """Test validation of message format and required fields."""
    required_fields = [
        'messageId',
        'timestamp',
        'predictionType',
        'objectId',
        'initialState',
        'predictedImpact'
    ]
    
    # Check required fields
    for field in required_fields:
        assert field in sample_trajectory_message
    
    # Validate timestamp format
    timestamp = datetime.fromisoformat(sample_trajectory_message['timestamp'].replace('Z', '+00:00'))
    assert isinstance(timestamp, datetime)
    
    # Validate numerical values
    assert isinstance(sample_trajectory_message['predictedImpact']['confidence'], float)
    assert 0 <= sample_trajectory_message['predictedImpact']['confidence'] <= 1

def test_error_handling(mock_kafka_producer):
    """Test error handling for Kafka connection and message sending."""
    # Simulate connection error
    mock_kafka_producer.send.side_effect = Exception("Connection failed")
    
    with pytest.raises(Exception) as exc_info:
        mock_kafka_producer.send('trajectory.predictions', value={})
    assert "Connection failed" in str(exc_info.value)
    
    # Simulate timeout
    mock_kafka_producer.send.side_effect = None
    future = Mock()
    future.get.side_effect = TimeoutError("Message delivery timed out")
    mock_kafka_producer.send.return_value = future
    
    with pytest.raises(TimeoutError) as exc_info:
        future.get(timeout=10)
    assert "Message delivery timed out" in str(exc_info.value)

def test_performance_monitoring_integration(mock_kafka_producer, real_time_monitor):
    """Test integration of performance monitoring with Kafka messaging."""
    # Start monitoring
    real_time_monitor.start()
    
    # Simulate message sending with timing
    for _ in range(5):
        start_time = datetime.now()
        mock_kafka_producer.send('trajectory.predictions', value={})
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        real_time_monitor.record_prediction_time(processing_time)
    
    # Get performance summary
    summary = real_time_monitor.get_performance_summary(window_minutes=1)
    
    assert 'avg_prediction_latency' in summary
    assert summary['avg_prediction_latency'] > 0
    
    # Stop monitoring
    real_time_monitor.stop() 