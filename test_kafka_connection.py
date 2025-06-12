#!/usr/bin/env python3
"""
Test script to verify Kafka connectivity, publishing, and subscribing
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any
import sys
import os

# Add the backend_fixed directory to Python path
sys.path.insert(0, 'backend_fixed')

from confluent_kafka import Producer, Consumer, KafkaError, KafkaException
from confluent_kafka.admin import AdminClient, NewTopic

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Kafka configuration
KAFKA_CONFIG = {
    'bootstrap.servers': 'localhost:9092',
    'client.id': 'astroshield-test-client'
}

CONSUMER_CONFIG = {
    **KAFKA_CONFIG,
    'group.id': 'astroshield-test-group',
    'auto.offset.reset': 'earliest',
    'enable.auto.commit': True
}

TEST_TOPIC = 'astroshield-test-topic'


def test_kafka_admin():
    """Test Kafka admin operations"""
    logger.info("Testing Kafka admin operations...")
    
    try:
        admin_client = AdminClient(KAFKA_CONFIG)
        
        # Get cluster metadata
        metadata = admin_client.list_topics(timeout=10)
        logger.info(f"Connected to Kafka cluster with {len(metadata.topics)} topics")
        
        # List existing topics
        logger.info("Existing topics:")
        for topic_name in metadata.topics:
            logger.info(f"  - {topic_name}")
            
        return True
        
    except Exception as e:
        logger.error(f"Admin test failed: {e}")
        return False


def test_kafka_producer():
    """Test Kafka producer"""
    logger.info("Testing Kafka producer...")
    
    try:
        producer = Producer(KAFKA_CONFIG)
        
        # Test message
        test_message = {
            "messageId": f"test-{int(time.time())}",
            "timestamp": datetime.utcnow().isoformat(),
            "source": "astroshield-test",
            "eventType": "connectivity_test",
            "data": {
                "test": True,
                "message": "Hello from AstroShield!",
                "system": "DnD Integration Test"
            }
        }
        
        # Produce message
        producer.produce(
            TEST_TOPIC,
            key="test-key",
            value=json.dumps(test_message),
            callback=delivery_callback
        )
        
        # Wait for message delivery
        producer.flush(timeout=10)
        logger.info("Message produced successfully")
        return True
        
    except Exception as e:
        logger.error(f"Producer test failed: {e}")
        return False


def delivery_callback(err, msg):
    """Callback for message delivery confirmation"""
    if err is not None:
        logger.error(f"Message delivery failed: {err}")
    else:
        logger.info(f"Message delivered to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}")


def test_kafka_consumer():
    """Test Kafka consumer"""
    logger.info("Testing Kafka consumer...")
    
    try:
        consumer = Consumer(CONSUMER_CONFIG)
        consumer.subscribe([TEST_TOPIC])
        
        logger.info(f"Subscribed to topic: {TEST_TOPIC}")
        
        # Poll for messages (timeout after 30 seconds)
        start_time = time.time()
        messages_received = 0
        
        while time.time() - start_time < 30:
            msg = consumer.poll(timeout=1.0)
            
            if msg is None:
                continue
                
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    logger.info(f"Reached end of partition {msg.partition()}")
                    continue
                else:
                    logger.error(f"Consumer error: {msg.error()}")
                    break
            
            # Process message
            try:
                message_data = json.loads(msg.value().decode('utf-8'))
                logger.info(f"Received message: {message_data['messageId']}")
                logger.info(f"Message content: {message_data}")
                messages_received += 1
                
                # Stop after receiving first message for this test
                break
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode message: {e}")
                
        consumer.close()
        
        if messages_received > 0:
            logger.info(f"Consumer test successful - received {messages_received} messages")
            return True
        else:
            logger.warning("No messages received within timeout period")
            return False
            
    except Exception as e:
        logger.error(f"Consumer test failed: {e}")
        return False


def test_standard_topics():
    """Test access to standard SDA topics"""
    logger.info("Testing standard SDA topics...")
    
    try:
        admin_client = AdminClient(KAFKA_CONFIG)
        metadata = admin_client.list_topics(timeout=10)
        
        # Standard topics we should have access to
        standard_topics = [
            'ss0.sensor.heartbeat',
            'ss2.data.state-vector',
            'ss4.ccdm.ccdm-db',
            'ss5.conjunction.events',
            'ui.event'
        ]
        
        existing_standard_topics = []
        for topic in standard_topics:
            if topic in metadata.topics:
                existing_standard_topics.append(topic)
                logger.info(f"✓ Found standard topic: {topic}")
            else:
                logger.warning(f"✗ Missing standard topic: {topic}")
                
        if existing_standard_topics:
            logger.info(f"Found {len(existing_standard_topics)} standard topics")
            return True
        else:
            logger.warning("No standard topics found - may need to create them")
            return False
            
    except Exception as e:
        logger.error(f"Standard topics test failed: {e}")
        return False


def create_test_topic():
    """Create test topic if it doesn't exist"""
    logger.info("Creating test topic...")
    
    try:
        admin_client = AdminClient(KAFKA_CONFIG)
        
        # Check if topic already exists
        metadata = admin_client.list_topics(timeout=10)
        if TEST_TOPIC in metadata.topics:
            logger.info(f"Test topic {TEST_TOPIC} already exists")
            return True
            
        # Create topic
        topic_config = NewTopic(
            topic=TEST_TOPIC,
            num_partitions=1,
            replication_factor=1
        )
        
        futures = admin_client.create_topics([topic_config])
        
        # Wait for topic creation
        for topic, future in futures.items():
            try:
                future.result(timeout=10)
                logger.info(f"Topic {topic} created successfully")
                return True
            except Exception as e:
                logger.error(f"Failed to create topic {topic}: {e}")
                return False
                
    except Exception as e:
        logger.error(f"Topic creation failed: {e}")
        return False


async def test_astroshield_kafka_client():
    """Test AstroShield's Kafka client"""
    logger.info("Testing AstroShield Kafka client...")
    
    try:
        from app.sda_integration.kafka.kafka_client import (
            KafkaConfig, WeldersArcKafkaClient, WeldersArcMessage
        )
        from app.sda_integration.kafka.standard_topics import StandardKafkaTopics
        
        # Initialize AstroShield Kafka client
        config = KafkaConfig(bootstrap_servers="localhost:9092")
        kafka_client = WeldersArcKafkaClient(config)
        
        # Test message
        test_message = WeldersArcMessage(
            message_id=f"astroshield-test-{int(time.time())}",
            timestamp=datetime.utcnow(),
            subsystem="ss0_sensors",
            event_type="test_event",
            data={
                "test": True,
                "system": "AstroShield",
                "component": "DnD Integration"
            }
        )
        
        # Try to publish to a standard topic
        await kafka_client.publish(TEST_TOPIC, test_message)
        logger.info("AstroShield Kafka client test successful")
        return True
        
    except Exception as e:
        logger.error(f"AstroShield Kafka client test failed: {e}")
        return False


def main():
    """Main test function"""
    logger.info("=" * 60)
    logger.info("AstroShield Kafka Connectivity Test")
    logger.info("=" * 60)
    
    test_results = {}
    
    # Test 1: Admin operations
    test_results['admin'] = test_kafka_admin()
    
    # Test 2: Create test topic
    test_results['topic_creation'] = create_test_topic()
    
    # Test 3: Producer
    test_results['producer'] = test_kafka_producer()
    
    # Test 4: Consumer
    test_results['consumer'] = test_kafka_consumer()
    
    # Test 5: Standard topics
    test_results['standard_topics'] = test_standard_topics()
    
    # Test 6: AstroShield client (async)
    try:
        test_results['astroshield_client'] = asyncio.run(test_astroshield_kafka_client())
    except Exception as e:
        logger.error(f"AstroShield client test error: {e}")
        test_results['astroshield_client'] = False
    
    # Summary
    logger.info("=" * 60)
    logger.info("Test Results Summary:")
    logger.info("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{test_name:20} : {status}")
        if result:
            passed += 1
    
    logger.info("=" * 60)
    logger.info(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All Kafka connectivity tests PASSED!")
        logger.info("✅ AstroShield can successfully connect to, publish to, and subscribe from Kafka")
    else:
        logger.warning(f"⚠️  {total - passed} test(s) failed")
        logger.info("Some Kafka functionality may not be working correctly")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 