#!/usr/bin/env python3
"""
Simple Kafka connectivity test using only permitted topics
"""

import asyncio
import json
import logging
import time
from datetime import datetime
import sys

# Add the backend_fixed directory to Python path
sys.path.insert(0, 'backend_fixed')

from app.sda_integration.kafka.kafka_client import (
    KafkaConfig, WeldersArcKafkaClient, WeldersArcMessage
)
from app.sda_integration.kafka.standard_topics import StandardKafkaTopics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_kafka_connectivity():
    """Test basic Kafka connectivity with permitted topics"""
    logger.info("🚀 Testing AstroShield Kafka Connectivity")
    logger.info("=" * 50)
    
    # Initialize Kafka client
    config = KafkaConfig(bootstrap_servers="localhost:9092")
    kafka_client = WeldersArcKafkaClient(config)
    
    try:
        # Initialize client
        logger.info("📡 Initializing Kafka client...")
        await kafka_client.initialize()
        logger.info("✅ Kafka client initialized successfully")
        
        # Test topics we have write access to
        test_topics = [
            StandardKafkaTopics.UI_EVENT,  # Both read/write
            StandardKafkaTopics.SS2_DATA_STATE_VECTOR,  # Both read/write
            StandardKafkaTopics.SS5_LAUNCH_ASAT_ASSESSMENT,  # Both read/write
        ]
        
        messages_sent = 0
        messages_received = []
        
        # Subscribe to topics
        def message_handler(message):
            messages_received.append(message)
            logger.info(f"📥 Received: {message.message_id} ({message.event_type})")
            
        for topic in test_topics:
            kafka_client.subscribe(topic, message_handler)
            logger.info(f"🔔 Subscribed to {topic}")
        
        # Publish test messages
        for i, topic in enumerate(test_topics):
            try:
                message = WeldersArcMessage(
                    message_id=f"test-{i}-{int(time.time())}",
                    timestamp=datetime.now(),
                    subsystem=f"ss{i}_test",
                    event_type="connectivity_test",
                    data={
                        "test_number": i,
                        "topic": topic,
                        "message": f"Test message {i} for AstroShield Kafka",
                        "timestamp": datetime.now().isoformat()
                    }
                )
                
                await kafka_client.publish(topic, message)
                messages_sent += 1
                logger.info(f"📤 Published to {topic}")
                
            except Exception as e:
                logger.error(f"❌ Failed to publish to {topic}: {e}")
        
        # Start consuming
        logger.info("🔄 Starting message consumption...")
        consume_task = asyncio.create_task(kafka_client.start_consuming())
        
        # Wait for messages
        await asyncio.sleep(5)
        
        # Stop consuming
        await kafka_client.stop()
        consume_task.cancel()
        
        try:
            await consume_task
        except asyncio.CancelledError:
            pass
        
        # Results
        logger.info("=" * 50)
        logger.info("📊 Test Results:")
        logger.info(f"   Messages sent: {messages_sent}")
        logger.info(f"   Messages received: {len(messages_received)}")
        
        if len(messages_received) > 0:
            logger.info("✅ SUCCESS: Kafka publish/subscribe is working!")
            logger.info("✅ AstroShield can communicate via Kafka message bus")
            
            # Show received messages
            for msg in messages_received:
                logger.info(f"   📨 {msg.message_id}: {msg.data.get('message', 'No message')}")
                
            return True
        else:
            logger.warning("⚠️  No messages received - check topic permissions")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

async def test_dnd_messaging():
    """Test DnD-specific messaging patterns"""
    logger.info("🎯 Testing DnD Messaging Patterns")
    logger.info("=" * 50)
    
    config = KafkaConfig(bootstrap_servers="localhost:9092")
    kafka_client = WeldersArcKafkaClient(config)
    
    try:
        await kafka_client.initialize()
        
        # Use UI_EVENT topic for DnD alerts (we have both read/write access)
        dnd_topic = StandardKafkaTopics.UI_EVENT
        
        received_messages = []
        
        def dnd_handler(message):
            received_messages.append(message)
            if message.event_type == "bogey_alert":
                data = message.data
                logger.info(f"🎯 BOGEY Alert: {data.get('dndId')} - Threat: {data.get('threatLevel')}")
            else:
                logger.info(f"📥 DnD Message: {message.message_id}")
        
        kafka_client.subscribe(dnd_topic, dnd_handler)
        
        # Simulate DnD BOGEY alert
        bogey_alert = WeldersArcMessage(
            message_id=f"bogey-alert-{int(time.time())}",
            timestamp=datetime.now(),
            subsystem="ss4_ccdm",
            event_type="bogey_alert",
            data={
                "alertType": "BOGEY_DETECTION",
                "dndId": "DND-92754",
                "threatLevel": "HIGH",
                "visualMagnitude": 17.2,
                "position": [42164.0, 0.0, 0.0],
                "suspectedTactics": ["SIGNATURE_MANAGEMENT"],
                "confidenceScore": 0.85,
                "recommendedActions": ["ENHANCED_SURVEILLANCE", "NOTIFY_ASSET_OPERATOR"]
            }
        )
        
        # Simulate conjunction alert
        conjunction_alert = WeldersArcMessage(
            message_id=f"conjunction-alert-{int(time.time())}",
            timestamp=datetime.now(),
            subsystem="ss4_ccdm",
            event_type="conjunction_alert",
            data={
                "alertType": "BOGEY_CONJUNCTION",
                "bogeyId": "DND-92754",
                "protectObjectId": "GPS-IIF-12",
                "riskLevel": 0.75,
                "estimatedTCA": (datetime.now()).isoformat(),
                "recommendedActions": ["IMMEDIATE_TRACKING_PRIORITY", "NOTIFY_ASSET_OPERATOR"]
            }
        )
        
        # Publish alerts
        await kafka_client.publish(dnd_topic, bogey_alert)
        await kafka_client.publish(dnd_topic, conjunction_alert)
        logger.info("📤 Published DnD alerts")
        
        # Consume messages
        consume_task = asyncio.create_task(kafka_client.start_consuming())
        await asyncio.sleep(3)
        await kafka_client.stop()
        consume_task.cancel()
        
        try:
            await consume_task
        except asyncio.CancelledError:
            pass
        
        # Check results
        dnd_messages = [msg for msg in received_messages if msg.event_type in ["bogey_alert", "conjunction_alert"]]
        
        logger.info("=" * 50)
        logger.info(f"📊 DnD Messages: {len(dnd_messages)} received")
        
        if dnd_messages:
            logger.info("✅ SUCCESS: DnD messaging patterns working!")
            return True
        else:
            logger.warning("⚠️  No DnD messages received")
            return False
            
    except Exception as e:
        logger.error(f"❌ DnD messaging test failed: {e}")
        return False

async def main():
    """Main test function"""
    logger.info("🛡️  AstroShield Kafka Connectivity Test")
    logger.info("=" * 60)
    
    # Test 1: Basic connectivity
    basic_test = await test_kafka_connectivity()
    
    # Test 2: DnD messaging
    dnd_test = await test_dnd_messaging()
    
    # Summary
    logger.info("=" * 60)
    logger.info("🏁 Final Results:")
    logger.info(f"   Basic Connectivity: {'✅ PASS' if basic_test else '❌ FAIL'}")
    logger.info(f"   DnD Messaging:      {'✅ PASS' if dnd_test else '❌ FAIL'}")
    
    if basic_test and dnd_test:
        logger.info("🎉 ALL TESTS PASSED!")
        logger.info("✅ AstroShield can successfully publish and subscribe to Kafka")
        logger.info("✅ DnD integration messaging is operational")
        return True
    else:
        logger.warning("⚠️  Some tests failed")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test failed: {e}")
        sys.exit(1) 