#!/usr/bin/env python3
"""
Comprehensive Kafka publish/subscribe test for AstroShield
Demonstrates real-time messaging capabilities
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
    KafkaConfig, WeldersArcKafkaClient, WeldersArcMessage, KafkaTopics
)
from app.sda_integration.kafka.standard_topics import StandardKafkaTopics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KafkaTestSuite:
    """Test suite for Kafka publish/subscribe operations"""
    
    def __init__(self):
        self.config = KafkaConfig(bootstrap_servers="localhost:9092")
        self.kafka_client = WeldersArcKafkaClient(self.config)
        self.received_messages = []
        
    async def initialize(self):
        """Initialize the Kafka client"""
        logger.info("Initializing Kafka client...")
        await self.kafka_client.initialize()
        logger.info("✅ Kafka client initialized successfully")
        
    async def test_basic_publish_subscribe(self):
        """Test basic publish/subscribe functionality"""
        logger.info("🧪 Testing basic publish/subscribe...")
        
        # Subscribe to test topic
        test_topic = StandardKafkaTopics.UI_EVENT
        self.kafka_client.subscribe(test_topic, self.message_handler)
        
        # Create test message
        test_message = WeldersArcMessage(
            message_id=f"test-basic-{int(time.time())}",
            timestamp=datetime.now(),
            subsystem="ss0_sensors",
            event_type="test_event",
            data={
                "test_type": "basic_publish_subscribe",
                "message": "Hello from AstroShield Kafka test!",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Publish message
        await self.kafka_client.publish(test_topic, test_message)
        logger.info(f"📤 Published message to {test_topic}")
        
        # Start consuming in background
        consume_task = asyncio.create_task(self.consume_for_duration(5))
        
        # Wait for consumption
        await consume_task
        
        # Check if message was received
        if self.received_messages:
            logger.info("✅ Basic publish/subscribe test PASSED")
            return True
        else:
            logger.error("❌ Basic publish/subscribe test FAILED - no messages received")
            return False
            
    async def test_dnd_bogey_simulation(self):
        """Test DnD BOGEY object simulation"""
        logger.info("🧪 Testing DnD BOGEY object simulation...")
        
        # Subscribe to CCDM topic
        ccdm_topic = StandardKafkaTopics.SS4_CCDM_CCDM_DB
        self.kafka_client.subscribe(ccdm_topic, self.bogey_message_handler)
        
        # Simulate BOGEY detection
        bogey_message = WeldersArcMessage(
            message_id=f"bogey-{int(time.time())}",
            timestamp=datetime.now(),
            subsystem="ss4_ccdm",
            event_type="bogey_detection",
            data={
                "dndId": "DND-92754",
                "threatLevel": "HIGH",
                "visualMagnitude": 17.2,
                "rmsAccuracy": 2.1,
                "position": [42164.0, 0.0, 0.0],  # GEO position
                "velocity": [0.0, 3.074, 0.0],
                "suspectedTactics": ["SIGNATURE_MANAGEMENT", "DEBRIS_EVENT_COVER"],
                "confidenceScore": 0.85,
                "detectionTime": datetime.now().isoformat()
            }
        )
        
        # Publish BOGEY message
        await self.kafka_client.publish(ccdm_topic, bogey_message)
        logger.info(f"📤 Published BOGEY detection to {ccdm_topic}")
        
        # Consume messages
        consume_task = asyncio.create_task(self.consume_for_duration(3))
        await consume_task
        
        # Check for BOGEY message
        bogey_received = any(msg.event_type == "bogey_detection" for msg in self.received_messages)
        if bogey_received:
            logger.info("✅ DnD BOGEY simulation test PASSED")
            return True
        else:
            logger.error("❌ DnD BOGEY simulation test FAILED")
            return False
            
    async def test_multi_topic_publishing(self):
        """Test publishing to multiple topics"""
        logger.info("🧪 Testing multi-topic publishing...")
        
        topics_to_test = [
            StandardKafkaTopics.SS0_SENSOR_HEARTBEAT,
            StandardKafkaTopics.SS2_DATA_STATE_VECTOR,
            StandardKafkaTopics.SS4_CCDM_OOI,
            StandardKafkaTopics.UI_EVENT
        ]
        
        # Subscribe to all test topics
        for topic in topics_to_test:
            self.kafka_client.subscribe(topic, self.multi_topic_handler)
            
        # Publish to each topic
        published_count = 0
        for i, topic in enumerate(topics_to_test):
            try:
                message = WeldersArcMessage(
                    message_id=f"multi-test-{i}-{int(time.time())}",
                    timestamp=datetime.now(),
                    subsystem=f"ss{i}_test",
                    event_type="multi_topic_test",
                    data={
                        "topic": topic,
                        "sequence": i,
                        "test_type": "multi_topic_publishing"
                    }
                )
                
                await self.kafka_client.publish(topic, message)
                published_count += 1
                logger.info(f"📤 Published to {topic}")
                
            except Exception as e:
                logger.warning(f"⚠️  Failed to publish to {topic}: {e}")
                
        # Consume messages
        consume_task = asyncio.create_task(self.consume_for_duration(5))
        await consume_task
        
        # Check results
        multi_topic_messages = [msg for msg in self.received_messages if msg.event_type == "multi_topic_test"]
        received_count = len(multi_topic_messages)
        
        logger.info(f"Published to {published_count} topics, received {received_count} messages")
        
        if received_count >= published_count * 0.8:  # Allow for some message loss
            logger.info("✅ Multi-topic publishing test PASSED")
            return True
        else:
            logger.error("❌ Multi-topic publishing test FAILED")
            return False
            
    async def test_high_frequency_messaging(self):
        """Test high-frequency message publishing"""
        logger.info("🧪 Testing high-frequency messaging...")
        
        topic = StandardKafkaTopics.SS2_DATA_OBSERVATION_TRACK
        self.kafka_client.subscribe(topic, self.high_freq_handler)
        
        # Publish 10 messages rapidly
        message_count = 10
        start_time = time.time()
        
        for i in range(message_count):
            message = WeldersArcMessage(
                message_id=f"high-freq-{i}-{int(time.time() * 1000)}",
                timestamp=datetime.now(),
                subsystem="ss2_state_estimation",
                event_type="observation_track",
                data={
                    "trackId": f"TRACK-{i:03d}",
                    "sequence": i,
                    "position": [7000 + i, 0, 0],
                    "velocity": [0, 7.5, 0],
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            await self.kafka_client.publish(topic, message)
            await asyncio.sleep(0.1)  # 10 Hz publishing rate
            
        publish_duration = time.time() - start_time
        logger.info(f"📤 Published {message_count} messages in {publish_duration:.2f} seconds")
        
        # Consume messages
        consume_task = asyncio.create_task(self.consume_for_duration(3))
        await consume_task
        
        # Check results
        high_freq_messages = [msg for msg in self.received_messages if msg.event_type == "observation_track"]
        received_count = len(high_freq_messages)
        
        if received_count >= message_count * 0.9:  # Allow for minimal message loss
            logger.info(f"✅ High-frequency messaging test PASSED ({received_count}/{message_count} received)")
            return True
        else:
            logger.error(f"❌ High-frequency messaging test FAILED ({received_count}/{message_count} received)")
            return False
            
    async def message_handler(self, message: WeldersArcMessage):
        """Basic message handler"""
        self.received_messages.append(message)
        logger.info(f"📥 Received message: {message.message_id} ({message.event_type})")
        
    async def bogey_message_handler(self, message: WeldersArcMessage):
        """BOGEY-specific message handler"""
        self.received_messages.append(message)
        if message.event_type == "bogey_detection":
            bogey_data = message.data
            logger.info(f"🎯 BOGEY detected: {bogey_data.get('dndId')} - Threat: {bogey_data.get('threatLevel')}")
        else:
            logger.info(f"📥 CCDM message: {message.message_id}")
            
    async def multi_topic_handler(self, message: WeldersArcMessage):
        """Multi-topic message handler"""
        self.received_messages.append(message)
        if message.event_type == "multi_topic_test":
            topic = message.data.get('topic', 'unknown')
            sequence = message.data.get('sequence', -1)
            logger.info(f"📥 Multi-topic message {sequence} from {topic}")
        else:
            logger.info(f"📥 Message: {message.message_id}")
            
    async def high_freq_handler(self, message: WeldersArcMessage):
        """High-frequency message handler"""
        self.received_messages.append(message)
        if message.event_type == "observation_track":
            track_id = message.data.get('trackId', 'unknown')
            sequence = message.data.get('sequence', -1)
            logger.debug(f"📥 Track {track_id} (seq: {sequence})")
        else:
            logger.info(f"📥 Message: {message.message_id}")
            
    async def consume_for_duration(self, duration: int):
        """Consume messages for a specified duration"""
        logger.info(f"🔄 Starting consumption for {duration} seconds...")
        
        # Start consuming
        consume_task = asyncio.create_task(self.kafka_client.start_consuming())
        
        # Wait for specified duration
        await asyncio.sleep(duration)
        
        # Stop consuming
        await self.kafka_client.stop()
        consume_task.cancel()
        
        try:
            await consume_task
        except asyncio.CancelledError:
            pass
            
        logger.info(f"⏹️  Stopped consumption after {duration} seconds")
        
    async def run_all_tests(self):
        """Run all Kafka tests"""
        logger.info("🚀 Starting AstroShield Kafka Test Suite")
        logger.info("=" * 60)
        
        test_results = {}
        
        try:
            # Initialize
            await self.initialize()
            
            # Test 1: Basic publish/subscribe
            self.received_messages.clear()
            test_results['basic_pub_sub'] = await self.test_basic_publish_subscribe()
            
            # Test 2: DnD BOGEY simulation
            self.received_messages.clear()
            test_results['dnd_bogey_sim'] = await self.test_dnd_bogey_simulation()
            
            # Test 3: Multi-topic publishing
            self.received_messages.clear()
            test_results['multi_topic'] = await self.test_multi_topic_publishing()
            
            # Test 4: High-frequency messaging
            self.received_messages.clear()
            test_results['high_frequency'] = await self.test_high_frequency_messaging()
            
        except Exception as e:
            logger.error(f"❌ Test suite failed with error: {e}")
            return False
            
        # Results summary
        logger.info("=" * 60)
        logger.info("🏁 Test Results Summary:")
        logger.info("=" * 60)
        
        passed = 0
        total = len(test_results)
        
        for test_name, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{test_name:20} : {status}")
            if result:
                passed += 1
                
        logger.info("=" * 60)
        logger.info(f"Overall: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 ALL KAFKA TESTS PASSED!")
            logger.info("✅ AstroShield can successfully publish and subscribe to Kafka")
            logger.info("✅ DnD integration messaging is working")
            logger.info("✅ Multi-topic and high-frequency messaging supported")
        else:
            logger.warning(f"⚠️  {total - passed} test(s) failed")
            
        return passed == total


async def main():
    """Main test function"""
    test_suite = KafkaTestSuite()
    success = await test_suite.run_all_tests()
    return success


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        sys.exit(1) 