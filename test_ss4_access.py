#!/usr/bin/env python3
"""
Test AstroShield SS4 Kafka Topic Access
Check if we can already subscribe/publish to SS4 topics
"""

import asyncio
import json
import logging
import time
import sys
from datetime import datetime
from typing import List

# Add the backend_fixed directory to Python path
sys.path.insert(0, 'backend_fixed')

from app.sda_integration.kafka.kafka_client import (
    KafkaConfig, WeldersArcKafkaClient, WeldersArcMessage
)
from app.sda_integration.kafka.standard_topics import StandardKafkaTopics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SS4AccessTest:
    """Test SS4 topic access for AstroShield"""
    
    def __init__(self):
        self.config = KafkaConfig(bootstrap_servers="localhost:9092")
        self.kafka_client = WeldersArcKafkaClient(self.config)
        self.received_messages = []
        self.test_results = {}
        
    async def message_handler(self, message: WeldersArcMessage):
        """Handle received messages"""
        self.received_messages.append(message)
        logger.info(f"✅ Received: {message.event_type} from {message.subsystem}")
        
    def get_ss4_topics_to_test(self) -> List[str]:
        """Get SS4 topics we want to test access for"""
        return [
            # Critical SS4 topics for conjunction assessment
            StandardKafkaTopics.SS4_INDICATORS_MANEUVERS_DETECTED,
            StandardKafkaTopics.SS4_INDICATORS_PROXIMITY_EVENTS_VALID_REMOTE_SENSE,
            StandardKafkaTopics.SS4_INDICATORS_OBJECT_THREAT_FROM_KNOWN_SITE,
            StandardKafkaTopics.SS4_CCDM_CCDM_DB,
            StandardKafkaTopics.SS4_CCDM_OOI,
            
            # Additional SS4 indicators
            StandardKafkaTopics.SS4_INDICATORS_IMAGING_MANEUVERS_POL_VIOLATIONS,
            StandardKafkaTopics.SS4_INDICATORS_MANEUVERS_RF_POL_OOF,
            StandardKafkaTopics.SS4_INDICATORS_SUB_SATS_DEPLOYED,
            StandardKafkaTopics.SS4_INDICATORS_VALID_IMAGING_MANEUVERS,
        ]
        
    async def test_topic_subscription(self, topic: str) -> bool:
        """Test if we can subscribe to a topic"""
        try:
            logger.info(f"🔍 Testing subscription to: {topic}")
            self.kafka_client.subscribe(topic, self.message_handler)
            logger.info(f"✅ Successfully subscribed to: {topic}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to subscribe to {topic}: {e}")
            return False
            
    async def test_topic_publish(self, topic: str) -> bool:
        """Test if we can publish to a topic"""
        try:
            logger.info(f"📤 Testing publish to: {topic}")
            
            # Create test message based on topic type
            test_message = self.create_test_message_for_topic(topic)
            
            await self.kafka_client.publish(topic, test_message)
            logger.info(f"✅ Successfully published to: {topic}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to publish to {topic}: {e}")
            return False
            
    def create_test_message_for_topic(self, topic: str) -> WeldersArcMessage:
        """Create appropriate test message for each topic type"""
        timestamp = datetime.now()
        message_id = f"test-{int(time.time())}"
        
        if "maneuvers-detected" in topic:
            return WeldersArcMessage(
                message_id=message_id,
                timestamp=timestamp,
                subsystem="ss4_ccdm",
                event_type="maneuver_detected",
                data={
                    "objectId": "TEST-12345",
                    "maneuverType": "STATION_KEEPING",
                    "detectionTime": timestamp.isoformat(),
                    "deltaV": 0.5,  # m/s
                    "confidence": 0.95,
                    "source": "astroshield_ml_detector"
                }
            )
        elif "proximity-events" in topic:
            return WeldersArcMessage(
                message_id=message_id,
                timestamp=timestamp,
                subsystem="ss4_ccdm",
                event_type="proximity_alert",
                data={
                    "primaryObject": "TEST-12345",
                    "secondaryObject": "TEST-67890",
                    "closestApproach": 0.85,  # km
                    "tca": timestamp.isoformat(),
                    "validated": True,
                    "confidence": 0.92
                }
            )
        elif "object-threat" in topic:
            return WeldersArcMessage(
                message_id=message_id,
                timestamp=timestamp,
                subsystem="ss4_ccdm",
                event_type="threat_assessment",
                data={
                    "objectId": "TEST-12345",
                    "threatLevel": "MEDIUM",
                    "knownSite": "UNKNOWN_LAUNCH_SITE",
                    "assessmentConfidence": 0.78,
                    "indicators": ["UNUSUAL_ORBIT", "MANEUVER_CAPABILITY"]
                }
            )
        elif "ccdm-db" in topic:
            return WeldersArcMessage(
                message_id=message_id,
                timestamp=timestamp,
                subsystem="ss4_ccdm",
                event_type="ccdm_update",
                data={
                    "objectId": "TEST-12345",
                    "ccdTactics": ["SIGNATURE_MANAGEMENT"],
                    "confidence": 0.85,
                    "lastUpdated": timestamp.isoformat()
                }
            )
        else:
            # Generic test message
            return WeldersArcMessage(
                message_id=message_id,
                timestamp=timestamp,
                subsystem="ss4_ccdm",
                event_type="test_message",
                data={
                    "topic": topic,
                    "test": True,
                    "timestamp": timestamp.isoformat()
                }
            )
            
    async def run_access_test(self):
        """Run comprehensive SS4 access test"""
        logger.info("🚀 AstroShield SS4 Kafka Access Test")
        logger.info("=" * 60)
        
        try:
            # Initialize Kafka client
            logger.info("📡 Initializing Kafka client...")
            await self.kafka_client.initialize()
            logger.info("✅ Connected to Kafka broker")
            
            # Get SS4 topics to test
            ss4_topics = self.get_ss4_topics_to_test()
            logger.info(f"🔍 Testing {len(ss4_topics)} SS4 topics")
            
            # Test subscription access
            logger.info("\n📥 Testing SUBSCRIPTION access...")
            subscription_results = {}
            for topic in ss4_topics:
                success = await self.test_topic_subscription(topic)
                subscription_results[topic] = success
                await asyncio.sleep(0.5)  # Brief pause between tests
                
            # Start consuming to test if subscriptions work
            consume_task = asyncio.create_task(self.kafka_client.start_consuming())
            await asyncio.sleep(2)  # Let consumer start
            
            # Test publish access
            logger.info("\n📤 Testing PUBLISH access...")
            publish_results = {}
            for topic in ss4_topics:
                success = await self.test_topic_publish(topic)
                publish_results[topic] = success
                await asyncio.sleep(0.5)  # Brief pause between tests
                
            # Wait for any messages to be received
            await asyncio.sleep(3)
            
            # Stop consuming
            await self.kafka_client.stop()
            consume_task.cancel()
            
            try:
                await consume_task
            except asyncio.CancelledError:
                pass
                
            # Analyze results
            self.analyze_results(subscription_results, publish_results)
            
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            return False
            
    def analyze_results(self, subscription_results: dict, publish_results: dict):
        """Analyze and report test results"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 SS4 ACCESS TEST RESULTS")
        logger.info("=" * 60)
        
        # Subscription results
        successful_subs = sum(1 for success in subscription_results.values() if success)
        total_subs = len(subscription_results)
        logger.info(f"📥 SUBSCRIPTION: {successful_subs}/{total_subs} topics accessible")
        
        for topic, success in subscription_results.items():
            status = "✅" if success else "❌"
            logger.info(f"  {status} {topic}")
            
        # Publish results
        successful_pubs = sum(1 for success in publish_results.values() if success)
        total_pubs = len(publish_results)
        logger.info(f"\n📤 PUBLISH: {successful_pubs}/{total_pubs} topics accessible")
        
        for topic, success in publish_results.items():
            status = "✅" if success else "❌"
            logger.info(f"  {status} {topic}")
            
        # Message reception
        logger.info(f"\n📨 MESSAGES RECEIVED: {len(self.received_messages)}")
        
        # Overall assessment
        logger.info("\n🎯 ASSESSMENT:")
        if successful_subs > 0 or successful_pubs > 0:
            logger.info("✅ AstroShield HAS SOME SS4 ACCESS!")
            logger.info("✅ Can proceed with available topics")
            
            if successful_subs == total_subs and successful_pubs == total_pubs:
                logger.info("🎉 FULL SS4 ACCESS CONFIRMED!")
            else:
                logger.info("⚠️  Partial access - may need to request additional topics")
                
        else:
            logger.info("❌ NO SS4 ACCESS DETECTED")
            logger.info("❌ Need to request SS4 topic access")
            
        # Recommendations
        logger.info("\n💡 RECOMMENDATIONS:")
        accessible_topics = [topic for topic, success in {**subscription_results, **publish_results}.items() if success]
        
        if accessible_topics:
            logger.info("1. Start with accessible topics for proof of concept")
            logger.info("2. Demonstrate value with current access")
            logger.info("3. Request additional topics based on proven success")
        else:
            logger.info("1. Submit formal request for SS4 topic access")
            logger.info("2. Focus on conjunction assessment TBD")
            logger.info("3. Use minimal topic set approach")

async def main():
    """Main test function"""
    test = SS4AccessTest()
    await test.run_access_test()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        sys.exit(1) 