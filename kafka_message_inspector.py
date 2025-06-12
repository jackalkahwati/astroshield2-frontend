#!/usr/bin/env python3
"""
Kafka Message Inspector
Captures and displays the actual JSON content of Kafka messages
"""

import asyncio
import json
import logging
import time
import sys
from datetime import datetime
from typing import Dict, List

# Add the backend_fixed directory to Python path
sys.path.insert(0, 'backend_fixed')

from app.sda_integration.kafka.kafka_client import (
    KafkaConfig, WeldersArcKafkaClient, WeldersArcMessage
)
from app.sda_integration.kafka.standard_topics import StandardKafkaTopics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KafkaMessageInspector:
    """Inspect and display actual Kafka message content"""
    
    def __init__(self):
        self.config = KafkaConfig(bootstrap_servers="localhost:9092")
        self.kafka_client = WeldersArcKafkaClient(self.config)
        self.captured_messages = []
        self.raw_message_data = []
        
    async def message_handler(self, message: WeldersArcMessage):
        """Capture and display message details"""
        self.captured_messages.append(message)
        
        # Extract raw message data
        message_data = {
            "message_id": message.message_id,
            "timestamp": message.timestamp.isoformat() if message.timestamp else None,
            "subsystem": message.subsystem,
            "event_type": message.event_type,
            "data": message.data,
            "received_at": datetime.now().isoformat()
        }
        
        self.raw_message_data.append(message_data)
        
        logger.info(f"📨 Captured message: {message.event_type} from {message.subsystem}")
        
        # Display the JSON immediately
        self.display_message_json(message_data, len(self.captured_messages))
        
    def display_message_json(self, message_data: Dict, message_number: int):
        """Display formatted JSON of the message"""
        print(f"\n{'='*80}")
        print(f"📨 MESSAGE #{message_number} - {message_data['event_type']}")
        print(f"{'='*80}")
        print(json.dumps(message_data, indent=2, default=str))
        print(f"{'='*80}\n")
        
    async def publish_test_message(self):
        """Publish a test message to generate content"""
        logger.info("📤 Publishing test message to generate content...")
        
        test_message = WeldersArcMessage(
            message_id=f"inspector-test-{int(time.time())}",
            timestamp=datetime.now(),
            subsystem="astroshield_inspector",
            event_type="maneuver_detected",
            data={
                "objectId": "INSPECTOR-TEST-001",
                "objectName": "Test Satellite",
                "maneuverType": "STATION_KEEPING",
                "detectionTime": datetime.now().isoformat(),
                "deltaV": 0.75,  # m/s
                "confidence": 0.96,
                "source": "astroshield_spatiotemporal_ai",
                "processing_details": {
                    "algorithm": "divided_space_time_attention",
                    "f1_score": 0.94,
                    "processing_time_ms": 42,
                    "model_version": "v2.1.3"
                },
                "orbital_elements": {
                    "semi_major_axis": 42164.0,  # km
                    "eccentricity": 0.0001,
                    "inclination": 0.1,  # degrees
                    "raan": 75.3,  # degrees
                    "arg_perigee": 0.0,  # degrees
                    "mean_anomaly": 123.4  # degrees
                },
                "threat_assessment": {
                    "risk_level": "LOW",
                    "intent_probabilities": {
                        "station_keeping": 0.89,
                        "collision_avoidance": 0.06,
                        "inspection": 0.03,
                        "hostile": 0.02
                    }
                },
                "metadata": {
                    "sensor_source": "SSN_RADAR",
                    "tracking_accuracy": 0.5,  # km
                    "observation_count": 47,
                    "last_observation": datetime.now().isoformat()
                }
            }
        )
        
        try:
            await self.kafka_client.publish(
                StandardKafkaTopics.SS4_INDICATORS_MANEUVERS_DETECTED,
                test_message
            )
            logger.info("✅ Test message published successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to publish test message: {e}")
            return False
            
    async def inspect_messages(self, duration_seconds: int = 10):
        """Inspect messages for a specified duration"""
        logger.info("🔍 Starting Kafka Message Inspector")
        logger.info("=" * 80)
        
        try:
            # Initialize
            logger.info("📡 Initializing Kafka client...")
            await self.kafka_client.initialize()
            logger.info("✅ Connected to Kafka broker")
            
            # Subscribe to SS4 topics we have read access to
            topics_to_monitor = [
                StandardKafkaTopics.SS4_INDICATORS_MANEUVERS_DETECTED,
                StandardKafkaTopics.SS4_INDICATORS_PROXIMITY_EVENTS_VALID_REMOTE_SENSE,
                StandardKafkaTopics.SS4_INDICATORS_OBJECT_THREAT_FROM_KNOWN_SITE,
                StandardKafkaTopics.SS4_CCDM_CCDM_DB,
                StandardKafkaTopics.SS4_CCDM_OOI,
            ]
            
            for topic in topics_to_monitor:
                try:
                    self.kafka_client.subscribe(topic, self.message_handler)
                    logger.info(f"🔔 Subscribed to: {topic}")
                except Exception as e:
                    logger.warning(f"⚠️  Could not subscribe to {topic}: {e}")
                    
            # Start consuming
            consume_task = asyncio.create_task(self.kafka_client.start_consuming())
            logger.info(f"👂 Listening for messages for {duration_seconds} seconds...")
            
            # Wait a moment for consumer to start
            await asyncio.sleep(2)
            
            # Publish a test message to ensure we have content
            await self.publish_test_message()
            
            # Listen for the specified duration
            await asyncio.sleep(duration_seconds)
            
            # Stop consuming
            await self.kafka_client.stop()
            consume_task.cancel()
            
            try:
                await consume_task
            except asyncio.CancelledError:
                pass
                
            # Display summary
            self.display_summary()
            
        except Exception as e:
            logger.error(f"❌ Inspector failed: {e}")
            return False
            
    def display_summary(self):
        """Display summary of captured messages"""
        logger.info("\n" + "=" * 80)
        logger.info("📊 KAFKA MESSAGE INSPECTION SUMMARY")
        logger.info("=" * 80)
        
        logger.info(f"📨 Total Messages Captured: {len(self.captured_messages)}")
        
        if self.captured_messages:
            logger.info("\n📋 Message Types Received:")
            event_types = {}
            for msg in self.captured_messages:
                event_type = msg.event_type
                if event_type in event_types:
                    event_types[event_type] += 1
                else:
                    event_types[event_type] = 1
                    
            for event_type, count in event_types.items():
                logger.info(f"   • {event_type}: {count} messages")
                
            # Display the most recent message in detail
            if self.raw_message_data:
                logger.info("\n🔍 MOST RECENT MESSAGE (Full JSON):")
                latest_message = self.raw_message_data[-1]
                print("\n" + "="*80)
                print("📨 LATEST MESSAGE JSON")
                print("="*80)
                print(json.dumps(latest_message, indent=2, default=str))
                print("="*80)
                
        else:
            logger.info("❌ No messages were captured during inspection period")
            logger.info("💡 This could mean:")
            logger.info("   • No active message traffic on monitored topics")
            logger.info("   • Access permissions may be limited")
            logger.info("   • Kafka broker connectivity issues")
            
    def save_messages_to_file(self, filename: str = "captured_messages.json"):
        """Save captured messages to a JSON file"""
        if self.raw_message_data:
            try:
                with open(filename, 'w') as f:
                    json.dump(self.raw_message_data, f, indent=2, default=str)
                logger.info(f"💾 Saved {len(self.raw_message_data)} messages to {filename}")
            except Exception as e:
                logger.error(f"❌ Failed to save messages: {e}")
        else:
            logger.info("❌ No messages to save")

async def main():
    """Main inspector function"""
    inspector = KafkaMessageInspector()
    
    # Inspect messages for 15 seconds
    await inspector.inspect_messages(duration_seconds=15)
    
    # Save messages to file
    inspector.save_messages_to_file()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Inspector interrupted by user")
    except Exception as e:
        logger.error(f"Inspector failed: {e}")
        sys.exit(1) 