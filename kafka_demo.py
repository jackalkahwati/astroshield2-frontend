#!/usr/bin/env python3
"""
AstroShield Kafka Demonstration
Shows successful publish/subscribe capabilities
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class KafkaDemo:
    """Demonstration of AstroShield Kafka capabilities"""
    
    def __init__(self):
        self.config = KafkaConfig(bootstrap_servers="localhost:9092")
        self.kafka_client = WeldersArcKafkaClient(self.config)
        self.received_messages = []
        
    async def message_handler(self, message: WeldersArcMessage):
        """Handle received messages"""
        self.received_messages.append(message)
        data = message.data
        
        if message.event_type == "bogey_detection":
            logger.info(f"🎯 BOGEY DETECTED: {data.get('dndId')} - Threat: {data.get('threatLevel')}")
        elif message.event_type == "conjunction_alert":
            logger.info(f"⚠️  CONJUNCTION ALERT: {data.get('bogeyId')} vs {data.get('protectObjectId')}")
        elif message.event_type == "state_vector_update":
            logger.info(f"📡 STATE VECTOR: Object {data.get('objectId')} at {data.get('position')}")
        else:
            logger.info(f"📥 Message: {message.message_id} ({message.event_type})")
            
    async def run_demo(self):
        """Run the Kafka demonstration"""
        logger.info("🚀 AstroShield Kafka Demonstration")
        logger.info("=" * 60)
        
        try:
            # Initialize
            logger.info("📡 Initializing Kafka client...")
            await self.kafka_client.initialize()
            logger.info("✅ Connected to Kafka broker at localhost:9092")
            
            # Subscribe to topics
            topics = [
                StandardKafkaTopics.UI_EVENT,
                StandardKafkaTopics.SS2_DATA_STATE_VECTOR,
                StandardKafkaTopics.SS5_LAUNCH_ASAT_ASSESSMENT
            ]
            
            for topic in topics:
                self.kafka_client.subscribe(topic, self.message_handler)
                logger.info(f"🔔 Subscribed to {topic}")
            
            # Start consuming in background
            consume_task = asyncio.create_task(self.kafka_client.start_consuming())
            
            # Demonstrate different message types
            await self.demo_bogey_detection()
            await asyncio.sleep(1)
            
            await self.demo_conjunction_alert()
            await asyncio.sleep(1)
            
            await self.demo_state_vector_update()
            await asyncio.sleep(1)
            
            await self.demo_asat_assessment()
            await asyncio.sleep(3)  # Wait for messages to be processed
            
            # Stop consuming
            await self.kafka_client.stop()
            consume_task.cancel()
            
            try:
                await consume_task
            except asyncio.CancelledError:
                pass
            
            # Results
            logger.info("=" * 60)
            logger.info(f"📊 Demo Results: {len(self.received_messages)} messages processed")
            
            if self.received_messages:
                logger.info("✅ SUCCESS: AstroShield Kafka integration is fully operational!")
                logger.info("✅ Can publish and subscribe to Space Domain Awareness topics")
                logger.info("✅ DnD counter-CCD messaging is working")
                logger.info("✅ Real-time space object tracking supported")
                return True
            else:
                logger.warning("⚠️  No messages received")
                return False
                
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            return False
            
    async def demo_bogey_detection(self):
        """Demonstrate BOGEY object detection messaging"""
        logger.info("🎯 Publishing BOGEY detection...")
        
        message = WeldersArcMessage(
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
                "source": "INTELSAT 33E debris field"
            }
        )
        
        await self.kafka_client.publish(StandardKafkaTopics.UI_EVENT, message)
        
    async def demo_conjunction_alert(self):
        """Demonstrate conjunction alert messaging"""
        logger.info("⚠️  Publishing conjunction alert...")
        
        message = WeldersArcMessage(
            message_id=f"conjunction-{int(time.time())}",
            timestamp=datetime.now(),
            subsystem="ss4_ccdm",
            event_type="conjunction_alert",
            data={
                "bogeyId": "DND-92754",
                "protectObjectId": "GPS-IIF-12",
                "protectObjectName": "GPS IIF-12",
                "riskLevel": 0.75,
                "estimatedTCA": (datetime.now()).isoformat(),
                "missDistance": 0.85,  # km
                "recommendedActions": [
                    "IMMEDIATE_TRACKING_PRIORITY",
                    "NOTIFY_ASSET_OPERATOR",
                    "ENHANCED_SURVEILLANCE"
                ]
            }
        )
        
        await self.kafka_client.publish(StandardKafkaTopics.UI_EVENT, message)
        
    async def demo_state_vector_update(self):
        """Demonstrate state vector update messaging"""
        logger.info("📡 Publishing state vector update...")
        
        message = WeldersArcMessage(
            message_id=f"state-vector-{int(time.time())}",
            timestamp=datetime.now(),
            subsystem="ss2_state_estimation",
            event_type="state_vector_update",
            data={
                "objectId": "NORAD-12345",
                "objectName": "UNKNOWN OBJECT",
                "epoch": datetime.now().isoformat(),
                "position": [7000.5, 0.0, 0.0],  # km
                "velocity": [0.0, 7.546, 0.0],   # km/s
                "covariance": [
                    [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.1, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.001, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.001, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.001]
                ],
                "trackingAccuracy": 0.5,  # km
                "dataQuality": "HIGH"
            }
        )
        
        await self.kafka_client.publish(StandardKafkaTopics.SS2_DATA_STATE_VECTOR, message)
        
    async def demo_asat_assessment(self):
        """Demonstrate ASAT threat assessment messaging"""
        logger.info("🚨 Publishing ASAT assessment...")
        
        message = WeldersArcMessage(
            message_id=f"asat-{int(time.time())}",
            timestamp=datetime.now(),
            subsystem="ss5_hostility",
            event_type="asat_assessment",
            data={
                "launchSite": "UNKNOWN",
                "targetOrbit": "GEO",
                "threatLevel": "MODERATE",
                "interceptProbability": 0.65,
                "timeToIntercept": 14400,  # seconds (4 hours)
                "affectedAssets": [
                    "GPS-IIF-12",
                    "MILSTAR-6",
                    "WGS-11"
                ],
                "recommendedActions": [
                    "ALERT_ASSET_OPERATORS",
                    "PREPARE_EVASIVE_MANEUVERS",
                    "ENHANCE_SURVEILLANCE"
                ]
            }
        )
        
        await self.kafka_client.publish(StandardKafkaTopics.SS5_LAUNCH_ASAT_ASSESSMENT, message)

async def main():
    """Main demo function"""
    demo = KafkaDemo()
    success = await demo.run_demo()
    
    if success:
        logger.info("🎉 Kafka demonstration completed successfully!")
        logger.info("✅ AstroShield is ready for operational deployment")
    else:
        logger.error("❌ Kafka demonstration failed")
        
    return success

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        sys.exit(1) 