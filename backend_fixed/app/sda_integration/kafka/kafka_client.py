"""
Kafka Message Bus Client for SDA Welders Arc Integration
Implements event-driven architecture with support for 122+ message schemas
"""

import json
import asyncio
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
import logging
from confluent_kafka import Producer, Consumer, KafkaError
from confluent_kafka.admin import AdminClient, NewTopic
from pydantic import BaseModel, Field

# Import standard topic definitions
from .standard_topics import StandardKafkaTopics, ASTROSHIELD_TOPIC_ACCESS

logger = logging.getLogger(__name__)


class KafkaConfig(BaseModel):
    """Kafka configuration for SDA Tap Lab"""
    bootstrap_servers: str = Field(default="localhost:9092")
    group_id: str = Field(default="astroshield-welders-arc")
    auto_offset_reset: str = Field(default="earliest")
    enable_auto_commit: bool = Field(default=True)
    session_timeout_ms: int = Field(default=30000)
    max_poll_interval_ms: int = Field(default=300000)


class WeldersArcMessage(BaseModel):
    """Base message schema for Welders Arc system"""
    message_id: str
    timestamp: datetime
    subsystem: str
    event_type: str
    data: Dict[str, Any]
    correlation_id: Optional[str] = None
    trace_id: Optional[str] = None
    

class EventType:
    """Seven core event types in Welders Arc"""
    LAUNCH = "launch"
    REENTRY = "reentry"
    MANEUVER = "maneuver"
    PROXIMITY = "proximity"
    SEPARATION = "separation"
    ATTITUDE_CHANGE = "attitude_change"
    LINK_CHANGE = "link_change"


class SubsystemID:
    """Welders Arc Subsystem Identifiers"""
    SS0_SENSORS = "ss0_sensors"
    SS1_TARGET_MODELING = "ss1_target_modeling"
    SS2_STATE_ESTIMATION = "ss2_state_estimation"
    SS3_COMMAND_CONTROL = "ss3_command_control"
    SS4_CCDM = "ss4_ccdm"
    SS5_HOSTILITY = "ss5_hostility"
    SS6_RESPONSE = "ss6_response"


# Use standard topic definitions
class KafkaTopics(StandardKafkaTopics):
    """Kafka topic definitions following standard naming convention"""
    
    # Create aliases for backward compatibility with existing code
    # Map old names to new standard names
    SENSOR_OBSERVATIONS = StandardKafkaTopics.SS2_DATA_OBSERVATION_TRACK
    SENSOR_HEARTBEAT = StandardKafkaTopics.SS0_SENSOR_HEARTBEAT
    SENSOR_STATUS = StandardKafkaTopics.SS0_SENSOR_HEARTBEAT
    SENSOR_CONTROL = StandardKafkaTopics.SS3_DATA_ACCESSWINDOW
    
    # State vectors and tracking
    UCT_TRACKS = StandardKafkaTopics.SS2_DATA_OBSERVATION_TRACK_TRUE_UCT
    STATE_VECTORS = StandardKafkaTopics.SS2_DATA_STATE_VECTOR
    ORBIT_DETERMINATION = StandardKafkaTopics.SS2_DATA_ORBIT_DETERMINATION
    CATALOG_CORRELATION = StandardKafkaTopics.SS2_DATA_STATE_VECTOR_CATALOG_NOMINEE
    
    # Target modeling
    TARGET_MODELS = StandardKafkaTopics.SS1_TMDB_OBJECT_UPDATED
    TARGET_MODEL_UPDATE = StandardKafkaTopics.SS1_TMDB_OBJECT_UPDATED
    TARGET_MODEL_INSERT = StandardKafkaTopics.SS1_TMDB_OBJECT_INSERTED
    
    # CCDM
    CCDM_INDICATORS = StandardKafkaTopics.SS4_CCDM_CCDM_DB
    OBJECT_INTEREST_LIST = StandardKafkaTopics.SS4_CCDM_OOI
    
    # Hostility monitoring
    THREAT_WARNING = StandardKafkaTopics.SS5_LAUNCH_INTENT_ASSESSMENT
    WEZ_PREDICTION = StandardKafkaTopics.SS5_PEZ_WEZ_PREDICTION_KKV
    HOSTILE_INTENT = StandardKafkaTopics.SS5_LAUNCH_INTENT_ASSESSMENT
    
    # Response
    ALERT_OPERATOR = StandardKafkaTopics.SS6_RESPONSE_RECOMMENDATION_ON_ORBIT
    ALERT_THREAT = StandardKafkaTopics.SS6_RESPONSE_RECOMMENDATION_LAUNCH_ASAT
    RESPONSE_STATUS = StandardKafkaTopics.SS6_RISK_MITIGATION_OPTIMAL_MANEUVER
    
    # Command & Control
    COMMAND_STATUS = StandardKafkaTopics.SS3_DATA_ACCESSWINDOW
    COMMAND_EXECUTION = StandardKafkaTopics.SS3_DATA_DETECTIONPROBABILITY
    ASSET_STATUS = StandardKafkaTopics.SS3_DATA_ACCESSWINDOW
    PLAN_EXECUTION = StandardKafkaTopics.SS3_DATA_DETECTIONPROBABILITY
    
    # Events
    EVENT_LAUNCH_DETECTION = StandardKafkaTopics.SS5_LAUNCH_DETECTION
    EVENT_MANEUVER_DETECTION = StandardKafkaTopics.SS4_INDICATORS_MANEUVERS_DETECTED
    EVENT_PROXIMITY_ALERT = StandardKafkaTopics.SS4_INDICATORS_PROXIMITY_EVENTS_VALID_REMOTE_SENSE
    EVENT_SEPARATION_DETECTED = StandardKafkaTopics.SS5_SEPARATION_DETECTION
    EVENT_RF_DETECTION = StandardKafkaTopics.SS4_INDICATORS_RF_DETECTED
    
    # Additional mappings
    FUSED_OBSERVATIONS = StandardKafkaTopics.SS5_LAUNCH_FUSED
    COLLECT_STATUS = StandardKafkaTopics.SS0_SENSOR_HEARTBEAT
    MANEUVER_PREDICTIONS = StandardKafkaTopics.SS4_INDICATORS_MANEUVERS_DETECTED
    MANEUVER_DETECTION = StandardKafkaTopics.SS4_INDICATORS_MANEUVERS_DETECTED
    METRICS = StandardKafkaTopics.UI_EVENT


class WeldersArcKafkaClient:
    """Main Kafka client for Welders Arc integration"""
    
    def __init__(self, config: KafkaConfig):
        self.config = config
        self.producer = None
        self.consumers: Dict[str, Consumer] = {}
        self.admin_client = None
        self.message_handlers: Dict[str, List[Callable]] = {}
        self._running = False
        self.topic_access = ASTROSHIELD_TOPIC_ACCESS
        
    async def initialize(self):
        """Initialize Kafka connections and create topics"""
        try:
            # Initialize producer
            self.producer = Producer({
                'bootstrap.servers': self.config.bootstrap_servers,
                'client.id': f'{self.config.group_id}-producer',
                'linger.ms': 10,
                'compression.type': 'snappy',
                'acks': 'all'
            })
            
            # Initialize admin client
            self.admin_client = AdminClient({
                'bootstrap.servers': self.config.bootstrap_servers
            })
            
            # Create topics if they don't exist
            await self._create_topics()
            
            logger.info("Kafka client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Kafka client: {e}")
            raise
            
    async def _create_topics(self):
        """Create all required topics based on access permissions"""
        topics_to_create = set()
        
        # Add topics where AstroShield has read, write, or both access
        for access_type in ['read', 'write', 'both']:
            topics_to_create.update(self.topic_access.get(access_type, []))
        
        # Create NewTopic objects
        new_topics = []
        for topic_name in topics_to_create:
            new_topics.append(NewTopic(
                topic_name,
                num_partitions=3,
                replication_factor=1,
                config={
                    'retention.ms': '604800000',  # 7 days
                    'compression.type': 'snappy'
                }
            ))
        
        # Create topics
        fs = self.admin_client.create_topics(new_topics, request_timeout=15.0)
        
        # Wait for operation to complete
        for topic, f in fs.items():
            try:
                f.result()  # The result itself is None
                logger.info(f"Topic {topic} created")
            except Exception as e:
                if 'already exists' in str(e).lower():
                    logger.debug(f"Topic {topic} already exists")
                else:
                    logger.error(f"Failed to create topic {topic}: {e}")
                    
    def _check_topic_access(self, topic: str, operation: str) -> bool:
        """Check if AstroShield has permission for the operation on the topic"""
        if operation == 'read':
            return topic in self.topic_access['read'] or topic in self.topic_access['both']
        elif operation == 'write':
            return topic in self.topic_access['write'] or topic in self.topic_access['both']
        return False
            
    def subscribe(self, topic: str, handler: Callable):
        """Subscribe to a topic with a message handler"""
        # Check read access
        if not self._check_topic_access(topic, 'read'):
            logger.warning(f"AstroShield does not have read access to topic {topic}")
            return
            
        if topic not in self.message_handlers:
            self.message_handlers[topic] = []
        self.message_handlers[topic].append(handler)
        
        # Create consumer if not exists
        if topic not in self.consumers:
            consumer = Consumer({
                'bootstrap.servers': self.config.bootstrap_servers,
                'group.id': f'{self.config.group_id}-{topic}',
                'auto.offset.reset': self.config.auto_offset_reset,
                'enable.auto.commit': self.config.enable_auto_commit,
                'session.timeout.ms': self.config.session_timeout_ms,
                'max.poll.interval.ms': self.config.max_poll_interval_ms
            })
            consumer.subscribe([topic])
            self.consumers[topic] = consumer
            
    async def publish(self, topic: str, message: WeldersArcMessage):
        """Publish a message to a topic"""
        if not self.producer:
            raise RuntimeError("Kafka producer not initialized")
        
        # Check write access
        if not self._check_topic_access(topic, 'write'):
            logger.error(f"AstroShield does not have write access to topic {topic}")
            raise PermissionError(f"No write access to topic {topic}")
            
        try:
            # Serialize message
            message_json = message.json()
            
            # Produce message
            self.producer.produce(
                topic=topic,
                key=message.message_id.encode('utf-8'),
                value=message_json.encode('utf-8'),
                on_delivery=self._delivery_callback
            )
            
            # Trigger delivery callbacks
            self.producer.poll(0)
            
        except Exception as e:
            logger.error(f"Failed to publish message to {topic}: {e}")
            raise
            
    def _delivery_callback(self, err, msg):
        """Callback for message delivery confirmation"""
        if err:
            logger.error(f"Message delivery failed: {err}")
        else:
            logger.debug(f"Message delivered to {msg.topic()} [{msg.partition()}]")
            
    async def start_consuming(self):
        """Start consuming messages from all subscribed topics"""
        self._running = True
        
        while self._running:
            try:
                # Poll each consumer
                for topic, consumer in self.consumers.items():
                    msg = consumer.poll(timeout=0.1)
                    
                    if msg is None:
                        continue
                        
                    if msg.error():
                        if msg.error().code() == KafkaError._PARTITION_EOF:
                            logger.debug(f"Reached end of partition for {topic}")
                        else:
                            logger.error(f"Consumer error: {msg.error()}")
                        continue
                        
                    # Parse message
                    try:
                        message_data = json.loads(msg.value().decode('utf-8'))
                        message = WeldersArcMessage(**message_data)
                        
                        # Call handlers
                        if topic in self.message_handlers:
                            for handler in self.message_handlers[topic]:
                                await handler(message)
                                
                    except Exception as e:
                        logger.error(f"Failed to process message from {topic}: {e}")
                        
                # Small delay to prevent busy loop
                await asyncio.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in consume loop: {e}")
                await asyncio.sleep(1)
                
    async def stop(self):
        """Stop consuming and close connections"""
        self._running = False
        
        # Close consumers
        for consumer in self.consumers.values():
            consumer.close()
            
        # Flush producer
        if self.producer:
            self.producer.flush()
            
        logger.info("Kafka client stopped")
        

class EventProcessor:
    """Process Welders Arc events through Kafka"""
    
    def __init__(self, kafka_client: WeldersArcKafkaClient):
        self.kafka_client = kafka_client
        self.event_workflows = {
            EventType.LAUNCH: self._process_launch_event,
            EventType.MANEUVER: self._process_maneuver_event,
            EventType.PROXIMITY: self._process_proximity_event,
            EventType.SEPARATION: self._process_separation_event,
            EventType.REENTRY: self._process_reentry_event,
            EventType.ATTITUDE_CHANGE: self._process_attitude_event,
            EventType.LINK_CHANGE: self._process_link_event
        }
        
    async def process_event(self, event_type: str, event_data: Dict[str, Any]):
        """Process an event through the appropriate workflow"""
        if event_type in self.event_workflows:
            await self.event_workflows[event_type](event_data)
        else:
            logger.warning(f"Unknown event type: {event_type}")
            
    async def _process_launch_event(self, event_data: Dict[str, Any]):
        """Launch event processing workflow"""
        # 1. Launch detection
        message = WeldersArcMessage(
            message_id=f"launch-{datetime.utcnow().timestamp()}",
            timestamp=datetime.utcnow(),
            subsystem=SubsystemID.SS5_HOSTILITY,
            event_type=EventType.LAUNCH,
            data=event_data
        )
        await self.kafka_client.publish(KafkaTopics.EVENT_LAUNCH_DETECTION, message)
        
        # 2. Check weather
        # 3. Generate predicted orbit
        # 4. Sensor orchestration
        # 5. Threat evaluation
        # 6. Defensive CoA
        
    async def _process_maneuver_event(self, event_data: Dict[str, Any]):
        """Maneuver event processing workflow"""
        message = WeldersArcMessage(
            message_id=f"maneuver-{datetime.utcnow().timestamp()}",
            timestamp=datetime.utcnow(),
            subsystem=SubsystemID.SS2_STATE_ESTIMATION,
            event_type=EventType.MANEUVER,
            data=event_data
        )
        await self.kafka_client.publish(KafkaTopics.EVENT_MANEUVER_DETECTION, message)
        
    async def _process_proximity_event(self, event_data: Dict[str, Any]):
        """Proximity event processing workflow"""
        message = WeldersArcMessage(
            message_id=f"proximity-{datetime.utcnow().timestamp()}",
            timestamp=datetime.utcnow(),
            subsystem=SubsystemID.SS5_HOSTILITY,
            event_type=EventType.PROXIMITY,
            data=event_data
        )
        await self.kafka_client.publish(KafkaTopics.EVENT_PROXIMITY_ALERT, message)
        
    async def _process_separation_event(self, event_data: Dict[str, Any]):
        """Separation event processing workflow"""
        message = WeldersArcMessage(
            message_id=f"separation-{datetime.utcnow().timestamp()}",
            timestamp=datetime.utcnow(),
            subsystem=SubsystemID.SS2_STATE_ESTIMATION,
            event_type=EventType.SEPARATION,
            data=event_data
        )
        await self.kafka_client.publish(KafkaTopics.EVENT_SEPARATION_DETECTED, message)
        
    async def _process_reentry_event(self, event_data: Dict[str, Any]):
        """Re-entry event processing workflow"""
        # Implementation for re-entry detection
        pass
        
    async def _process_attitude_event(self, event_data: Dict[str, Any]):
        """Attitude change event processing workflow"""
        # Implementation for attitude change detection
        pass
        
    async def _process_link_event(self, event_data: Dict[str, Any]):
        """Link state change event processing workflow"""
        # Implementation for link state changes
        pass 