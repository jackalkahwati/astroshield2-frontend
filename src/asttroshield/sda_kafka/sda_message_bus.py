"""
SDA Kafka Message Bus Integration
Connects to SDA Subsystem 4 Kafka bus with AWS MSK for event-driven architecture
Based on SDA Welders Arc specifications and requirements
"""

import os
import json
import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum
import uuid

try:
    from confluent_kafka import Producer, Consumer, KafkaError, KafkaException
    from confluent_kafka.admin import AdminClient, NewTopic
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    
try:
    from pydantic import BaseModel, Field, validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

# Import SDA schemas
try:
    from .sda_schemas import (
        SDAManeuverDetected, SDALaunchDetected, SDATLEUpdate,
        SDASchemaFactory, validate_sda_schema
    )
    SDA_SCHEMAS_AVAILABLE = True
except ImportError:
    SDA_SCHEMAS_AVAILABLE = False

logger = logging.getLogger(__name__)


class SDASubsystem(Enum):
    """SDA Subsystem identifiers"""
    SS0_DATA_INGESTION = "SS0"
    SS1_TARGET_MODELING = "SS1"
    SS2_STATE_ESTIMATION = "SS2"
    SS3_COMMAND_CONTROL = "SS3"
    SS4_CCDM = "SS4"
    SS5_HOSTILITY_MONITORING = "SS5"
    SS6_THREAT_ASSESSMENT = "SS6"


class MessagePriority(Enum):
    """Message priority levels for SDA topics"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SDAKafkaCredentials:
    """SDA Kafka authentication credentials"""
    bootstrap_servers: str
    username: str
    password: str
    ssl_ca_location: Optional[str] = None
    security_protocol: str = "SASL_SSL"
    sasl_mechanism: str = "SCRAM-SHA-512"
    
    @classmethod
    def from_environment(cls) -> 'SDAKafkaCredentials':
        """Load credentials from environment variables"""
        return cls(
            bootstrap_servers=os.getenv('SDA_KAFKA_BOOTSTRAP_SERVERS', 'kafka.sda.mil:9092'),
            username=os.getenv('SDA_KAFKA_USERNAME'),
            password=os.getenv('SDA_KAFKA_PASSWORD'),
            ssl_ca_location=os.getenv('SDA_KAFKA_SSL_CA_LOCATION'),
            security_protocol=os.getenv('SDA_KAFKA_SECURITY_PROTOCOL', 'SASL_SSL'),
            sasl_mechanism=os.getenv('SDA_KAFKA_SASL_MECHANISM', 'SCRAM-SHA-512')
        )


if PYDANTIC_AVAILABLE:
    class SDAMessageSchema(BaseModel):
        """Base schema for SDA Kafka messages"""
        message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
        timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
        source_system: str = Field(description="Source system identifier (e.g., astroshield)")
        subsystem: SDASubsystem = Field(description="SDA subsystem")
        message_type: str = Field(description="Message type identifier")
        priority: MessagePriority = Field(default=MessagePriority.NORMAL)
        correlation_id: Optional[str] = Field(None, description="For message correlation")
        data: Dict[str, Any] = Field(description="Message payload data")
        metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
        
        @validator('timestamp', pre=True)
        def validate_timestamp(cls, v):
            if isinstance(v, str):
                return datetime.fromisoformat(v.replace('Z', '+00:00'))
            return v
        
        class Config:
            json_encoders = {
                datetime: lambda v: v.isoformat()
            }
else:
    # Fallback implementation without Pydantic
    class SDAMessageSchema:
        """Base schema for SDA Kafka messages"""
        def __init__(self, **kwargs):
            self.message_id = kwargs.get('message_id', str(uuid.uuid4()))
            self.timestamp = kwargs.get('timestamp', datetime.now(timezone.utc))
            self.source_system = kwargs.get('source_system', 'astroshield')
            self.subsystem = kwargs.get('subsystem', SDASubsystem.SS1_TARGET_MODELING)
            self.message_type = kwargs.get('message_type', 'general')
            self.priority = kwargs.get('priority', MessagePriority.NORMAL)
            self.correlation_id = kwargs.get('correlation_id')
            self.data = kwargs.get('data', {})
            self.metadata = kwargs.get('metadata', {})
        
        def json(self):
            return json.dumps(self.__dict__, default=str)


class SDATopicManager:
    """Manages SDA Kafka topic structure and routing"""
    
    # SDA Topic naming convention: {subsystem}.{category}.{subcategory}
    # Based on official SDA Welders Arc documentation
    TOPICS = {
        # === SS0 DATA INGESTION (Official Topics) ===
        
        # Weather Data (EarthCast)
        "weather_lightning": "ss0.data.weather.lightning",
        "weather_orbital_density": "ss0.data.weather.realtime-orbital-density-predictions",
        "weather_clouds": "ss0.data.weather.clouds", 
        "weather_reflectivity": "ss0.data.weather.reflectivity",
        "weather_turbulence": "ss0.data.weather.turbulence",
        "weather_vtec": "ss0.data.weather.vtec",
        "weather_windshear_low": "ss0.data.weather.windshear-low-level",
        "weather_windshear_jet": "ss0.data.weather.windshear-jetstream-level",
        "weather_neutral_density": "ss0.data.weather.neutral-densitiy",
        
        # Launch and Sensor Data
        "ss0_launch_detection": "ss0.data.launch-detection",
        "ss0_launch_prediction": "ss0.launch-prediction.launch-window",
        "ss0_sensor_heartbeat": "ss0.sensor.heartbeat",
        "ss0_overhead_images": "ss0.data.over-head-images",
        
        # === SS2 STATE ESTIMATION (Official Topics) ===
        
        # Elset Data
        "elset_sgp4": "ss2.data.elset.sgp4",
        "elset_sgp4_xp": "ss2.data.elset.sgp4-xp", 
        "elset_best_state": "ss2.data.elset.best-state",
        "elset_uct_candidate": "ss2.data.elset.uct-candidate",
        
        # State Vector Data
        "state_vector_uct_candidate": "ss2.data.state-vector.uct-candidate",
        "state_vector_best_state": "ss2.data.state-vector.best-state",
        
        # Observation Tracks
        "observation_track": "ss2.data.observation-track",
        "observation_track_correlated": "ss2.data.observation-track.correlated",
        
        # Analysis and Services
        "ss2_association_message": "ss2.analysis.association-message",
        "ss2_state_request": "ss2.request.state-recommendation",
        "ss2_state_response": "ss2.response.state-recommendation",
        
        # === SS4 CCDM (Conjunction Detection and Collision Monitoring) ===
        "maneuver_detection": "ss4.maneuver.detection",
        "maneuver_classification": "ss4.maneuver.classification",
        "conjunction_warning": "ss4.conjunction.warning",
        "ccdm_detection": "ss4.ccdm.detection",
        "ccdm_analysis": "ss4.ccdm.analysis",
        "ccdm_correlation": "ss4.ccdm.correlation",
        
        # === SS5 HOSTILITY MONITORING (Official Topics) ===
        
        # Launch Detection and Analysis
        "launch_asat_assessment": "ss5.launch.asat-assessment",
        "launch_coplanar_assessment": "ss5.launch.coplanar-assessment",
        "launch_coplanar_prediction": "ss5.launch.coplanar-prediction",
        "launch_detection": "ss5.launch.detection",
        "launch_intent_assessment": "ss5.launch.intent-assessment",
        "launch_nominal": "ss5.launch.nominal",
        "launch_prediction": "ss5.launch.prediction",
        "launch_trajectory": "ss5.launch.trajectory",
        "launch_weather_check": "ss5.launch.weather-check",
        
        # PEZ-WEZ (Probability of Engagement Zone - Weapon Engagement Zone)
        "pez_wez_analysis_eo": "ss5.pez-wez-analysis.eo",
        "pez_wez_prediction_conjunction": "ss5.pez-wez-prediction.conjunction",
        "pez_wez_prediction_eo": "ss5.pez-wez-prediction.eo",
        "pez_wez_prediction_grappler": "ss5.pez-wez-prediction.grappler",
        "pez_wez_prediction_kkv": "ss5.pez-wez-prediction.kkv",
        "pez_wez_prediction_rf": "ss5.pez-wez-prediction.rf",
        "pez_wez_intent_assessment": "ss5.pez-wez.intent-assessment",
        
        # Reentry and Separation
        "reentry_prediction": "ss5.reentry.prediction",
        "separation_detection": "ss5.separation.detection",
        
        # Service Topics
        "ss5_service_heartbeat": "ss5.service.heartbeat",
        
        # === SS6 RESPONSE RECOMMENDATION (Official Topics) ===
        "response_recommendation_launch": "ss6.response-recommendation.launch",
        "response_recommendation_on_orbit": "ss6.response-recommendation.on-orbit",
        
        # === SS1 TARGET MODELING ===
        "tle_update": "ss1.tle.update",
        "orbital_analysis": "ss1.orbital.analysis",
        
        # === TEST TOPICS (for development/testing) ===
        "test_launch": "test.launch.detection",
        "test_maneuver": "test.maneuver.detection",
        "test_tle": "test.tle.update",
        "test_pez_wez": "test.pez-wez.prediction",
        "test_intent": "test.intent.assessment",
        "test_state": "test.state.estimation",
        "test_weather": "test.weather.data",
        "test_response": "test.response.recommendation",
        "test_general": "test.general.message"
    }
    
    @classmethod
    def get_topic(cls, category: str, use_test: bool = False) -> str:
        """Get topic name with optional test environment routing"""
        if use_test and category in cls.TOPICS:
            test_key = f"test_{category.split('_')[0]}"
            return cls.TOPICS.get(test_key, cls.TOPICS["test_general"])
        return cls.TOPICS.get(category, cls.TOPICS["test_general"])
    
    @classmethod
    def list_topics(cls, test_only: bool = False) -> List[str]:
        """List available topics"""
        if test_only:
            return [topic for key, topic in cls.TOPICS.items() if key.startswith("test_")]
        return list(cls.TOPICS.values())


class SDAKafkaClient:
    """Main SDA Kafka message bus client"""
    
    def __init__(self, credentials: SDAKafkaCredentials, client_id: str = "astroshield"):
        if not KAFKA_AVAILABLE:
            logger.warning("Confluent Kafka not available - using mock client")
            
        self.credentials = credentials
        self.client_id = client_id
        self.producer: Optional[Producer] = None
        self.consumers: Dict[str, Consumer] = {}
        self.admin_client: Optional[AdminClient] = None
        self.message_handlers: Dict[str, List[Callable]] = {}
        self._running = False
        self.test_mode = os.getenv('SDA_KAFKA_TEST_MODE', 'false').lower() == 'true'
        
        # Message size optimization settings
        self.max_message_size = int(os.getenv('SDA_KAFKA_MAX_MESSAGE_SIZE', '1000000'))  # 1MB default
        self.batch_size = int(os.getenv('SDA_KAFKA_BATCH_SIZE', '100'))
        
    def _get_kafka_config(self) -> Dict[str, Any]:
        """Get base Kafka configuration"""
        config = {
            'bootstrap.servers': self.credentials.bootstrap_servers,
            'security.protocol': self.credentials.security_protocol,
            'sasl.mechanism': self.credentials.sasl_mechanism,
            'sasl.username': self.credentials.username,
            'sasl.password': self.credentials.password,
            'client.id': f'{self.client_id}-{os.getpid()}'
        }
        
        if self.credentials.ssl_ca_location:
            config['ssl.ca.location'] = self.credentials.ssl_ca_location
        
        return config
    
    async def initialize(self) -> None:
        """Initialize SDA Kafka connections"""
        if not KAFKA_AVAILABLE:
            logger.warning("Kafka not available - running in mock mode")
            return
            
        try:
            base_config = self._get_kafka_config()
            
            # Initialize producer with optimized settings
            producer_config = {
                **base_config,
                'linger.ms': 10,  # Small batching delay
                'compression.type': 'snappy',  # Fast compression
                'acks': 'all',  # Wait for all replicas
                'retries': 3,
                'max.in.flight.requests.per.connection': 1,  # Ensure ordering
                'message.max.bytes': self.max_message_size
            }
            
            self.producer = Producer(producer_config)
            
            # Initialize admin client
            self.admin_client = AdminClient(base_config)
            
            # Verify connection
            await self._verify_connection()
            
            logger.info(f"SDA Kafka client initialized (test_mode={self.test_mode})")
            
        except Exception as e:
            logger.error(f"Failed to initialize SDA Kafka client: {e}")
            raise
    
    async def _verify_connection(self) -> None:
        """Verify Kafka connection and topic access"""
        if not self.admin_client:
            return
            
        try:
            # Get cluster metadata
            metadata = self.admin_client.list_topics(timeout=10)
            available_topics = set(metadata.topics.keys())
            
            # Check access to SDA topics
            sda_topics = set(SDATopicManager.list_topics(test_only=self.test_mode))
            accessible_topics = sda_topics.intersection(available_topics)
            
            if accessible_topics:
                logger.info(f"Verified access to {len(accessible_topics)} SDA topics")
            else:
                logger.warning("No SDA topics accessible - may need IP whitelisting")
                
        except Exception as e:
            logger.error(f"Connection verification failed: {e}")
            raise
    
    def subscribe(self, topic_category: str, handler: Callable[[SDAMessageSchema], None], 
                  consumer_group: Optional[str] = None) -> None:
        """Subscribe to SDA topic with message handler"""
        topic = SDATopicManager.get_topic(topic_category, use_test=self.test_mode)
        
        if topic not in self.message_handlers:
            self.message_handlers[topic] = []
        self.message_handlers[topic].append(handler)
        
        if not KAFKA_AVAILABLE:
            logger.warning(f"Mock subscription to {topic}")
            return
        
        # Create consumer if not exists
        if topic not in self.consumers:
            group_id = consumer_group or f"{self.client_id}-{topic_category}"
            
            consumer_config = {
                **self._get_kafka_config(),
                'group.id': group_id,
                'auto.offset.reset': 'earliest',
                'enable.auto.commit': True,
                'session.timeout.ms': 30000,
                'max.poll.interval.ms': 300000,
                'fetch.min.bytes': 1,
                'fetch.max.wait.ms': 500
            }
            
            consumer = Consumer(consumer_config)
            consumer.subscribe([topic])
            self.consumers[topic] = consumer
            
            logger.info(f"Subscribed to SDA topic: {topic}")
    
    async def publish(self, topic_category: str, message: SDAMessageSchema, 
                     key: Optional[str] = None) -> bool:
        """Publish message to SDA topic"""
        topic = SDATopicManager.get_topic(topic_category, use_test=self.test_mode)
        
        if not KAFKA_AVAILABLE:
            logger.info(f"Mock publish to {topic}: {message.message_id}")
            return True
            
        if not self.producer:
            raise RuntimeError("SDA Kafka producer not initialized")
        
        try:
            # Serialize message
            message_json = message.json()
            message_size = len(message_json.encode('utf-8'))
            
            # Check message size
            if message_size > self.max_message_size:
                logger.warning(f"Message size ({message_size}) exceeds limit ({self.max_message_size})")
                return False
            
            # Use message_id as key if none provided
            message_key = key or message.message_id
            
            # Produce message
            self.producer.produce(
                topic=topic,
                key=message_key.encode('utf-8'),
                value=message_json.encode('utf-8'),
                on_delivery=self._delivery_callback
            )
            
            # Trigger delivery callbacks
            self.producer.poll(0)
            
            logger.debug(f"Published message to {topic}: {message.message_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish to {topic}: {e}")
            return False
    
    # === SS2 STATE ESTIMATION METHODS ===
    
    async def publish_state_vector(
        self,
        object_id: str,
        position: List[float],
        velocity: List[float],
        epoch: datetime,
        topic_type: str = "best_state",
        covariance: Optional[List[List[float]]] = None,
        coordinate_frame: str = "ITRF",
        data_source: Optional[str] = None,
        quality_metric: Optional[float] = None,
        propagated_from: Optional[datetime] = None
    ) -> bool:
        """Publish SS2 state vector to SDA message bus"""
        try:
            if SDA_SCHEMAS_AVAILABLE:
                # Create state vector message using SDA schema
                from .sda_schemas import SDASchemaFactory
                
                state_vector = SDASchemaFactory.create_ss2_state_vector(
                    object_id=object_id,
                    position=position,
                    velocity=velocity,
                    epoch=epoch,
                    covariance=covariance,
                    coordinate_frame=coordinate_frame,
                    data_source=data_source,
                    quality_metric=quality_metric,
                    propagated_from=propagated_from
                )
                
                # Select appropriate topic
                topic_map = {
                    "best_state": "state_vector_best_state",
                    "uct_candidate": "state_vector_uct_candidate"
                }
                topic_key = topic_map.get(topic_type, "state_vector_best_state")
                topic = SDATopicManager.get_topic(topic_key, use_test=self.test_mode)
                
                # Publish message directly
                message_data = state_vector.json()
                if self.producer:
                    self.producer.produce(topic, value=message_data, on_delivery=self._delivery_callback)
                    self.producer.poll(0)
                
                logger.info(f"Published SS2 state vector for object {object_id} to {topic}")
                return True
            else:
                # Fallback to generic message
                message = SDAMessageSchema(
                    source_system="astroshield",
                    subsystem=SDASubsystem.SS2_STATE_ESTIMATION,
                    message_type="state_vector",
                    data={
                        "object_id": object_id,
                        "position": position,
                        "velocity": velocity,
                        "epoch": epoch.isoformat()
                    }
                )
                return await self.publish("state_vector_best_state", message)
            
        except Exception as e:
            logger.error(f"Failed to publish state vector: {str(e)}")
            return False
    
    async def publish_elset(
        self,
        object_id: str,
        epoch: datetime,
        mean_motion: float,
        eccentricity: float,
        inclination: float,
        arg_of_perigee: float,
        raan: float,
        mean_anomaly: float,
        topic_type: str = "best_state",
        catalog_number: Optional[str] = None,
        classification: str = "U",
        intl_designator: Optional[str] = None,
        line1: Optional[str] = None,
        line2: Optional[str] = None,
        rcs_size: Optional[str] = None,
        object_type: Optional[str] = None,
        data_source: Optional[str] = None,
        quality_metric: Optional[float] = None
    ) -> bool:
        """Publish SS2 elset to SDA message bus"""
        try:
            if SDA_SCHEMAS_AVAILABLE:
                from .sda_schemas import SDASchemaFactory
                
                elset = SDASchemaFactory.create_ss2_elset(
                    object_id=object_id,
                    epoch=epoch,
                    mean_motion=mean_motion,
                    eccentricity=eccentricity,
                    inclination=inclination,
                    arg_of_perigee=arg_of_perigee,
                    raan=raan,
                    mean_anomaly=mean_anomaly,
                    catalog_number=catalog_number,
                    classification=classification,
                    intl_designator=intl_designator,
                    line1=line1,
                    line2=line2,
                    rcs_size=rcs_size,
                    object_type=object_type,
                    data_source=data_source,
                    quality_metric=quality_metric
                )
                
                # Select appropriate topic
                topic_map = {
                    "sgp4": "elset_sgp4",
                    "sgp4_xp": "elset_sgp4_xp",
                    "best_state": "elset_best_state",
                    "uct_candidate": "elset_uct_candidate"
                }
                topic_key = topic_map.get(topic_type, "elset_best_state")
                topic = SDATopicManager.get_topic(topic_key, use_test=self.test_mode)
                
                # Publish message directly
                message_data = elset.json()
                if self.producer:
                    self.producer.produce(topic, value=message_data, on_delivery=self._delivery_callback)
                    self.producer.poll(0)
                
                logger.info(f"Published SS2 elset for object {object_id} to {topic}")
                return True
            else:
                # Fallback to generic message
                message = SDAMessageSchema(
                    source_system="astroshield",
                    subsystem=SDASubsystem.SS2_STATE_ESTIMATION,
                    message_type="elset",
                    data={
                        "object_id": object_id,
                        "mean_motion": mean_motion,
                        "eccentricity": eccentricity,
                        "epoch": epoch.isoformat()
                    }
                )
                return await self.publish("elset_best_state", message)
            
        except Exception as e:
            logger.error(f"Failed to publish elset: {str(e)}")
            return False
    
    # === SS6 RESPONSE RECOMMENDATION METHODS ===
    
    async def publish_response_recommendation(
        self,
        threat_id: str,
        threat_type: str,
        threat_level: str,
        threatened_assets: List[str],
        primary_coa: str,
        priority: str,
        confidence: float,
        recommendation_type: str = "on_orbit",
        response_id: Optional[str] = None,
        alternate_coas: Optional[List[str]] = None,
        tactics_and_procedures: Optional[List[str]] = None,
        time_to_implement: Optional[float] = None,
        effective_window: Optional[List[datetime]] = None,
        rationale: Optional[str] = None,
        risk_assessment: Optional[str] = None,
        analyst_id: Optional[str] = None
    ) -> bool:
        """Publish SS6 response recommendation to SDA message bus"""
        try:
            if SDA_SCHEMAS_AVAILABLE:
                from .sda_schemas import SDASchemaFactory
                
                response = SDASchemaFactory.create_ss6_response_recommendation(
                    threat_id=threat_id,
                    threat_type=threat_type,
                    threat_level=threat_level,
                    threatened_assets=threatened_assets,
                    primary_coa=primary_coa,
                    priority=priority,
                    confidence=confidence,
                    response_id=response_id,
                    alternate_coas=alternate_coas,
                    tactics_and_procedures=tactics_and_procedures,
                    time_to_implement=time_to_implement,
                    effective_window=effective_window,
                    rationale=rationale,
                    risk_assessment=risk_assessment,
                    analyst_id=analyst_id
                )
                
                # Select appropriate topic
                topic_map = {
                    "launch": "response_recommendation_launch",
                    "on_orbit": "response_recommendation_on_orbit"
                }
                topic_key = topic_map.get(recommendation_type, "response_recommendation_on_orbit")
                topic = SDATopicManager.get_topic(topic_key, use_test=self.test_mode)
                
                # Publish message directly
                message_data = response.json()
                if self.producer:
                    self.producer.produce(topic, value=message_data, on_delivery=self._delivery_callback)
                    self.producer.poll(0)
                
                logger.info(f"Published SS6 response recommendation for threat {threat_id} to {topic}")
                return True
            else:
                # Fallback to generic message
                message = SDAMessageSchema(
                    source_system="astroshield",
                    subsystem=SDASubsystem.SS6_THREAT_ASSESSMENT,
                    message_type="response_recommendation",
                    data={
                        "threat_id": threat_id,
                        "threat_type": threat_type,
                        "primary_coa": primary_coa
                    }
                )
                return await self.publish("response_recommendation_on_orbit", message)
            
        except Exception as e:
            logger.error(f"Failed to publish response recommendation: {str(e)}")
            return False
    
    # === SS0 DATA INGESTION METHODS ===
    
    async def publish_weather_data(
        self,
        data_type: str,
        timestamp: datetime,
        weather_type: str = "lightning",
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        altitude: Optional[float] = None,
        resolution: Optional[float] = None,
        valid_time: Optional[datetime] = None,
        forecast_time: Optional[datetime] = None,
        value: Optional[float] = None,
        values: Optional[List[float]] = None,
        units: Optional[str] = None,
        quality_flag: Optional[str] = None,
        confidence: Optional[float] = None
    ) -> bool:
        """Publish SS0 weather data to SDA message bus"""
        try:
            if SDA_SCHEMAS_AVAILABLE:
                from .sda_schemas import SDASchemaFactory
                
                weather_data = SDASchemaFactory.create_ss0_weather_data(
                    data_type=data_type,
                    timestamp=timestamp,
                    latitude=latitude,
                    longitude=longitude,
                    altitude=altitude,
                    resolution=resolution,
                    valid_time=valid_time,
                    forecast_time=forecast_time,
                    value=value,
                    values=values,
                    units=units,
                    quality_flag=quality_flag,
                    confidence=confidence
                )
                
                # Select appropriate weather topic
                weather_topic_map = {
                    "lightning": "weather_lightning",
                    "orbital_density": "weather_orbital_density",
                    "clouds": "weather_clouds",
                    "reflectivity": "weather_reflectivity",
                    "turbulence": "weather_turbulence",
                    "vtec": "weather_vtec",
                    "windshear_low": "weather_windshear_low",
                    "windshear_jet": "weather_windshear_jet",
                    "neutral_density": "weather_neutral_density"
                }
                topic_key = weather_topic_map.get(weather_type, "weather_lightning")
                topic = SDATopicManager.get_topic(topic_key, use_test=self.test_mode)
                
                # Publish message directly
                message_data = weather_data.json()
                if self.producer:
                    self.producer.produce(topic, value=message_data, on_delivery=self._delivery_callback)
                    self.producer.poll(0)
                
                logger.info(f"Published SS0 weather data ({data_type}) to {topic}")
                return True
            else:
                # Fallback to generic message
                message = SDAMessageSchema(
                    source_system="EarthCast",
                    subsystem=SDASubsystem.SS0_DATA_INGESTION,
                    message_type="weather_data",
                    data={
                        "data_type": data_type,
                        "timestamp": timestamp.isoformat(),
                        "value": value
                    }
                )
                return await self.publish("weather_lightning", message)
            
        except Exception as e:
            logger.error(f"Failed to publish weather data: {str(e)}")
            return False
    
    def _delivery_callback(self, err, msg):
        """Delivery report callback"""
        if err is not None:
            logger.error(f"Message delivery failed: {err}")
        else:
            logger.debug(f"Message delivered to {msg.topic()} [{msg.partition()}]")
    
    async def start_consuming(self) -> None:
        """Start consuming messages from subscribed topics"""
        if not KAFKA_AVAILABLE:
            logger.warning("Kafka not available - mock consuming")
            return
            
        if not self.consumers:
            logger.warning("No consumers configured")
            return
        
        self._running = True
        
        # Start consumer tasks
        tasks = []
        for topic, consumer in self.consumers.items():
            task = asyncio.create_task(self._consume_topic(topic, consumer))
            tasks.append(task)
        
        logger.info(f"Started consuming from {len(tasks)} SDA topics")
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Error in consumer tasks: {e}")
        finally:
            self._running = False
    
    async def _consume_topic(self, topic: str, consumer: Consumer) -> None:
        """Consume messages from a specific topic"""
        while self._running:
            try:
                msg = consumer.poll(1.0)
                
                if msg is None:
                    continue
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        logger.error(f"Consumer error: {msg.error()}")
                        continue
                
                # Process message
                await self._process_message(topic, msg)
                
            except Exception as e:
                logger.error(f"Error consuming from {topic}: {e}")
                await asyncio.sleep(1)
    
    async def _process_message(self, topic: str, msg) -> None:
        """Process received message"""
        try:
            # Parse message
            message_data = json.loads(msg.value().decode('utf-8'))
            message = SDAMessageSchema(**message_data)
            
            # Call registered handlers
            handlers = self.message_handlers.get(topic, [])
            for handler in handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(message)
                    else:
                        handler(message)
                except Exception as e:
                    logger.error(f"Handler error for {topic}: {e}")
            
            logger.debug(f"Processed message from {topic}: {message.message_id}")
            
        except Exception as e:
            logger.error(f"Failed to process message from {topic}: {e}")
    
    async def stop(self) -> None:
        """Stop consuming and close connections"""
        self._running = False
        
        if KAFKA_AVAILABLE:
            # Close consumers
            for consumer in self.consumers.values():
                consumer.close()
            
            # Flush producer
            if self.producer:
                self.producer.flush(30)
        
        logger.info("SDA Kafka client stopped")


class AstroShieldSDAIntegration:
    """Integration layer for AstroShield with SDA Kafka message bus"""
    
    def __init__(self, credentials: Optional[SDAKafkaCredentials] = None):
        self.credentials = credentials or SDAKafkaCredentials.from_environment()
        self.kafka_client = SDAKafkaClient(self.credentials, client_id="astroshield")
        self.orbital_intelligence_enabled = True
        
    async def initialize(self) -> None:
        """Initialize SDA integration"""
        await self.kafka_client.initialize()
        
        # Subscribe to relevant SDA topics
        self._setup_subscriptions()
        
        logger.info("AstroShield SDA integration initialized")
    
    def _setup_subscriptions(self) -> None:
        """Setup subscriptions to SDA topics of interest"""
        # TLE and orbital data
        self.kafka_client.subscribe("tle_update", self._handle_tle_update)
        self.kafka_client.subscribe("orbital_analysis", self._handle_orbital_analysis)
        
        # Maneuver detection
        self.kafka_client.subscribe("maneuver_detection", self._handle_maneuver_detection)
        
        # Launch events
        self.kafka_client.subscribe("launch_detection", self._handle_launch_detection)
        
        # CCDM events
        self.kafka_client.subscribe("ccdm_detection", self._handle_ccdm_detection)
    
    async def publish_tle_analysis(self, tle_data: Dict[str, Any], 
                                  analysis_results: Dict[str, Any]) -> bool:
        """Publish TLE analysis results to SDA"""
        message = SDAMessageSchema(
            source_system="astroshield",
            subsystem=SDASubsystem.SS1_TARGET_MODELING,
            message_type="tle_analysis",
            priority=MessagePriority.NORMAL,
            data={
                "tle_data": tle_data,
                "analysis": analysis_results,
                "orbital_intelligence": self._generate_orbital_intelligence(analysis_results)
            }
        )
        
        return await self.kafka_client.publish("orbital_analysis", message)
    
    async def publish_maneuver_detection(self, satellite_id: str, 
                                       maneuver_data: Dict[str, Any]) -> bool:
        """Publish maneuver detection to SDA using official SS4 schema"""
        
        if SDA_SCHEMAS_AVAILABLE:
            # Use official SDA maneuver detected schema
            try:
                # Extract data for SDA schema
                pre_position = maneuver_data.get("pre_position")
                pre_velocity = maneuver_data.get("pre_velocity")
                post_position = maneuver_data.get("post_position")
                post_velocity = maneuver_data.get("post_velocity")
                event_start = maneuver_data.get("event_start_time")
                event_stop = maneuver_data.get("event_stop_time")
                pre_covariance = maneuver_data.get("pre_covariance")
                post_covariance = maneuver_data.get("post_covariance")
                
                # Create SDA-compliant message
                sda_message = SDASchemaFactory.create_maneuver_detected(
                    satellite_id=satellite_id,
                    source="astroshield",
                    pre_position=pre_position,
                    pre_velocity=pre_velocity,
                    post_position=post_position,
                    post_velocity=post_velocity,
                    event_start=event_start,
                    event_stop=event_stop,
                    pre_covariance=pre_covariance,
                    post_covariance=post_covariance
                )
                
                # Convert to JSON for direct publishing
                message_json = sda_message.json() if hasattr(sda_message, 'json') else json.dumps(sda_message.__dict__, default=str)
                
                # Publish directly to maneuver detection topic
                topic = SDATopicManager.get_topic("maneuver_detection", use_test=self.kafka_client.test_mode)
                
                if not KAFKA_AVAILABLE:
                    logger.info(f"Mock publish SDA maneuver detection to {topic}: {satellite_id}")
                    return True
                
                if not self.kafka_client.producer:
                    raise RuntimeError("SDA Kafka producer not initialized")
                
                # Produce message using official SDA schema
                self.kafka_client.producer.produce(
                    topic=topic,
                    key=satellite_id.encode('utf-8'),
                    value=message_json.encode('utf-8'),
                    on_delivery=self.kafka_client._delivery_callback
                )
                
                self.kafka_client.producer.poll(0)
                logger.info(f"Published SDA maneuver detection for satellite {satellite_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to publish SDA maneuver detection: {e}")
                return False
        
        else:
            # Fallback to generic message format
            message = SDAMessageSchema(
                source_system="astroshield",
                subsystem=SDASubsystem.SS2_STATE_ESTIMATION,
                message_type="maneuver_detection",
                priority=MessagePriority.HIGH,
                data={
                    "satellite_id": satellite_id,
                    "maneuver_details": maneuver_data,
                    "detection_confidence": maneuver_data.get("confidence", 0.0),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
            
            return await self.kafka_client.publish("maneuver_detection", message)
    
    async def publish_threat_assessment(self, threat_data: Dict[str, Any]) -> bool:
        """Publish threat assessment to SDA"""
        message = SDAMessageSchema(
            source_system="astroshield",
            subsystem=SDASubsystem.SS5_HOSTILITY_MONITORING,
            message_type="threat_assessment",
            priority=MessagePriority.CRITICAL,
            data=threat_data
        )
        
        return await self.kafka_client.publish("threat_assessment", message)
    
    async def publish_launch_intent_assessment(self, launch_id: str, 
                                             intent_data: Dict[str, Any]) -> bool:
        """Publish launch intent assessment to SDA SS5"""
        
        if SDA_SCHEMAS_AVAILABLE:
            try:
                from .sda_schemas import SDASchemaFactory
                
                # Create SDA-compliant intent assessment message
                sda_message = SDASchemaFactory.create_launch_intent_assessment(
                    launch_id=launch_id,
                    source="astroshield",
                    intent_category=intent_data.get("intent_category"),
                    threat_level=intent_data.get("threat_level"),
                    hostility_score=intent_data.get("hostility_score"),
                    confidence=intent_data.get("confidence"),
                    potential_targets=intent_data.get("potential_targets"),
                    target_type=intent_data.get("target_type"),
                    threat_indicators=intent_data.get("threat_indicators"),
                    asat_capability=intent_data.get("asat_capability"),
                    coplanar_threat=intent_data.get("coplanar_threat"),
                    analyst_id=intent_data.get("analyst_id", "astroshield-ai")
                )
                
                # Publish directly to SS5 launch intent assessment topic
                topic = SDATopicManager.get_topic("launch_intent_assessment", 
                                                use_test=self.kafka_client.test_mode)
                
                if not KAFKA_AVAILABLE:
                    logger.info(f"Mock publish SS5 launch intent assessment to {topic}: {launch_id}")
                    return True
                
                if not self.kafka_client.producer:
                    raise RuntimeError("SDA Kafka producer not initialized")
                
                message_json = sda_message.json() if hasattr(sda_message, 'json') else json.dumps(sda_message.__dict__, default=str)
                
                self.kafka_client.producer.produce(
                    topic=topic,
                    key=launch_id.encode('utf-8'),
                    value=message_json.encode('utf-8'),
                    on_delivery=self.kafka_client._delivery_callback
                )
                
                self.kafka_client.producer.poll(0)
                logger.info(f"Published SS5 launch intent assessment for {launch_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to publish SS5 launch intent assessment: {e}")
                return False
        
        else:
            # Fallback to generic message format
            message = SDAMessageSchema(
                source_system="astroshield",
                subsystem=SDASubsystem.SS5_HOSTILITY_MONITORING,
                message_type="launch_intent_assessment",
                priority=MessagePriority.HIGH,
                data={
                    "launch_id": launch_id,
                    **intent_data
                }
            )
            
            return await self.kafka_client.publish("launch_intent_assessment", message)
    
    async def publish_pez_wez_prediction(self, threat_id: str, 
                                       prediction_data: Dict[str, Any]) -> bool:
        """Publish PEZ-WEZ prediction to SDA SS5"""
        
        if SDA_SCHEMAS_AVAILABLE:
            try:
                from .sda_schemas import SDASchemaFactory
                
                # Create SDA-compliant PEZ-WEZ prediction message
                sda_message = SDASchemaFactory.create_pez_wez_prediction(
                    threat_id=threat_id,
                    source="astroshield",
                    weapon_type=prediction_data.get("weapon_type"),
                    pez_radius=prediction_data.get("pez_radius"),
                    wez_radius=prediction_data.get("wez_radius"),
                    engagement_probability=prediction_data.get("engagement_probability"),
                    time_to_engagement=prediction_data.get("time_to_engagement"),
                    engagement_window=prediction_data.get("engagement_window"),
                    target_assets=prediction_data.get("target_assets"),
                    primary_target=prediction_data.get("primary_target"),
                    validity_period=prediction_data.get("validity_period"),
                    confidence=prediction_data.get("confidence")
                )
                
                # Determine specific PEZ-WEZ topic based on weapon type
                weapon_type = prediction_data.get("weapon_type", "eo")
                topic_key = f"pez_wez_prediction_{weapon_type}"
                
                topic = SDATopicManager.get_topic(topic_key, use_test=self.kafka_client.test_mode)
                
                if not KAFKA_AVAILABLE:
                    logger.info(f"Mock publish SS5 PEZ-WEZ prediction to {topic}: {threat_id}")
                    return True
                
                if not self.kafka_client.producer:
                    raise RuntimeError("SDA Kafka producer not initialized")
                
                message_json = sda_message.json() if hasattr(sda_message, 'json') else json.dumps(sda_message.__dict__, default=str)
                
                self.kafka_client.producer.produce(
                    topic=topic,
                    key=threat_id.encode('utf-8'),
                    value=message_json.encode('utf-8'),
                    on_delivery=self.kafka_client._delivery_callback
                )
                
                self.kafka_client.producer.poll(0)
                logger.info(f"Published SS5 PEZ-WEZ prediction for {threat_id} to {topic}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to publish SS5 PEZ-WEZ prediction: {e}")
                return False
        
        else:
            # Fallback to generic message format
            message = SDAMessageSchema(
                source_system="astroshield",
                subsystem=SDASubsystem.SS5_HOSTILITY_MONITORING,
                message_type="pez_wez_prediction",
                priority=MessagePriority.HIGH,
                data={
                    "threat_id": threat_id,
                    **prediction_data
                }
            )
            
            weapon_type = prediction_data.get("weapon_type", "eo")
            topic_key = f"pez_wez_prediction_{weapon_type}"
            return await self.kafka_client.publish(topic_key, message)
    
    async def publish_asat_assessment(self, threat_id: str, 
                                    assessment_data: Dict[str, Any]) -> bool:
        """Publish ASAT assessment to SDA SS5"""
        
        if SDA_SCHEMAS_AVAILABLE:
            try:
                from .sda_schemas import SDASchemaFactory
                
                # Create SDA-compliant ASAT assessment message
                sda_message = SDASchemaFactory.create_asat_assessment(
                    threat_id=threat_id,
                    source="astroshield",
                    asat_type=assessment_data.get("asat_type"),
                    asat_capability=assessment_data.get("asat_capability"),
                    threat_level=assessment_data.get("threat_level"),
                    targeted_assets=assessment_data.get("targeted_assets"),
                    orbit_regimes_threatened=assessment_data.get("orbit_regimes_threatened"),
                    intercept_capability=assessment_data.get("intercept_capability"),
                    max_reach_altitude=assessment_data.get("max_reach_altitude"),
                    effective_range=assessment_data.get("effective_range"),
                    launch_to_impact=assessment_data.get("launch_to_impact"),
                    confidence=assessment_data.get("confidence"),
                    intelligence_sources=assessment_data.get("intelligence_sources")
                )
                
                # Publish to SS5 launch ASAT assessment topic
                topic = SDATopicManager.get_topic("launch_asat_assessment", 
                                                use_test=self.kafka_client.test_mode)
                
                if not KAFKA_AVAILABLE:
                    logger.info(f"Mock publish SS5 ASAT assessment to {topic}: {threat_id}")
                    return True
                
                if not self.kafka_client.producer:
                    raise RuntimeError("SDA Kafka producer not initialized")
                
                message_json = sda_message.json() if hasattr(sda_message, 'json') else json.dumps(sda_message.__dict__, default=str)
                
                self.kafka_client.producer.produce(
                    topic=topic,
                    key=threat_id.encode('utf-8'),
                    value=message_json.encode('utf-8'),
                    on_delivery=self.kafka_client._delivery_callback
                )
                
                self.kafka_client.producer.poll(0)
                logger.info(f"Published SS5 ASAT assessment for {threat_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to publish SS5 ASAT assessment: {e}")
                return False
        
        else:
            # Fallback to generic message format
            message = SDAMessageSchema(
                source_system="astroshield",
                subsystem=SDASubsystem.SS5_HOSTILITY_MONITORING,
                message_type="asat_assessment",
                priority=MessagePriority.CRITICAL,
                data={
                    "threat_id": threat_id,
                    **assessment_data
                }
            )
            
            return await self.kafka_client.publish("launch_asat_assessment", message)
    
    def _generate_orbital_intelligence(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate orbital intelligence summary from analysis"""
        if not self.orbital_intelligence_enabled:
            return {}
        
        return {
            "orbital_accuracy": analysis_results.get("accuracy_score", 0.0),
            "risk_assessment": analysis_results.get("risk_level", "unknown"),
            "satellite_recognition": analysis_results.get("satellite_type", "unknown"),
            "maneuver_probability": analysis_results.get("maneuver_likelihood", 0.0),
            "threat_level": analysis_results.get("threat_assessment", "low"),
            "recommendation": analysis_results.get("recommended_action", "monitor")
        }
    
    # Message handlers
    async def _handle_tle_update(self, message: SDAMessageSchema) -> None:
        """Handle incoming TLE updates from SDA"""
        logger.info(f"Received TLE update: {message.message_id}")
        # Process with AstroShield TLE analysis
    
    async def _handle_orbital_analysis(self, message: SDAMessageSchema) -> None:
        """Handle orbital analysis from other systems"""
        logger.info(f"Received orbital analysis: {message.message_id}")
    
    async def _handle_maneuver_detection(self, message: SDAMessageSchema) -> None:
        """Handle maneuver detection from SDA"""
        logger.info(f"Received maneuver detection: {message.message_id}")
    
    async def _handle_launch_detection(self, message: SDAMessageSchema) -> None:
        """Handle launch detection from SDA"""
        logger.info(f"Received launch detection: {message.message_id}")
    
    async def _handle_ccdm_detection(self, message: SDAMessageSchema) -> None:
        """Handle CCDM detection from SDA"""
        logger.info(f"Received CCDM detection: {message.message_id}")
    
    async def start(self) -> None:
        """Start SDA message bus integration"""
        await self.kafka_client.start_consuming()
    
    async def stop(self) -> None:
        """Stop SDA integration"""
        await self.kafka_client.stop() 