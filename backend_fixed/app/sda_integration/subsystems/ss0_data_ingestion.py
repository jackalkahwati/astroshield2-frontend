"""
Subsystem 0: Data Ingestion & Sensors
All observational data and sensor inputs into Welders Arc
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
from enum import Enum
import numpy as np
from collections import defaultdict

from ..kafka.kafka_client import (
    WeldersArcKafkaClient,
    WeldersArcMessage,
    KafkaTopics,
    SubsystemID,
    EventType
)
from ..udl.udl_client import UDLClient, CollectionRequest, SensorObservation

logger = logging.getLogger(__name__)


class SensorType(Enum):
    """Types of sensors in the SDA network"""
    OPTICAL = "optical"
    RADAR = "radar"
    RF = "rf"
    INFRARED = "infrared"
    LASER = "laser"
    MULTISPECTRAL = "multispectral"


class DataQuality(Enum):
    """Data quality levels"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    DEGRADED = "degraded"
    FAILED = "failed"


@dataclass
class SensorStatus:
    """Status of a sensor"""
    sensor_id: str
    sensor_type: SensorType
    location: Dict[str, float]  # lat, lon, alt
    status: str
    last_heartbeat: datetime
    data_quality: DataQuality
    collection_rate: float  # observations per minute
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FusedObservation:
    """Multi-phenomenology fused observation"""
    object_id: str
    timestamp: datetime
    position: np.ndarray  # ECI coordinates
    velocity: Optional[np.ndarray] = None
    covariance: Optional[np.ndarray] = None
    sensor_ids: List[str] = field(default_factory=list)
    phenomenologies: List[SensorType] = field(default_factory=list)
    confidence: float = 0.0
    raw_observations: List[SensorObservation] = field(default_factory=list)


class SensorFusion:
    """Multi-phenomenology sensor fusion engine"""
    
    def __init__(self):
        self.fusion_buffer: Dict[str, List[SensorObservation]] = defaultdict(list)
        self.fusion_window = timedelta(seconds=30)  # 30-second fusion window
        
    async def add_observation(self, observation: SensorObservation) -> Optional[FusedObservation]:
        """Add observation and check if fusion is possible"""
        object_id = observation.target_id or f"UCT-{observation.observation_id}"
        self.fusion_buffer[object_id].append(observation)
        
        # Clean old observations
        cutoff_time = datetime.utcnow() - self.fusion_window
        self.fusion_buffer[object_id] = [
            obs for obs in self.fusion_buffer[object_id]
            if obs.timestamp > cutoff_time
        ]
        
        # Check if we have enough observations for fusion
        if len(self.fusion_buffer[object_id]) >= 2:
            return await self._fuse_observations(object_id)
            
        return None
        
    async def _fuse_observations(self, object_id: str) -> FusedObservation:
        """Fuse multiple observations into a single estimate"""
        observations = self.fusion_buffer[object_id]
        
        # Group by phenomenology
        phenom_groups = defaultdict(list)
        for obs in observations:
            sensor_type = self._classify_sensor_type(obs)
            phenom_groups[sensor_type].append(obs)
            
        # Weighted fusion based on sensor type accuracy
        weights = {
            SensorType.OPTICAL: 1.0,
            SensorType.RADAR: 0.9,
            SensorType.RF: 0.7,
            SensorType.INFRARED: 0.8,
            SensorType.LASER: 0.95,
            SensorType.MULTISPECTRAL: 0.85
        }
        
        # Calculate fused position
        total_weight = 0
        fused_position = np.zeros(3)
        sensor_ids = []
        phenomenologies = []
        
        for sensor_type, obs_list in phenom_groups.items():
            weight = weights.get(sensor_type, 0.5)
            phenomenologies.append(sensor_type)
            
            for obs in obs_list:
                # Convert observation to ECI if needed
                pos_eci = self._convert_to_eci(obs)
                fused_position += pos_eci * weight
                total_weight += weight
                sensor_ids.append(obs.sensor_id)
                
        if total_weight > 0:
            fused_position /= total_weight
            
        # Calculate confidence based on number of sensors and agreement
        confidence = self._calculate_fusion_confidence(observations, fused_position)
        
        # Estimate velocity if we have temporal data
        velocity = None
        if len(observations) > 2:
            velocity = self._estimate_velocity(observations, fused_position)
            
        return FusedObservation(
            object_id=object_id,
            timestamp=datetime.utcnow(),
            position=fused_position,
            velocity=velocity,
            sensor_ids=list(set(sensor_ids)),
            phenomenologies=list(set(phenomenologies)),
            confidence=confidence,
            raw_observations=observations
        )
        
    def _classify_sensor_type(self, observation: SensorObservation) -> SensorType:
        """Classify sensor type from observation metadata"""
        obs_type = observation.observation_type.lower()
        
        if "optical" in obs_type or "telescope" in obs_type:
            return SensorType.OPTICAL
        elif "radar" in obs_type:
            return SensorType.RADAR
        elif "rf" in obs_type or "radio" in obs_type:
            return SensorType.RF
        elif "ir" in obs_type or "infrared" in obs_type:
            return SensorType.INFRARED
        elif "laser" in obs_type or "lidar" in obs_type:
            return SensorType.LASER
        else:
            return SensorType.MULTISPECTRAL
            
    def _convert_to_eci(self, observation: SensorObservation) -> np.ndarray:
        """Convert observation to ECI coordinates"""
        # Simplified conversion - in production would use proper coordinate transforms
        if observation.position and all(k in observation.position for k in ["x", "y", "z"]):
            return np.array([
                observation.position["x"],
                observation.position["y"],
                observation.position["z"]
            ])
        elif observation.position and all(k in observation.position for k in ["lat", "lon", "alt"]):
            # Convert LLA to ECI (simplified)
            Re = 6378.137  # Earth radius km
            lat = np.radians(observation.position["lat"])
            lon = np.radians(observation.position["lon"])
            alt = observation.position["alt"]
            
            r = Re + alt
            x = r * np.cos(lat) * np.cos(lon)
            y = r * np.cos(lat) * np.sin(lon)
            z = r * np.sin(lat)
            
            return np.array([x, y, z])
        else:
            # Default position if conversion fails
            return np.array([7000.0, 0.0, 0.0])
            
    def _calculate_fusion_confidence(self, observations: List[SensorObservation], 
                                   fused_position: np.ndarray) -> float:
        """Calculate confidence score for fused observation"""
        if len(observations) < 2:
            return 0.5
            
        # Calculate variance in observations
        positions = [self._convert_to_eci(obs) for obs in observations]
        distances = [np.linalg.norm(pos - fused_position) for pos in positions]
        
        # Lower variance = higher confidence
        variance = np.var(distances)
        max_variance = 100.0  # km^2
        
        confidence = max(0.0, 1.0 - (variance / max_variance))
        
        # Boost confidence for multiple phenomenologies
        unique_sensors = len(set(obs.sensor_id for obs in observations))
        confidence = min(1.0, confidence + 0.1 * (unique_sensors - 1))
        
        return confidence
        
    def _estimate_velocity(self, observations: List[SensorObservation], 
                          position: np.ndarray) -> np.ndarray:
        """Estimate velocity from temporal observations"""
        # Sort by timestamp
        sorted_obs = sorted(observations, key=lambda x: x.timestamp)
        
        if len(sorted_obs) < 2:
            return None
            
        # Simple finite difference
        dt = (sorted_obs[-1].timestamp - sorted_obs[0].timestamp).total_seconds()
        if dt > 0:
            pos1 = self._convert_to_eci(sorted_obs[0])
            pos2 = self._convert_to_eci(sorted_obs[-1])
            velocity = (pos2 - pos1) / dt
            return velocity
            
        return None


class DataIngestionSubsystem:
    """Main data ingestion and sensor management subsystem"""
    
    def __init__(self, kafka_client: WeldersArcKafkaClient, udl_client: UDLClient):
        self.kafka_client = kafka_client
        self.udl_client = udl_client
        self.sensor_fusion = SensorFusion()
        self.sensors: Dict[str, SensorStatus] = {}
        self.active_collections: Dict[str, CollectionRequest] = {}
        self.heartbeat_interval = 30  # seconds
        self.collection_queue: asyncio.Queue = asyncio.Queue()
        
    async def initialize(self):
        """Initialize data ingestion subsystem"""
        # Subscribe to sensor data topics
        self.kafka_client.subscribe(
            KafkaTopics.SENSOR_OBSERVATIONS,
            self._handle_sensor_observation
        )
        
        self.kafka_client.subscribe(
            KafkaTopics.SENSOR_STATUS,
            self._handle_sensor_status
        )
        
        # Start background tasks
        asyncio.create_task(self._heartbeat_monitor())
        asyncio.create_task(self._collection_processor())
        asyncio.create_task(self._udl_poller())
        
        logger.info("Data ingestion subsystem initialized")
        
    async def _handle_sensor_observation(self, message: WeldersArcMessage):
        """Handle incoming sensor observation"""
        try:
            # Convert to SensorObservation
            obs_data = message.data
            observation = SensorObservation(
                observation_id=obs_data["observation_id"],
                sensor_id=obs_data["sensor_id"],
                target_id=obs_data.get("target_id"),
                timestamp=datetime.fromisoformat(obs_data["timestamp"]),
                observation_type=obs_data["observation_type"],
                position=obs_data.get("position"),
                velocity=obs_data.get("velocity"),
                brightness=obs_data.get("brightness"),
                snr=obs_data.get("snr"),
                metadata=obs_data.get("metadata", {})
            )
            
            # Update sensor metrics
            if observation.sensor_id in self.sensors:
                sensor = self.sensors[observation.sensor_id]
                sensor.last_heartbeat = datetime.utcnow()
                sensor.collection_rate = self._update_collection_rate(sensor)
                
            # Attempt sensor fusion
            fused = await self.sensor_fusion.add_observation(observation)
            
            if fused:
                # Publish fused observation
                await self._publish_fused_observation(fused)
                
            # Forward raw observation
            await self._forward_raw_observation(observation)
            
        except Exception as e:
            logger.error(f"Error handling sensor observation: {e}")
            
    async def _handle_sensor_status(self, message: WeldersArcMessage):
        """Handle sensor status update"""
        status_data = message.data
        sensor_id = status_data["sensor_id"]
        
        sensor_status = SensorStatus(
            sensor_id=sensor_id,
            sensor_type=SensorType(status_data["sensor_type"]),
            location=status_data["location"],
            status=status_data["status"],
            last_heartbeat=datetime.utcnow(),
            data_quality=DataQuality(status_data.get("data_quality", "medium")),
            collection_rate=status_data.get("collection_rate", 0.0),
            metadata=status_data.get("metadata", {})
        )
        
        self.sensors[sensor_id] = sensor_status
        
        # Check for sensor failures
        if sensor_status.status == "failed" or sensor_status.data_quality == DataQuality.FAILED:
            await self._handle_sensor_failure(sensor_id)
            
    async def _heartbeat_monitor(self):
        """Monitor sensor heartbeats"""
        while True:
            try:
                current_time = datetime.utcnow()
                failed_sensors = []
                
                for sensor_id, sensor in self.sensors.items():
                    time_since_heartbeat = (current_time - sensor.last_heartbeat).total_seconds()
                    
                    if time_since_heartbeat > self.heartbeat_interval * 3:
                        # Sensor is non-responsive
                        failed_sensors.append(sensor_id)
                        sensor.status = "failed"
                        sensor.data_quality = DataQuality.FAILED
                        
                    elif time_since_heartbeat > self.heartbeat_interval * 2:
                        # Sensor is degraded
                        sensor.data_quality = DataQuality.DEGRADED
                        
                # Handle failures
                for sensor_id in failed_sensors:
                    await self._handle_sensor_failure(sensor_id)
                    
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in heartbeat monitor: {e}")
                await asyncio.sleep(5)
                
    async def _collection_processor(self):
        """Process collection requests"""
        while True:
            try:
                # Get next collection request
                request = await self.collection_queue.get()
                
                # Submit to UDL
                response = await self.udl_client.submit_collection_request(request)
                
                if response:
                    self.active_collections[request.request_id] = request
                    logger.info(f"Collection request {request.request_id} submitted")
                    
                    # Publish collection status
                    message = WeldersArcMessage(
                        message_id=f"collection-{request.request_id}",
                        timestamp=datetime.utcnow(),
                        subsystem=SubsystemID.SS0_INGESTION,
                        event_type="collection_submitted",
                        data={
                            "request_id": request.request_id,
                            "sensor_id": request.sensor_id,
                            "target_id": request.target_id,
                            "status": response.status,
                            "scheduled_time": response.scheduled_time.isoformat()
                        }
                    )
                    
                    await self.kafka_client.publish(KafkaTopics.COLLECT_STATUS, message)
                    
            except Exception as e:
                logger.error(f"Error processing collection: {e}")
                await asyncio.sleep(1)
                
    async def _udl_poller(self):
        """Poll UDL for new observations"""
        while True:
            try:
                # Get latest observations from all sensors
                for sensor_id in self.sensors.keys():
                    observations = await self.udl_client.get_sensor_observations(
                        sensor_id=sensor_id,
                        limit=50
                    )
                    
                    for obs in observations:
                        # Convert to Kafka message
                        message = WeldersArcMessage(
                            message_id=f"obs-{obs.observation_id}",
                            timestamp=obs.timestamp,
                            subsystem=SubsystemID.SS0_INGESTION,
                            event_type="sensor_observation",
                            data={
                                "observation_id": obs.observation_id,
                                "sensor_id": obs.sensor_id,
                                "target_id": obs.target_id,
                                "timestamp": obs.timestamp.isoformat(),
                                "observation_type": obs.observation_type,
                                "position": obs.position,
                                "velocity": obs.velocity,
                                "brightness": obs.brightness,
                                "snr": obs.snr,
                                "metadata": obs.metadata
                            }
                        )
                        
                        await self.kafka_client.publish(
                            KafkaTopics.SENSOR_OBSERVATIONS,
                            message
                        )
                        
                await asyncio.sleep(10)  # Poll every 10 seconds
                
            except Exception as e:
                logger.error(f"Error polling UDL: {e}")
                await asyncio.sleep(30)
                
    async def _publish_fused_observation(self, fused: FusedObservation):
        """Publish fused multi-phenomenology observation"""
        message = WeldersArcMessage(
            message_id=f"fused-{fused.object_id}-{datetime.utcnow().timestamp()}",
            timestamp=fused.timestamp,
            subsystem=SubsystemID.SS0_INGESTION,
            event_type="fused_observation",
            data={
                "object_id": fused.object_id,
                "position": fused.position.tolist(),
                "velocity": fused.velocity.tolist() if fused.velocity is not None else None,
                "confidence": fused.confidence,
                "sensor_ids": fused.sensor_ids,
                "phenomenologies": [p.value for p in fused.phenomenologies],
                "observation_count": len(fused.raw_observations)
            }
        )
        
        await self.kafka_client.publish(KafkaTopics.FUSED_OBSERVATIONS, message)
        
        # Also publish as UCT if confidence is high and no known object
        if fused.confidence > 0.8 and fused.object_id.startswith("UCT"):
            uct_message = WeldersArcMessage(
                message_id=f"uct-{fused.object_id}-{datetime.utcnow().timestamp()}",
                timestamp=fused.timestamp,
                subsystem=SubsystemID.SS0_INGESTION,
                event_type="uct_detection",
                data={
                    "track_id": fused.object_id,
                    "sensor_id": "fusion-engine",
                    "position": fused.position.tolist(),
                    "velocity": fused.velocity.tolist() if fused.velocity is not None else None,
                    "timestamp": fused.timestamp.isoformat(),
                    "metadata": {
                        "confidence": fused.confidence,
                        "phenomenologies": [p.value for p in fused.phenomenologies]
                    }
                }
            )
            
            await self.kafka_client.publish(KafkaTopics.UCT_TRACKS, uct_message)
            
    async def _forward_raw_observation(self, observation: SensorObservation):
        """Forward raw observation to appropriate subsystems"""
        # Classify observation type
        sensor_type = self.sensor_fusion._classify_sensor_type(observation)
        
        # Route based on type
        if sensor_type == SensorType.RF:
            # RF observations go to hostility monitoring
            message = WeldersArcMessage(
                message_id=f"rf-{observation.observation_id}",
                timestamp=observation.timestamp,
                subsystem=SubsystemID.SS0_INGESTION,
                event_type=EventType.RF_EMISSION,
                data={
                    "object_id": observation.target_id or "unknown",
                    "sensor_id": observation.sensor_id,
                    "frequency": observation.metadata.get("frequency"),
                    "power": observation.metadata.get("power"),
                    "modulation": observation.metadata.get("modulation"),
                    "timestamp": observation.timestamp.isoformat()
                }
            )
            
            await self.kafka_client.publish(KafkaTopics.EVENT_RF_DETECTION, message)
            
    async def _handle_sensor_failure(self, sensor_id: str):
        """Handle sensor failure"""
        logger.error(f"Sensor {sensor_id} has failed")
        
        # Publish alert
        message = WeldersArcMessage(
            message_id=f"sensor-failure-{sensor_id}-{datetime.utcnow().timestamp()}",
            timestamp=datetime.utcnow(),
            subsystem=SubsystemID.SS0_INGESTION,
            event_type="sensor_failure",
            data={
                "sensor_id": sensor_id,
                "sensor_type": self.sensors[sensor_id].sensor_type.value,
                "location": self.sensors[sensor_id].location,
                "last_heartbeat": self.sensors[sensor_id].last_heartbeat.isoformat()
            }
        )
        
        await self.kafka_client.publish(KafkaTopics.ALERT_OPERATOR, message)
        
        # Attempt to find alternate sensors
        await self._find_alternate_sensors(sensor_id)
        
    async def _find_alternate_sensors(self, failed_sensor_id: str):
        """Find alternate sensors to cover gap"""
        failed_sensor = self.sensors.get(failed_sensor_id)
        if not failed_sensor:
            return
            
        # Find sensors of same type in similar location
        alternates = []
        for sensor_id, sensor in self.sensors.items():
            if (sensor_id != failed_sensor_id and 
                sensor.sensor_type == failed_sensor.sensor_type and
                sensor.status != "failed"):
                # Check if coverage overlaps (simplified)
                alternates.append(sensor_id)
                
        if alternates:
            logger.info(f"Found {len(alternates)} alternate sensors for {failed_sensor_id}")
            
            # Increase collection rate on alternates
            for alt_id in alternates:
                await self._increase_collection_rate(alt_id)
                
    def _update_collection_rate(self, sensor: SensorStatus) -> float:
        """Update collection rate metric"""
        # Simple exponential moving average
        alpha = 0.1
        current_rate = 1.0  # Placeholder - would calculate from actual observations
        
        return alpha * current_rate + (1 - alpha) * sensor.collection_rate
        
    async def _increase_collection_rate(self, sensor_id: str):
        """Request increased collection rate from sensor"""
        message = WeldersArcMessage(
            message_id=f"rate-increase-{sensor_id}-{datetime.utcnow().timestamp()}",
            timestamp=datetime.utcnow(),
            subsystem=SubsystemID.SS0_INGESTION,
            event_type="collection_rate_change",
            data={
                "sensor_id": sensor_id,
                "requested_rate": 2.0,  # Double the rate
                "reason": "coverage_gap",
                "duration_minutes": 60
            }
        )
        
        await self.kafka_client.publish(KafkaTopics.SENSOR_CONTROL, message)
        
    async def submit_collection_request(
        self,
        sensor_id: str,
        target_id: str,
        collection_type: str,
        priority: int = 5
    ):
        """Submit a collection request"""
        request = CollectionRequest(
            request_id=f"req-{datetime.utcnow().timestamp()}",
            sensor_id=sensor_id,
            target_id=target_id,
            collection_type=collection_type,
            priority=priority,
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow() + timedelta(hours=1)
        )
        
        await self.collection_queue.put(request)
        
    def get_sensor_coverage(self) -> Dict[str, Any]:
        """Get current sensor coverage statistics"""
        total_sensors = len(self.sensors)
        active_sensors = sum(1 for s in self.sensors.values() if s.status != "failed")
        
        coverage_by_type = defaultdict(int)
        for sensor in self.sensors.values():
            if sensor.status != "failed":
                coverage_by_type[sensor.sensor_type.value] += 1
                
        return {
            "total_sensors": total_sensors,
            "active_sensors": active_sensors,
            "coverage_percentage": (active_sensors / total_sensors * 100) if total_sensors > 0 else 0,
            "coverage_by_type": dict(coverage_by_type),
            "failed_sensors": [s for s, status in self.sensors.items() if status.status == "failed"]
        } 