"""
Subsystem 1: Target Modeling & Characterization
High-fidelity models of the behavior and intent of space objects
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
from enum import Enum
import numpy as np
from collections import defaultdict, deque
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from ..kafka.kafka_client import (
    WeldersArcKafkaClient,
    WeldersArcMessage,
    KafkaTopics,
    SubsystemID,
    EventType
)
from .ss2_state_estimation import OrbitDetermination, StateVector

logger = logging.getLogger(__name__)


class ObjectType(Enum):
    """Types of space objects"""
    ACTIVE_SATELLITE = "active_satellite"
    DEBRIS = "debris"
    ROCKET_BODY = "rocket_body"
    UNKNOWN = "unknown"
    WEAPON = "weapon"
    DECOY = "decoy"


class BehaviorPattern(Enum):
    """Identified behavior patterns"""
    STATION_KEEPING = "station_keeping"
    ORBIT_RAISING = "orbit_raising"
    ORBIT_LOWERING = "orbit_lowering"
    PLANE_CHANGE = "plane_change"
    RENDEZVOUS = "rendezvous"
    PROXIMITY_OPS = "proximity_ops"
    DEORBITING = "deorbiting"
    ANOMALOUS = "anomalous"
    EVASIVE = "evasive"
    AGGRESSIVE = "aggressive"


@dataclass
class ManeuverCapability:
    """Estimated maneuver capabilities"""
    delta_v_budget: float  # m/s
    thrust_level: float  # N
    specific_impulse: float  # s
    fuel_remaining: float  # kg (estimated)
    max_acceleration: float  # m/s²
    maneuver_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TargetModel:
    """Comprehensive model of a space object"""
    object_id: str
    object_type: ObjectType
    last_updated: datetime
    
    # Physical properties
    mass: float  # kg
    cross_sectional_area: float  # m²
    ballistic_coefficient: float
    area_to_mass_ratio: float
    
    # Orbital characteristics
    mean_motion: float  # rev/day
    eccentricity: float
    inclination: float  # degrees
    semi_major_axis: float  # km
    
    # Behavioral model
    behavior_pattern: BehaviorPattern
    pattern_confidence: float
    maneuver_capability: Optional[ManeuverCapability]
    
    # Pattern of Life
    typical_behaviors: Dict[str, float] = field(default_factory=dict)
    anomaly_score: float = 0.0
    
    # Intent indicators
    threat_indicators: Dict[str, float] = field(default_factory=dict)
    cooperative_behavior: bool = True
    
    # Historical data
    state_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    maneuver_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    data_sources: Set[str] = field(default_factory=set)
    confidence: float = 0.0


class OrbitCharacterizer:
    """Characterize orbital regimes and behaviors"""
    
    @staticmethod
    def characterize_orbit(state: StateVector) -> Dict[str, Any]:
        """Characterize the orbital regime"""
        # Calculate orbital elements
        r = np.linalg.norm(state.position)
        v = np.linalg.norm(state.velocity)
        
        # Specific energy
        mu = 398600.4418  # Earth's gravitational parameter km³/s²
        epsilon = v**2 / 2 - mu / r
        
        # Semi-major axis
        a = -mu / (2 * epsilon) if epsilon < 0 else float('inf')
        
        # Eccentricity vector
        h = np.cross(state.position, state.velocity)
        e_vec = np.cross(state.velocity, h) / mu - state.position / r
        e = np.linalg.norm(e_vec)
        
        # Inclination
        i = np.degrees(np.arccos(h[2] / np.linalg.norm(h)))
        
        # Classify orbit type
        altitude = r - 6378.137  # km above Earth's surface
        
        orbit_type = "unknown"
        if altitude < 2000:
            orbit_type = "LEO"
        elif 2000 <= altitude < 35786:
            orbit_type = "MEO"
        elif 35786 - 100 < altitude < 35786 + 100 and i < 5:
            orbit_type = "GEO"
        elif e > 0.7:
            orbit_type = "HEO"
            
        return {
            "semi_major_axis": a,
            "eccentricity": e,
            "inclination": i,
            "altitude": altitude,
            "orbit_type": orbit_type,
            "mean_motion": 86400 * np.sqrt(mu / a**3) if a > 0 else 0  # rev/day
        }


class BehaviorAnalyzer:
    """Analyze patterns of behavior"""
    
    def __init__(self):
        self.pattern_detectors = {
            BehaviorPattern.STATION_KEEPING: self._detect_station_keeping,
            BehaviorPattern.ORBIT_RAISING: self._detect_orbit_change,
            BehaviorPattern.RENDEZVOUS: self._detect_rendezvous,
            BehaviorPattern.PROXIMITY_OPS: self._detect_proximity_ops,
            BehaviorPattern.EVASIVE: self._detect_evasive,
            BehaviorPattern.AGGRESSIVE: self._detect_aggressive
        }
        
    async def analyze_behavior(self, model: TargetModel) -> Tuple[BehaviorPattern, float]:
        """Analyze behavior pattern from state history"""
        if len(model.state_history) < 10:
            return BehaviorPattern.STATION_KEEPING, 0.5
            
        # Run all pattern detectors
        pattern_scores = {}
        for pattern, detector in self.pattern_detectors.items():
            score = await detector(model)
            if score > 0:
                pattern_scores[pattern] = score
                
        # Select highest scoring pattern
        if pattern_scores:
            best_pattern = max(pattern_scores, key=pattern_scores.get)
            confidence = pattern_scores[best_pattern]
            return best_pattern, confidence
        else:
            return BehaviorPattern.ANOMALOUS, 0.8
            
    async def _detect_station_keeping(self, model: TargetModel) -> float:
        """Detect station keeping behavior"""
        if len(model.state_history) < 20:
            return 0.0
            
        # Calculate position variance
        positions = [state["position"] for state in model.state_history]
        position_array = np.array(positions)
        variance = np.var(position_array, axis=0)
        
        # Low variance indicates station keeping
        if np.all(variance < 10.0):  # km²
            return 0.9
        elif np.all(variance < 50.0):
            return 0.6
        else:
            return 0.0
            
    async def _detect_orbit_change(self, model: TargetModel) -> float:
        """Detect orbit raising/lowering"""
        if len(model.state_history) < 50:
            return 0.0
            
        # Track semi-major axis changes
        sma_values = []
        for state in model.state_history:
            orbit_params = OrbitCharacterizer.characterize_orbit(
                StateVector(
                    position=np.array(state["position"]),
                    velocity=np.array(state["velocity"]),
                    timestamp=state["timestamp"]
                )
            )
            sma_values.append(orbit_params["semi_major_axis"])
            
        # Check for consistent trend
        if len(sma_values) > 10:
            trend = np.polyfit(range(len(sma_values)), sma_values, 1)[0]
            
            if abs(trend) > 0.1:  # km per observation
                return 0.8
                
        return 0.0
        
    async def _detect_rendezvous(self, model: TargetModel) -> float:
        """Detect rendezvous operations"""
        # Check recent maneuvers
        recent_maneuvers = [m for m in model.maneuver_history 
                           if (datetime.utcnow() - m["timestamp"]).days < 7]
        
        if len(recent_maneuvers) > 3:
            # Multiple maneuvers suggest active operations
            return 0.7
            
        return 0.0
        
    async def _detect_proximity_ops(self, model: TargetModel) -> float:
        """Detect proximity operations"""
        # Would check distances to other objects
        # Simplified for now
        if model.object_type == ObjectType.ACTIVE_SATELLITE:
            if any("proximity" in indicator for indicator in model.threat_indicators):
                return 0.8
                
        return 0.0
        
    async def _detect_evasive(self, model: TargetModel) -> float:
        """Detect evasive maneuvers"""
        if len(model.maneuver_history) < 2:
            return 0.0
            
        # Rapid, unpredictable maneuvers
        recent = [m for m in model.maneuver_history[-5:]
                 if (datetime.utcnow() - m["timestamp"]).hours < 24]
        
        if len(recent) >= 2:
            # Check for direction changes
            directions = [m.get("direction", [0, 0, 0]) for m in recent]
            
            # Calculate angle between maneuvers
            if len(directions) >= 2:
                angle = np.arccos(np.dot(directions[0], directions[1]) / 
                                (np.linalg.norm(directions[0]) * np.linalg.norm(directions[1])))
                
                if angle > np.pi / 2:  # > 90 degrees
                    return 0.9
                    
        return 0.0
        
    async def _detect_aggressive(self, model: TargetModel) -> float:
        """Detect aggressive behavior"""
        threat_score = sum(model.threat_indicators.values()) / len(model.threat_indicators) \
                      if model.threat_indicators else 0
                      
        if threat_score > 0.7:
            return 0.85
        elif threat_score > 0.5:
            return 0.6
            
        return 0.0


class ManeuverPredictor:
    """Predict future maneuvers"""
    
    def __init__(self):
        self.prediction_horizon = timedelta(hours=24)
        
    async def predict_maneuvers(self, model: TargetModel) -> List[Dict[str, Any]]:
        """Predict likely future maneuvers"""
        predictions = []
        
        # Analyze historical patterns
        if len(model.maneuver_history) < 3:
            return predictions
            
        # Find periodic patterns
        maneuver_times = [m["timestamp"] for m in model.maneuver_history]
        if len(maneuver_times) > 5:
            # Calculate intervals
            intervals = []
            for i in range(1, len(maneuver_times)):
                interval = (maneuver_times[i] - maneuver_times[i-1]).total_seconds()
                intervals.append(interval)
                
            # Check for periodicity
            if len(intervals) > 3:
                mean_interval = np.mean(intervals)
                std_interval = np.std(intervals)
                
                if std_interval / mean_interval < 0.3:  # Regular pattern
                    # Predict next maneuver
                    next_time = maneuver_times[-1] + timedelta(seconds=mean_interval)
                    
                    if next_time < datetime.utcnow() + self.prediction_horizon:
                        predictions.append({
                            "predicted_time": next_time,
                            "confidence": 0.7,
                            "type": "periodic",
                            "expected_delta_v": self._estimate_delta_v(model)
                        })
                        
        # Check for reactive patterns (e.g., collision avoidance)
        if model.behavior_pattern == BehaviorPattern.EVASIVE:
            predictions.append({
                "predicted_time": datetime.utcnow() + timedelta(hours=1),
                "confidence": 0.5,
                "type": "reactive",
                "expected_delta_v": 5.0  # m/s
            })
            
        return predictions
        
    def _estimate_delta_v(self, model: TargetModel) -> float:
        """Estimate delta-v for predicted maneuver"""
        if model.maneuver_capability:
            recent_maneuvers = model.maneuver_history[-5:]
            if recent_maneuvers:
                delta_vs = [m.get("delta_v", 0) for m in recent_maneuvers]
                return np.mean(delta_vs)
                
        return 1.0  # Default 1 m/s


class TargetModelingSubsystem:
    """Main target modeling and characterization subsystem"""
    
    def __init__(self, kafka_client: WeldersArcKafkaClient):
        self.kafka_client = kafka_client
        self.models: Dict[str, TargetModel] = {}
        self.behavior_analyzer = BehaviorAnalyzer()
        self.maneuver_predictor = ManeuverPredictor()
        self.update_interval = 60  # seconds
        
    async def initialize(self):
        """Initialize target modeling subsystem"""
        # Subscribe to relevant topics
        self.kafka_client.subscribe(
            KafkaTopics.STATE_VECTORS,
            self._handle_state_update
        )
        
        self.kafka_client.subscribe(
            KafkaTopics.MANEUVER_DETECTION,
            self._handle_maneuver_detection
        )
        
        self.kafka_client.subscribe(
            KafkaTopics.EVENT_RF_DETECTION,
            self._handle_rf_detection
        )
        
        # Start background tasks
        asyncio.create_task(self._model_updater())
        asyncio.create_task(self._pattern_analyzer())
        
        logger.info("Target modeling subsystem initialized")
        
    async def _handle_state_update(self, message: WeldersArcMessage):
        """Handle state vector update"""
        try:
            object_id = message.data["object_id"]
            
            # Get or create model
            if object_id not in self.models:
                self.models[object_id] = await self._create_initial_model(object_id)
                
            model = self.models[object_id]
            
            # Add to state history
            state_data = {
                "position": message.data["position"],
                "velocity": message.data["velocity"],
                "timestamp": message.timestamp,
                "covariance": message.data.get("covariance")
            }
            model.state_history.append(state_data)
            
            # Update orbital characteristics
            state_vector = StateVector(
                position=np.array(message.data["position"]),
                velocity=np.array(message.data["velocity"]),
                timestamp=message.timestamp
            )
            
            orbit_params = OrbitCharacterizer.characterize_orbit(state_vector)
            model.semi_major_axis = orbit_params["semi_major_axis"]
            model.eccentricity = orbit_params["eccentricity"]
            model.inclination = orbit_params["inclination"]
            model.mean_motion = orbit_params["mean_motion"]
            
            # Update physical properties if available
            if "area_to_mass" in message.data:
                model.area_to_mass_ratio = message.data["area_to_mass"]
                model.ballistic_coefficient = 1 / model.area_to_mass_ratio
                
            model.last_updated = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error handling state update: {e}")
            
    async def _handle_maneuver_detection(self, message: WeldersArcMessage):
        """Handle maneuver detection event"""
        object_id = message.data["object_id"]
        
        if object_id in self.models:
            model = self.models[object_id]
            
            maneuver_data = {
                "timestamp": message.timestamp,
                "delta_v": message.data.get("delta_v", 0),
                "direction": message.data.get("direction", [0, 0, 0]),
                "confidence": message.data.get("confidence", 0.5),
                "type": message.data.get("maneuver_type", "unknown")
            }
            
            model.maneuver_history.append(maneuver_data)
            
            # Update maneuver capability estimate
            if model.maneuver_capability:
                model.maneuver_capability.maneuver_history.append(maneuver_data)
                
                # Update fuel estimate
                if maneuver_data["delta_v"] > 0:
                    fuel_used = self._estimate_fuel_usage(
                        model.mass,
                        maneuver_data["delta_v"],
                        model.maneuver_capability.specific_impulse
                    )
                    model.maneuver_capability.fuel_remaining -= fuel_used
                    
            # Flag as non-cooperative if unexpected maneuver
            if await self._is_unexpected_maneuver(model, maneuver_data):
                model.cooperative_behavior = False
                model.threat_indicators["unexpected_maneuver"] = 0.8
                
    async def _handle_rf_detection(self, message: WeldersArcMessage):
        """Handle RF emission detection"""
        object_id = message.data.get("object_id")
        
        if object_id and object_id in self.models:
            model = self.models[object_id]
            
            # Update object type if unknown
            if model.object_type == ObjectType.UNKNOWN:
                model.object_type = ObjectType.ACTIVE_SATELLITE
                
            # Add RF activity to threat indicators
            rf_power = message.data.get("power", 0)
            if rf_power > 100:  # High power transmission
                model.threat_indicators["high_power_rf"] = min(1.0, rf_power / 1000)
                
    async def _create_initial_model(self, object_id: str) -> TargetModel:
        """Create initial target model"""
        return TargetModel(
            object_id=object_id,
            object_type=ObjectType.UNKNOWN,
            last_updated=datetime.utcnow(),
            mass=1000.0,  # Default 1000 kg
            cross_sectional_area=10.0,  # Default 10 m²
            ballistic_coefficient=100.0,
            area_to_mass_ratio=0.01,
            mean_motion=15.0,  # Typical LEO
            eccentricity=0.001,
            inclination=0.0,
            semi_major_axis=6778.0,  # 400 km altitude
            behavior_pattern=BehaviorPattern.STATION_KEEPING,
            pattern_confidence=0.5,
            maneuver_capability=ManeuverCapability(
                delta_v_budget=100.0,  # m/s
                thrust_level=1.0,  # N
                specific_impulse=300.0,  # s
                fuel_remaining=50.0,  # kg
                max_acceleration=0.001  # m/s²
            )
        )
        
    async def _model_updater(self):
        """Periodically update models"""
        while True:
            try:
                for object_id, model in self.models.items():
                    # Skip if recently updated
                    if (datetime.utcnow() - model.last_updated).seconds < self.update_interval:
                        continue
                        
                    # Update behavior analysis
                    pattern, confidence = await self.behavior_analyzer.analyze_behavior(model)
                    model.behavior_pattern = pattern
                    model.pattern_confidence = confidence
                    
                    # Update anomaly score
                    model.anomaly_score = await self._calculate_anomaly_score(model)
                    
                    # Predict future maneuvers
                    predictions = await self.maneuver_predictor.predict_maneuvers(model)
                    
                    if predictions:
                        # Publish predictions
                        message = WeldersArcMessage(
                            message_id=f"prediction-{object_id}-{datetime.utcnow().timestamp()}",
                            timestamp=datetime.utcnow(),
                            subsystem=SubsystemID.SS1_MODELING,
                            event_type="maneuver_prediction",
                            data={
                                "object_id": object_id,
                                "predictions": predictions,
                                "behavior_pattern": pattern.value,
                                "confidence": confidence
                            }
                        )
                        
                        await self.kafka_client.publish(
                            KafkaTopics.MANEUVER_PREDICTIONS,
                            message
                        )
                        
                    # Publish updated model
                    await self._publish_model_update(model)
                    
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in model updater: {e}")
                await asyncio.sleep(5)
                
    async def _pattern_analyzer(self):
        """Analyze patterns across multiple objects"""
        while True:
            try:
                # Group objects by behavior
                behavior_groups = defaultdict(list)
                for model in self.models.values():
                    behavior_groups[model.behavior_pattern].append(model)
                    
                # Check for coordinated behaviors
                for pattern, models in behavior_groups.items():
                    if len(models) > 2 and pattern in [
                        BehaviorPattern.RENDEZVOUS,
                        BehaviorPattern.PROXIMITY_OPS,
                        BehaviorPattern.AGGRESSIVE
                    ]:
                        # Possible coordinated activity
                        await self._analyze_coordination(models)
                        
                await asyncio.sleep(300)  # Every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in pattern analyzer: {e}")
                await asyncio.sleep(60)
                
    async def _analyze_coordination(self, models: List[TargetModel]):
        """Analyze potential coordination between objects"""
        # Calculate spatial clustering
        positions = []
        for model in models:
            if model.state_history:
                positions.append(model.state_history[-1]["position"])
                
        if len(positions) > 2:
            # Use DBSCAN for clustering
            clustering = DBSCAN(eps=100, min_samples=2)  # 100 km threshold
            clusters = clustering.fit_predict(positions)
            
            # Group by cluster
            cluster_groups = defaultdict(list)
            for i, cluster in enumerate(clusters):
                if cluster != -1:  # Not noise
                    cluster_groups[cluster].append(models[i])
                    
            # Publish coordination alerts
            for cluster_id, cluster_models in cluster_groups.items():
                if len(cluster_models) > 1:
                    message = WeldersArcMessage(
                        message_id=f"coordination-{datetime.utcnow().timestamp()}",
                        timestamp=datetime.utcnow(),
                        subsystem=SubsystemID.SS1_MODELING,
                        event_type="coordinated_behavior",
                        data={
                            "object_ids": [m.object_id for m in cluster_models],
                            "behavior_pattern": cluster_models[0].behavior_pattern.value,
                            "cluster_size": len(cluster_models),
                            "mean_separation": self._calculate_mean_separation(cluster_models)
                        }
                    )
                    
                    await self.kafka_client.publish(
                        KafkaTopics.ALERT_OPERATOR,
                        message
                    )
                    
    def _calculate_mean_separation(self, models: List[TargetModel]) -> float:
        """Calculate mean separation between objects"""
        if len(models) < 2:
            return 0.0
            
        separations = []
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                if models[i].state_history and models[j].state_history:
                    pos1 = np.array(models[i].state_history[-1]["position"])
                    pos2 = np.array(models[j].state_history[-1]["position"])
                    separation = np.linalg.norm(pos1 - pos2)
                    separations.append(separation)
                    
        return np.mean(separations) if separations else 0.0
        
    async def _calculate_anomaly_score(self, model: TargetModel) -> float:
        """Calculate anomaly score for object"""
        scores = []
        
        # Orbital anomaly
        if model.eccentricity > 0.1:  # High eccentricity
            scores.append(0.7)
            
        # Behavioral anomaly
        if model.behavior_pattern == BehaviorPattern.ANOMALOUS:
            scores.append(0.9)
            
        # Maneuver frequency anomaly
        recent_maneuvers = [m for m in model.maneuver_history
                           if (datetime.utcnow() - m["timestamp"]).days < 7]
        if len(recent_maneuvers) > 10:
            scores.append(0.8)
            
        # Threat indicator anomaly
        if model.threat_indicators:
            threat_score = sum(model.threat_indicators.values()) / len(model.threat_indicators)
            scores.append(threat_score)
            
        return np.mean(scores) if scores else 0.0
        
    async def _is_unexpected_maneuver(self, model: TargetModel, 
                                     maneuver: Dict[str, Any]) -> bool:
        """Check if maneuver is unexpected"""
        # Check against predicted maneuvers
        predictions = await self.maneuver_predictor.predict_maneuvers(model)
        
        for pred in predictions:
            time_diff = abs((pred["predicted_time"] - maneuver["timestamp"]).total_seconds())
            if time_diff < 3600:  # Within 1 hour
                return False  # Expected
                
        # Check against typical patterns
        if model.behavior_pattern == BehaviorPattern.STATION_KEEPING:
            if maneuver["delta_v"] > 10:  # Large maneuver for station keeping
                return True
                
        return False
        
    def _estimate_fuel_usage(self, mass: float, delta_v: float, isp: float) -> float:
        """Estimate fuel usage from maneuver"""
        # Tsiolkovsky rocket equation
        g0 = 9.81  # m/s²
        mass_ratio = np.exp(delta_v / (isp * g0))
        fuel_used = mass * (mass_ratio - 1) / mass_ratio
        return fuel_used
        
    async def _publish_model_update(self, model: TargetModel):
        """Publish updated target model"""
        message = WeldersArcMessage(
            message_id=f"model-{model.object_id}-{datetime.utcnow().timestamp()}",
            timestamp=datetime.utcnow(),
            subsystem=SubsystemID.SS1_MODELING,
            event_type="model_update",
            data={
                "object_id": model.object_id,
                "object_type": model.object_type.value,
                "behavior_pattern": model.behavior_pattern.value,
                "pattern_confidence": model.pattern_confidence,
                "anomaly_score": model.anomaly_score,
                "threat_indicators": model.threat_indicators,
                "cooperative": model.cooperative_behavior,
                "orbital_parameters": {
                    "semi_major_axis": model.semi_major_axis,
                    "eccentricity": model.eccentricity,
                    "inclination": model.inclination,
                    "mean_motion": model.mean_motion
                },
                "maneuver_capability": {
                    "delta_v_remaining": model.maneuver_capability.delta_v_budget
                    if model.maneuver_capability else 0,
                    "fuel_remaining": model.maneuver_capability.fuel_remaining
                    if model.maneuver_capability else 0
                }
            }
        )
        
        await self.kafka_client.publish(KafkaTopics.TARGET_MODELS, message)
        
    def get_model(self, object_id: str) -> Optional[TargetModel]:
        """Get target model for object"""
        return self.models.get(object_id)
        
    def get_all_models(self) -> Dict[str, TargetModel]:
        """Get all target models"""
        return self.models.copy()
        
    def get_threats(self, threshold: float = 0.7) -> List[TargetModel]:
        """Get objects above threat threshold"""
        threats = []
        for model in self.models.values():
            if model.anomaly_score > threshold or not model.cooperative_behavior:
                threats.append(model)
        return threats 