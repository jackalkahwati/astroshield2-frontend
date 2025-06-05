"""
Subsystem 5: Hostility Monitoring
Weapon Engagement Zone prediction, intent assessment, and threat evaluation
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import asyncio
import logging
from enum import Enum

from ..kafka.kafka_client import (
    WeldersArcKafkaClient,
    WeldersArcMessage,
    KafkaTopics,
    SubsystemID,
    EventType
)

logger = logging.getLogger(__name__)


class ThreatType(Enum):
    """Types of space threats"""
    KINETIC_KILL = "kinetic_kill"
    CO_ORBITAL_INTERCEPT = "co_orbital_intercept"
    ORBITAL_BOMBARDMENT = "orbital_bombardment"
    RF_JAMMING = "rf_jamming"
    LASER_DAZZLE = "laser_dazzle"
    EMP_ATTACK = "emp_attack"
    CYBER_ATTACK = "cyber_attack"


class IntentLevel(Enum):
    """Hostile intent assessment levels"""
    BENIGN = "benign"
    UNKNOWN = "unknown"
    SUSPICIOUS = "suspicious"
    LIKELY_HOSTILE = "likely_hostile"
    HOSTILE = "hostile"
    IMMINENT_THREAT = "imminent_threat"


@dataclass
class WeaponEngagementZone:
    """Predicted weapon engagement opportunity"""
    threat_id: str
    target_id: str
    threat_type: ThreatType
    start_time: datetime
    end_time: datetime
    probability: float
    geometry: Dict[str, Any]  # Range, angle, relative velocity
    constraints: Dict[str, Any]  # Environmental, technical constraints


@dataclass
class PatternOfLife:
    """Object behavioral pattern"""
    object_id: str
    maneuver_frequency: float  # Maneuvers per day
    maneuver_times: List[datetime]
    typical_delta_v: float
    orbital_regime: str
    mission_type: Optional[str] = None


@dataclass
class ThreatAssessment:
    """Comprehensive threat assessment"""
    object_id: str
    threat_level: IntentLevel
    threat_types: List[ThreatType]
    confidence: float
    indicators: Dict[str, Any]
    recommended_actions: List[str]


class WEZPredictor:
    """Weapon Engagement Zone prediction engine"""
    
    def __init__(self):
        self.threat_models = {
            ThreatType.KINETIC_KILL: self._predict_kinetic_kill,
            ThreatType.CO_ORBITAL_INTERCEPT: self._predict_co_orbital,
            ThreatType.RF_JAMMING: self._predict_rf_jamming,
            ThreatType.LASER_DAZZLE: self._predict_laser_engagement
        }
        
    async def predict_wez(
        self,
        threat_state: Dict[str, Any],
        target_state: Dict[str, Any],
        threat_capabilities: Dict[str, Any],
        time_window: Tuple[datetime, datetime]
    ) -> List[WeaponEngagementZone]:
        """Predict all weapon engagement opportunities"""
        wez_predictions = []
        
        for threat_type, predictor in self.threat_models.items():
            if self._can_employ_weapon(threat_capabilities, threat_type):
                zones = await predictor(
                    threat_state,
                    target_state,
                    threat_capabilities,
                    time_window
                )
                wez_predictions.extend(zones)
                
        return wez_predictions
        
    def _can_employ_weapon(self, capabilities: Dict[str, Any], threat_type: ThreatType) -> bool:
        """Check if threat has capability for weapon type"""
        capability_map = {
            ThreatType.KINETIC_KILL: "kinetic_interceptor",
            ThreatType.RF_JAMMING: "rf_transmitter",
            ThreatType.LASER_DAZZLE: "laser_payload",
            ThreatType.CO_ORBITAL_INTERCEPT: "proximity_ops"
        }
        
        required_capability = capability_map.get(threat_type)
        return capabilities.get(required_capability, False)
        
    async def _predict_kinetic_kill(
        self,
        threat_state: Dict[str, Any],
        target_state: Dict[str, Any],
        capabilities: Dict[str, Any],
        time_window: Tuple[datetime, datetime]
    ) -> List[WeaponEngagementZone]:
        """Predict kinetic kill opportunities"""
        zones = []
        
        # Calculate intercept opportunities
        threat_pos = np.array(threat_state["position"])
        threat_vel = np.array(threat_state["velocity"])
        target_pos = np.array(target_state["position"])
        target_vel = np.array(target_state["velocity"])
        
        # Simplified intercept calculation
        rel_pos = target_pos - threat_pos
        rel_vel = target_vel - threat_vel
        range_km = np.linalg.norm(rel_pos)
        closing_rate = -np.dot(rel_pos, rel_vel) / range_km
        
        if closing_rate > 0 and range_km < capabilities.get("max_intercept_range", 1000):
            time_to_intercept = range_km / closing_rate
            intercept_time = datetime.utcnow() + timedelta(seconds=time_to_intercept)
            
            if time_window[0] <= intercept_time <= time_window[1]:
                zones.append(WeaponEngagementZone(
                    threat_id=threat_state["object_id"],
                    target_id=target_state["object_id"],
                    threat_type=ThreatType.KINETIC_KILL,
                    start_time=intercept_time - timedelta(minutes=5),
                    end_time=intercept_time + timedelta(minutes=5),
                    probability=0.85,
                    geometry={
                        "range_km": range_km,
                        "closing_rate_km_s": closing_rate,
                        "intercept_angle_deg": 0  # Placeholder
                    },
                    constraints={
                        "delta_v_required": 0.5,  # km/s placeholder
                        "fuel_required": 10  # kg placeholder
                    }
                ))
                
        return zones
        
    async def _predict_co_orbital(
        self,
        threat_state: Dict[str, Any],
        target_state: Dict[str, Any],
        capabilities: Dict[str, Any],
        time_window: Tuple[datetime, datetime]
    ) -> List[WeaponEngagementZone]:
        """Predict co-orbital intercept opportunities"""
        zones = []
        
        # Extract orbital elements
        threat_pos = np.array(threat_state["position"])
        threat_vel = np.array(threat_state["velocity"])
        target_pos = np.array(target_state["position"])
        target_vel = np.array(target_state["velocity"])
        
        # Calculate relative orbital elements
        rel_pos = target_pos - threat_pos
        rel_vel = target_vel - threat_vel
        
        # Check if already in co-orbital regime
        range_km = np.linalg.norm(rel_pos)
        rel_speed = np.linalg.norm(rel_vel)
        
        co_orbital_range = capabilities.get("co_orbital_range_km", 50)
        max_approach_speed = capabilities.get("max_approach_speed_km_s", 0.01)
        
        if range_km <= co_orbital_range:
            # Already in co-orbital regime
            # Check for proximity operations capability
            if rel_speed <= max_approach_speed:
                zones.append(WeaponEngagementZone(
                    threat_id=threat_state["object_id"],
                    target_id=target_state["object_id"],
                    threat_type=ThreatType.CO_ORBITAL_INTERCEPT,
                    start_time=datetime.utcnow(),
                    end_time=time_window[1],
                    probability=0.90,
                    geometry={
                        "range_km": range_km,
                        "relative_speed_km_s": rel_speed,
                        "phase_angle_deg": self._calculate_phase_angle(threat_pos, target_pos)
                    },
                    constraints={
                        "fuel_required": self._estimate_fuel_for_proximity_ops(rel_speed, capabilities),
                        "time_to_intercept": range_km / max(rel_speed, 0.001)
                    }
                ))
        else:
            # Calculate transfer orbit requirements
            transfer_dv = self._calculate_hohmann_transfer(threat_state, target_state)
            
            if transfer_dv <= capabilities.get("max_delta_v_km_s", 1.0):
                transfer_time = self._calculate_transfer_time(threat_state, target_state)
                arrival_time = datetime.utcnow() + timedelta(seconds=transfer_time)
                
                if time_window[0] <= arrival_time <= time_window[1]:
                    zones.append(WeaponEngagementZone(
                        threat_id=threat_state["object_id"],
                        target_id=target_state["object_id"],
                        threat_type=ThreatType.CO_ORBITAL_INTERCEPT,
                        start_time=arrival_time - timedelta(hours=1),
                        end_time=arrival_time + timedelta(hours=1),
                        probability=0.75,
                        geometry={
                            "current_range_km": range_km,
                            "transfer_delta_v_km_s": transfer_dv,
                            "transfer_time_hours": transfer_time / 3600
                        },
                        constraints={
                            "fuel_required": self._estimate_fuel_from_dv(transfer_dv, capabilities),
                            "transfer_feasible": True
                        }
                    ))
                    
        return zones
        
    def _calculate_phase_angle(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Calculate phase angle between two position vectors"""
        dot_product = np.dot(pos1, pos2)
        magnitudes = np.linalg.norm(pos1) * np.linalg.norm(pos2)
        cos_angle = dot_product / magnitudes
        return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        
    def _calculate_hohmann_transfer(self, state1: Dict, state2: Dict) -> float:
        """Calculate delta-V for Hohmann transfer"""
        mu = 398600.4418  # Earth gravitational parameter
        
        r1 = np.linalg.norm(state1["position"])
        r2 = np.linalg.norm(state2["position"])
        
        # Hohmann transfer delta-Vs
        a_transfer = (r1 + r2) / 2
        
        v1 = np.sqrt(mu / r1)
        v_transfer_1 = np.sqrt(mu * (2/r1 - 1/a_transfer))
        dv1 = abs(v_transfer_1 - v1)
        
        v2 = np.sqrt(mu / r2)
        v_transfer_2 = np.sqrt(mu * (2/r2 - 1/a_transfer))
        dv2 = abs(v2 - v_transfer_2)
        
        return dv1 + dv2
        
    def _calculate_transfer_time(self, state1: Dict, state2: Dict) -> float:
        """Calculate transfer time for orbital maneuver"""
        mu = 398600.4418
        
        r1 = np.linalg.norm(state1["position"])
        r2 = np.linalg.norm(state2["position"])
        
        a_transfer = (r1 + r2) / 2
        transfer_time = np.pi * np.sqrt(a_transfer**3 / mu)
        
        return transfer_time
        
    def _estimate_fuel_for_proximity_ops(self, rel_speed: float, capabilities: Dict) -> float:
        """Estimate fuel required for proximity operations"""
        isp = capabilities.get("specific_impulse", 300)
        dry_mass = capabilities.get("dry_mass_kg", 500)
        
        # Tsiolkovsky equation
        fuel_fraction = 1 - np.exp(-rel_speed * 1000 / (isp * 9.81))
        fuel_required = dry_mass * fuel_fraction / (1 - fuel_fraction)
        
        return fuel_required
        
    def _estimate_fuel_from_dv(self, delta_v: float, capabilities: Dict) -> float:
        """Estimate fuel from delta-V requirement"""
        isp = capabilities.get("specific_impulse", 300)
        dry_mass = capabilities.get("dry_mass_kg", 500)
        
        fuel_fraction = 1 - np.exp(-delta_v * 1000 / (isp * 9.81))
        fuel_required = dry_mass * fuel_fraction / (1 - fuel_fraction)
        
        return fuel_required
        
    async def _predict_rf_jamming(
        self,
        threat_state: Dict[str, Any],
        target_state: Dict[str, Any],
        capabilities: Dict[str, Any],
        time_window: Tuple[datetime, datetime]
    ) -> List[WeaponEngagementZone]:
        """Predict RF jamming opportunities"""
        zones = []
        
        # Check if in RF range
        threat_pos = np.array(threat_state["position"])
        target_pos = np.array(target_state["position"])
        range_km = np.linalg.norm(target_pos - threat_pos)
        
        max_jamming_range = capabilities.get("rf_max_range_km", 5000)
        
        if range_km <= max_jamming_range:
            # Check frequency compatibility
            target_freqs = target_state.get("rf_frequencies", [])
            jammer_freqs = capabilities.get("rf_jamming_frequencies", [])
            
            if any(freq in jammer_freqs for freq in target_freqs):
                zones.append(WeaponEngagementZone(
                    threat_id=threat_state["object_id"],
                    target_id=target_state["object_id"],
                    threat_type=ThreatType.RF_JAMMING,
                    start_time=datetime.utcnow(),
                    end_time=datetime.utcnow() + timedelta(hours=1),
                    probability=0.95,
                    geometry={
                        "range_km": range_km,
                        "elevation_angle": 0,  # Placeholder
                        "azimuth_angle": 0  # Placeholder
                    },
                    constraints={
                        "power_required_watts": 100,
                        "frequency_overlap": True
                    }
                ))
                
        return zones
        
    async def _predict_laser_engagement(
        self,
        threat_state: Dict[str, Any],
        target_state: Dict[str, Any],
        capabilities: Dict[str, Any],
        time_window: Tuple[datetime, datetime]
    ) -> List[WeaponEngagementZone]:
        """Predict laser dazzle/damage opportunities"""
        zones = []
        
        # Extract positions
        threat_pos = np.array(threat_state["position"])
        target_pos = np.array(target_state["position"])
        range_km = np.linalg.norm(target_pos - threat_pos)
        
        # Laser parameters
        max_range = capabilities.get("laser_max_range_km", 1000)
        min_range = capabilities.get("laser_min_range_km", 10)
        power_watts = capabilities.get("laser_power_watts", 10000)
        
        if min_range <= range_km <= max_range:
            # Check line of sight (simplified - not considering Earth obstruction)
            los_clear = self._check_line_of_sight(threat_pos, target_pos)
            
            if los_clear:
                # Calculate beam parameters
                wavelength = capabilities.get("laser_wavelength_nm", 1064) * 1e-9
                aperture = capabilities.get("aperture_diameter_m", 0.5)
                
                # Diffraction-limited spot size
                spot_diameter = 1.22 * wavelength * range_km * 1000 / aperture
                
                # Power density at target
                spot_area = np.pi * (spot_diameter / 2) ** 2
                power_density = power_watts / spot_area
                
                # Effectiveness thresholds
                dazzle_threshold = 1e3  # W/m^2
                damage_threshold = 1e6  # W/m^2
                
                if power_density >= dazzle_threshold:
                    effectiveness = "damage" if power_density >= damage_threshold else "dazzle"
                    
                    # Calculate engagement window based on relative motion
                    rel_vel = np.array(target_state["velocity"]) - np.array(threat_state["velocity"])
                    angular_rate = np.linalg.norm(rel_vel) / range_km
                    
                    # Maximum tracking rate
                    max_tracking_rate = capabilities.get("max_tracking_rate_rad_s", 0.1)
                    
                    if angular_rate <= max_tracking_rate:
                        zones.append(WeaponEngagementZone(
                            threat_id=threat_state["object_id"],
                            target_id=target_state["object_id"],
                            threat_type=ThreatType.LASER_DAZZLE,
                            start_time=datetime.utcnow(),
                            end_time=datetime.utcnow() + timedelta(minutes=30),
                            probability=0.85 if effectiveness == "damage" else 0.95,
                            geometry={
                                "range_km": range_km,
                                "spot_diameter_m": spot_diameter,
                                "power_density_w_m2": power_density,
                                "angular_rate_rad_s": angular_rate
                            },
                            constraints={
                                "power_required_watts": power_watts,
                                "effectiveness": effectiveness,
                                "atmospheric_transmission": 0.8  # Simplified
                            }
                        ))
                        
        return zones
        
    def _check_line_of_sight(self, pos1: np.ndarray, pos2: np.ndarray) -> bool:
        """Check if line of sight is clear (simplified)"""
        # Earth radius
        Re = 6378.137  # km
        
        # Vector from pos1 to pos2
        los_vector = pos2 - pos1
        los_distance = np.linalg.norm(los_vector)
        
        # Check if line passes through Earth
        # Closest approach distance to Earth center
        t = -np.dot(pos1, los_vector) / (los_distance ** 2)
        t = np.clip(t, 0, 1)
        
        closest_point = pos1 + t * los_vector
        closest_distance = np.linalg.norm(closest_point)
        
        # Add some margin for atmosphere
        return closest_distance > Re + 100  # 100 km atmosphere


class IntentAssessor:
    """Assess hostile intent from behaviors and indicators"""
    
    def __init__(self):
        self.pattern_database: Dict[str, PatternOfLife] = {}
        self.intent_thresholds = {
            IntentLevel.BENIGN: 0.0,
            IntentLevel.UNKNOWN: 0.2,
            IntentLevel.SUSPICIOUS: 0.4,
            IntentLevel.LIKELY_HOSTILE: 0.6,
            IntentLevel.HOSTILE: 0.8,
            IntentLevel.IMMINENT_THREAT: 0.95
        }
        
    async def assess_intent(
        self,
        object_id: str,
        current_behavior: Dict[str, Any],
        wez_predictions: List[WeaponEngagementZone],
        historical_data: Dict[str, Any]
    ) -> ThreatAssessment:
        """Comprehensive intent assessment"""
        indicators = {}
        threat_score = 0.0
        
        # Check pattern of life violations
        if object_id in self.pattern_database:
            pol_score = await self._check_pattern_violations(
                object_id,
                current_behavior
            )
            indicators["pattern_violation"] = pol_score
            threat_score += pol_score * 0.3
            
        # Check weapon engagement zones
        wez_score = self._assess_wez_patterns(wez_predictions)
        indicators["wez_patterns"] = wez_score
        threat_score += wez_score * 0.4
        
        # Check pursuit behaviors
        pursuit_score = await self._detect_pursuit_behavior(
            object_id,
            current_behavior,
            historical_data
        )
        indicators["pursuit_behavior"] = pursuit_score
        threat_score += pursuit_score * 0.3
        
        # Determine threat level
        threat_level = IntentLevel.UNKNOWN
        for level, threshold in sorted(
            self.intent_thresholds.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            if threat_score >= threshold:
                threat_level = level
                break
                
        # Determine threat types
        threat_types = []
        for wez in wez_predictions:
            if wez.probability > 0.7:
                threat_types.append(wez.threat_type)
                
        # Generate recommendations
        recommendations = self._generate_recommendations(
            threat_level,
            threat_types,
            indicators
        )
        
        return ThreatAssessment(
            object_id=object_id,
            threat_level=threat_level,
            threat_types=list(set(threat_types)),
            confidence=min(0.95, threat_score + 0.2),
            indicators=indicators,
            recommended_actions=recommendations
        )
        
    async def _check_pattern_violations(
        self,
        object_id: str,
        current_behavior: Dict[str, Any]
    ) -> float:
        """Check for pattern of life violations"""
        pattern = self.pattern_database[object_id]
        violation_score = 0.0
        
        # Check maneuver timing
        current_time = datetime.utcnow()
        expected_maneuver_time = self._predict_next_maneuver(pattern)
        
        if current_behavior.get("maneuvering", False):
            time_diff = abs((current_time - expected_maneuver_time).total_seconds())
            if time_diff > 3600 * 6:  # More than 6 hours off schedule
                violation_score += 0.5
                
        # Check maneuver magnitude
        if "delta_v" in current_behavior:
            delta_v = current_behavior["delta_v"]
            if delta_v > pattern.typical_delta_v * 2:
                violation_score += 0.5
                
        return min(1.0, violation_score)
        
    def _assess_wez_patterns(
        self,
        wez_predictions: List[WeaponEngagementZone]
    ) -> float:
        """Assess weapon engagement zone patterns"""
        if not wez_predictions:
            return 0.0
            
        # High probability WEZ is concerning
        max_probability = max(wez.probability for wez in wez_predictions)
        
        # Multiple threat types is concerning
        threat_types = set(wez.threat_type for wez in wez_predictions)
        
        score = max_probability * 0.7 + len(threat_types) * 0.1
        return min(1.0, score)
        
    async def _detect_pursuit_behavior(
        self,
        object_id: str,
        current_behavior: Dict[str, Any],
        historical_data: Dict[str, Any]
    ) -> float:
        """Detect pursuit or shadowing behavior"""
        pursuit_score = 0.0
        
        # Check if maintaining constant range to target
        if "relative_motion" in current_behavior:
            rel_motion = current_behavior["relative_motion"]
            if rel_motion.get("range_rate_km_s", 1) < 0.01:  # Nearly constant range
                pursuit_score += 0.3
                
        # Check if matching target maneuvers
        if "correlated_maneuvers" in historical_data:
            correlation = historical_data["correlated_maneuvers"]
            if correlation > 0.8:
                pursuit_score += 0.7
                
        return min(1.0, pursuit_score)
        
    def _predict_next_maneuver(self, pattern: PatternOfLife) -> datetime:
        """Predict next expected maneuver time"""
        if not pattern.maneuver_times:
            return datetime.utcnow() + timedelta(days=7)
            
        # Simple prediction based on average interval
        intervals = []
        for i in range(1, len(pattern.maneuver_times)):
            interval = (pattern.maneuver_times[i] - pattern.maneuver_times[i-1]).total_seconds()
            intervals.append(interval)
            
        avg_interval = np.mean(intervals) if intervals else 86400 * 7
        last_maneuver = pattern.maneuver_times[-1]
        
        return last_maneuver + timedelta(seconds=avg_interval)
        
    def _generate_recommendations(
        self,
        threat_level: IntentLevel,
        threat_types: List[ThreatType],
        indicators: Dict[str, Any]
    ) -> List[str]:
        """Generate recommended defensive actions"""
        recommendations = []
        
        if threat_level == IntentLevel.IMMINENT_THREAT:
            recommendations.append("IMMEDIATE EVASIVE ACTION REQUIRED")
            recommendations.append("ACTIVATE ALL COUNTERMEASURES")
            recommendations.append("NOTIFY COMMAND IMMEDIATELY")
            
        elif threat_level == IntentLevel.HOSTILE:
            recommendations.append("PREPARE EVASIVE MANEUVERS")
            recommendations.append("INCREASE MONITORING FREQUENCY")
            recommendations.append("READY COUNTERMEASURES")
            
        elif threat_level == IntentLevel.LIKELY_HOSTILE:
            recommendations.append("ENHANCE SURVEILLANCE")
            recommendations.append("PREPARE CONTINGENCY PLANS")
            recommendations.append("REQUEST ADDITIONAL SENSOR COVERAGE")
            
        elif threat_level == IntentLevel.SUSPICIOUS:
            recommendations.append("MAINTAIN VIGILANCE")
            recommendations.append("COLLECT ADDITIONAL DATA")
            
        # Add threat-specific recommendations
        if ThreatType.RF_JAMMING in threat_types:
            recommendations.append("SWITCH TO BACKUP FREQUENCIES")
            
        if ThreatType.KINETIC_KILL in threat_types:
            recommendations.append("CALCULATE EVASION MANEUVERS")
            
        return recommendations


class HostilityMonitor:
    """Main hostility monitoring subsystem"""
    
    def __init__(self, kafka_client: WeldersArcKafkaClient):
        self.kafka_client = kafka_client
        self.wez_predictor = WEZPredictor()
        self.intent_assessor = IntentAssessor()
        self.threat_catalog: Dict[str, ThreatAssessment] = {}
        
    async def initialize(self):
        """Initialize hostility monitoring subsystem"""
        # Subscribe to relevant topics
        self.kafka_client.subscribe(
            KafkaTopics.STATE_VECTORS,
            self._handle_state_update
        )
        
        self.kafka_client.subscribe(
            KafkaTopics.TARGET_MODEL_UPDATE,
            self._handle_capability_update
        )
        
        self.kafka_client.subscribe(
            KafkaTopics.EVENT_MANEUVER_DETECTION,
            self._handle_maneuver_event
        )
        
        self.kafka_client.subscribe(
            KafkaTopics.EVENT_PROXIMITY_ALERT,
            self._handle_proximity_event
        )
        
        logger.info("Hostility monitoring subsystem initialized")
        
    async def _handle_state_update(self, message: WeldersArcMessage):
        """Handle state vector updates"""
        state_data = message.data
        object_id = state_data["object_id"]
        
        # Check if this is a tracked threat
        if object_id in self.threat_catalog:
            # Update WEZ predictions based on new state
            threat_state = {
                "object_id": object_id,
                "position": state_data["state_vector"]["position"],
                "velocity": state_data["state_vector"]["velocity"]
            }
            
            # Get potential targets
            # In production, this would query the catalog
            potential_targets = await self._get_high_value_targets()
            
            for target in potential_targets:
                # Predict WEZ for each target
                target_state = await self._get_target_state(target["object_id"])
                
                if target_state:
                    # Get threat capabilities from target model database
                    capabilities = await self._get_threat_capabilities(object_id)
                    
                    if capabilities:
                        time_window = (
                            datetime.utcnow(),
                            datetime.utcnow() + timedelta(hours=24)
                        )
                        
                        wez_predictions = await self.wez_predictor.predict_wez(
                            threat_state,
                            target_state,
                            capabilities,
                            time_window
                        )
                        
                        if wez_predictions:
                            # Reassess intent with new WEZ data
                            assessment = await self.intent_assessor.assess_intent(
                                object_id,
                                {"state_update": state_data},
                                wez_predictions,
                                {}
                            )
                            
                            # Update threat catalog
                            self.threat_catalog[object_id] = assessment
                            
                            # Publish if threat level is significant
                            if assessment.threat_level >= IntentLevel.LIKELY_HOSTILE:
                                await self._publish_threat_warning(assessment)
                                
    async def _get_high_value_targets(self) -> List[Dict[str, Any]]:
        """Get list of high value targets to protect"""
        # In production, this would query a database
        return [
            {"object_id": "USA-300", "value": "high", "type": "communication"},
            {"object_id": "USA-301", "value": "critical", "type": "navigation"},
            {"object_id": "USA-302", "value": "high", "type": "reconnaissance"}
        ]
        
    async def _get_target_state(self, object_id: str) -> Optional[Dict[str, Any]]:
        """Get current state of a target"""
        # In production, query state estimation service
        # Placeholder implementation
        return {
            "object_id": object_id,
            "position": [7000.0, 0.0, 0.0],  # km
            "velocity": [0.0, 7.5, 0.0]       # km/s
        }
        
    async def _get_threat_capabilities(self, object_id: str) -> Optional[Dict[str, Any]]:
        """Get threat capabilities from target model database"""
        # In production, query SS1 target model database
        # Placeholder based on object characteristics
        return {
            "kinetic_interceptor": True,
            "rf_transmitter": True,
            "laser_payload": False,
            "proximity_ops": True,
            "max_intercept_range": 1000,
            "rf_max_range_km": 5000,
            "rf_jamming_frequencies": [2200, 2300, 8000, 8100],  # MHz
            "max_delta_v_km_s": 0.5,
            "specific_impulse": 300,
            "dry_mass_kg": 500,
            "co_orbital_range_km": 50,
            "max_approach_speed_km_s": 0.01
        }
        
    async def _handle_capability_update(self, message: WeldersArcMessage):
        """Handle target capability updates"""
        capability_data = message.data
        object_id = capability_data["object_id"]
        
        # Check if this affects any threat assessments
        if object_id in self.threat_catalog:
            # Re-evaluate threat with new capabilities
            logger.info(f"Capability update for tracked threat {object_id}")
            
            # Trigger full reassessment
            await self._reassess_threat(object_id, capability_data)
            
    async def _reassess_threat(self, object_id: str, new_capabilities: Dict[str, Any]):
        """Reassess threat with updated information"""
        # Get current assessment
        current_assessment = self.threat_catalog.get(object_id)
        
        if current_assessment:
            # Re-run WEZ predictions with new capabilities
            # This would be a full reassessment workflow
            logger.info(f"Reassessing threat {object_id} with new capabilities")

    async def _handle_maneuver_event(self, message: WeldersArcMessage):
        """Handle maneuver detection events"""
        object_id = message.data["object_id"]
        maneuver_data = message.data
        
        # Update pattern of life
        if object_id not in self.intent_assessor.pattern_database:
            self.intent_assessor.pattern_database[object_id] = PatternOfLife(
                object_id=object_id,
                maneuver_frequency=0.0,
                maneuver_times=[],
                typical_delta_v=0.0,
                orbital_regime="LEO"  # Would determine from orbit
            )
            
        pattern = self.intent_assessor.pattern_database[object_id]
        pattern.maneuver_times.append(datetime.utcnow())
        
        # Update typical delta-V
        if "delta_v_magnitude" in maneuver_data:
            if pattern.typical_delta_v == 0:
                pattern.typical_delta_v = maneuver_data["delta_v_magnitude"]
            else:
                # Exponential moving average
                alpha = 0.3
                pattern.typical_delta_v = (
                    alpha * maneuver_data["delta_v_magnitude"] + 
                    (1 - alpha) * pattern.typical_delta_v
                )
                
        # Calculate maneuver frequency
        if len(pattern.maneuver_times) > 1:
            time_span = (pattern.maneuver_times[-1] - pattern.maneuver_times[0]).total_seconds()
            pattern.maneuver_frequency = len(pattern.maneuver_times) / (time_span / 86400)  # per day
            
        # Reassess intent
        current_behavior = {
            "maneuvering": True,
            "delta_v": maneuver_data.get("delta_v_magnitude", 0),
            "maneuver_type": maneuver_data.get("maneuver_type", "unknown")
        }
        
        # Get WEZ predictions for this object
        wez_predictions = []  # Would fetch from recent calculations
        
        assessment = await self.intent_assessor.assess_intent(
            object_id,
            current_behavior,
            wez_predictions,
            {"maneuver_history": pattern.maneuver_times}
        )
        
        # Update catalog
        self.threat_catalog[object_id] = assessment
        
        # Publish threat warning if necessary
        if assessment.threat_level >= IntentLevel.LIKELY_HOSTILE:
            await self._publish_threat_warning(assessment)

    async def _handle_proximity_event(self, message: WeldersArcMessage):
        """Handle proximity events"""
        proximity_data = message.data
        object1_id = proximity_data["object1_id"]
        object2_id = proximity_data["object2_id"]
        
        # Immediate threat assessment for close approaches
        range_km = proximity_data["range_km"]
        relative_velocity = proximity_data["relative_velocity_km_s"]
        
        # Check if either object is in threat catalog
        threat_id = None
        target_id = None
        
        if object1_id in self.threat_catalog:
            threat_id = object1_id
            target_id = object2_id
        elif object2_id in self.threat_catalog:
            threat_id = object2_id
            target_id = object1_id
            
        if threat_id:
            # Emergency WEZ assessment
            logger.warning(f"Proximity event involving tracked threat {threat_id}")
            
            # Create immediate threat assessment
            assessment = ThreatAssessment(
                object_id=threat_id,
                threat_level=IntentLevel.IMMINENT_THREAT if range_km < 10 else IntentLevel.HOSTILE,
                threat_types=[ThreatType.KINETIC_KILL, ThreatType.CO_ORBITAL_INTERCEPT],
                confidence=0.95,
                indicators={
                    "proximity_range_km": range_km,
                    "closing_rate_km_s": relative_velocity,
                    "time_to_impact_s": range_km / relative_velocity if relative_velocity > 0 else float('inf')
                },
                recommended_actions=[
                    "IMMEDIATE EVASIVE MANEUVER REQUIRED",
                    "ACTIVATE ALL DEFENSIVE SYSTEMS",
                    "ALERT COMMAND AUTHORITY"
                ]
            )
            
            # Immediate publication
            await self._publish_threat_warning(assessment)
            
            # Also publish emergency alert
            emergency_message = WeldersArcMessage(
                message_id=f"emergency-proximity-{threat_id}-{datetime.utcnow().timestamp()}",
                timestamp=datetime.utcnow(),
                subsystem=SubsystemID.SS5_HOSTILITY,
                event_type="emergency_proximity",
                data={
                    "threat_id": threat_id,
                    "target_id": target_id,
                    "range_km": range_km,
                    "closing_rate_km_s": relative_velocity,
                    "recommended_action": "IMMEDIATE_EVASION"
                }
            )
            
            await self.kafka_client.publish(KafkaTopics.ALERT_OPERATOR, emergency_message)

    async def _publish_threat_warning(self, assessment: ThreatAssessment):
        """Publish threat warning to response subsystem"""
        message = WeldersArcMessage(
            message_id=f"threat-warning-{assessment.object_id}-{datetime.utcnow().timestamp()}",
            timestamp=datetime.utcnow(),
            subsystem=SubsystemID.SS5_HOSTILITY,
            event_type="threat_warning",
            data={
                "object_id": assessment.object_id,
                "threat_level": assessment.threat_level.value,
                "threat_types": [t.value for t in assessment.threat_types],
                "confidence": assessment.confidence,
                "indicators": assessment.indicators,
                "recommended_actions": assessment.recommended_actions
            }
        )
        
        await self.kafka_client.publish(KafkaTopics.THREAT_WARNING, message) 