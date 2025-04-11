"""Proximity event processor."""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import numpy as np

from app.models.events import (
    Event, EventType, EventStatus, EventDetection,
    ThreatLevel, CourseOfAction, ProximityDetection
)
from app.services.event_processor_base import EventProcessorBase

logger = logging.getLogger(__name__)

class ProximityProcessor(EventProcessorBase):
    """Processor for proximity events."""
    
    def __init__(self):
        super().__init__()
        self.event_type = EventType.PROXIMITY
    
    async def detect_entry_criteria(self, data: Dict[str, Any]) -> Optional[EventDetection]:
        """
        Check if data meets entry criteria for proximity events.
        
        Entry criteria:
        1. Two objects with predicted close approach below threshold
        2. Confidence above threshold
        
        Args:
            data: Sensor data
            
        Returns:
            EventDetection if criteria met, None otherwise
        """
        # Extract required fields
        if not all(key in data for key in ["primary_object_id", "secondary_object_id", "minimum_distance", "confidence"]):
            return None
        
        # Check if minimum distance is below threshold (e.g., 10 km)
        if data["minimum_distance"] > 10000:  # 10km in meters
            return None
        
        # Check confidence
        if data["confidence"] < 0.7:
            return None
        
        # Create detection
        return EventDetection(
            event_type=EventType.PROXIMITY,
            object_id=data["primary_object_id"],  # Use primary object as the main ID
            detection_time=datetime.utcnow(),
            confidence=data["confidence"],
            sensor_data={
                "primary_object_id": data["primary_object_id"],
                "secondary_object_id": data["secondary_object_id"],
                "minimum_distance": data["minimum_distance"],
                "relative_velocity": data.get("relative_velocity", 0),
                "closest_approach_time": data.get("closest_approach_time", datetime.utcnow().isoformat()),
                "radial_separation": data.get("radial_separation", 0),
                "in_track_separation": data.get("in_track_separation", 0),
                "cross_track_separation": data.get("cross_track_separation", 0),
                "sensor_id": data.get("sensor_id", "unknown")
            }
        )
    
    async def process_event(self, event: Event) -> Event:
        """
        Process a proximity event.
        
        Processing steps:
        1. Update status to PROCESSING
        2. Classify proximity type
        3. Analyze object characteristics
        4. Assess conjunction risk
        5. Assess hostility
        6. Generate COA
        7. Update status to COMPLETED
        
        Args:
            event: The event to process
            
        Returns:
            Updated event
        """
        # Update status
        event.status = EventStatus.PROCESSING
        event = await self.add_processing_step(
            event=event,
            step_name="start_processing",
            status="success"
        )
        
        try:
            # Step 1: Classify proximity type
            proximity_classification = await self._classify_proximity(event)
            event = await self.add_processing_step(
                event=event,
                step_name="proximity_classification",
                status="success",
                output=proximity_classification
            )
            
            # Step 2: Analyze involved objects
            object_analysis = await self._analyze_objects(event, proximity_classification)
            event = await self.add_processing_step(
                event=event,
                step_name="object_analysis",
                status="success",
                output=object_analysis
            )
            
            # Step 3: Assess conjunction risk
            conjunction_risk = await self._assess_conjunction_risk(event, object_analysis)
            event = await self.add_processing_step(
                event=event,
                step_name="conjunction_risk",
                status="success",
                output=conjunction_risk
            )
            
            # Step 4: Assess hostility
            hostility_assessment, threat_level = await self.assess_hostility(event)
            event = await self.add_processing_step(
                event=event,
                step_name="hostility_assessment",
                status="success",
                output={"threat_level": threat_level.value}
            )
            
            # Step 5: Generate COA
            coa = await self.generate_coa(event, threat_level)
            
            # Complete event
            event = await self.complete_event(
                event=event,
                threat_level=threat_level,
                hostility_assessment=hostility_assessment,
                coa=coa
            )
            
            return event
            
        except Exception as e:
            logger.error(f"Error processing proximity event: {str(e)}", exc_info=True)
            event.status = EventStatus.ERROR
            event = await self.add_processing_step(
                event=event,
                step_name="error",
                status="failed",
                error=str(e)
            )
            return event
    
    async def _classify_proximity(self, event: Event) -> Dict[str, Any]:
        """
        Classify the type of proximity event.
        
        Classification types:
        - Natural conjunction (random)
        - Intentional rendezvous
        - Intercept trajectory
        - Station-keeping proximity
        - Fly-by
        
        Args:
            event: The proximity event
            
        Returns:
            Classification results
        """
        # Extract relevant data
        sensor_data = event.detection_data.sensor_data
        minimum_distance = sensor_data.get("minimum_distance", 0)  # meters
        relative_velocity = sensor_data.get("relative_velocity", 0)  # m/s
        
        # Component separations
        radial_separation = sensor_data.get("radial_separation", 0)
        in_track_separation = sensor_data.get("in_track_separation", 0)
        cross_track_separation = sensor_data.get("cross_track_separation", 0)
        
        # Perform classification based on distance and velocity
        classifications = {}
        
        # Natural conjunction (random) typically has high relative velocity
        classifications["natural_conjunction"] = 0.9 if relative_velocity > 100 else 0.1
        
        # Intentional rendezvous has low relative velocity and very close approach
        classifications["intentional_rendezvous"] = 0.9 if (relative_velocity < 10 and minimum_distance < 1000) else 0.1
        
        # Intercept trajectory has moderate velocity and very close approach
        classifications["intercept_trajectory"] = 0.9 if (10 < relative_velocity < 100 and minimum_distance < 500) else 0.1
        
        # Station-keeping proximity has very low velocity and moderate distance
        classifications["station_keeping"] = 0.9 if (relative_velocity < 1 and 500 < minimum_distance < 5000) else 0.1
        
        # Fly-by has moderate to high velocity and moderate distance
        classifications["fly_by"] = 0.9 if (relative_velocity > 50 and 500 < minimum_distance < 10000) else 0.1
        
        # Determine primary classification
        primary_classification = max(classifications.items(), key=lambda x: x[1])
        
        return {
            "primary_type": primary_classification[0],
            "confidence": primary_classification[1],
            "classifications": classifications,
            "minimum_distance_m": minimum_distance,
            "relative_velocity_m_s": relative_velocity,
            "proximity_characteristics": {
                "controlled": relative_velocity < 10,
                "high_speed": relative_velocity > 100,
                "extremely_close": minimum_distance < 100,
                "component_separations": {
                    "radial_m": radial_separation,
                    "in_track_m": in_track_separation,
                    "cross_track_m": cross_track_separation
                }
            }
        }
    
    async def _analyze_objects(self, event: Event, 
                           classification_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the objects involved in the proximity event.
        
        Args:
            event: The proximity event
            classification_results: Results from proximity classification
            
        Returns:
            Object analysis
        """
        # Extract relevant data
        sensor_data = event.detection_data.sensor_data
        primary_object_id = sensor_data.get("primary_object_id", "unknown")
        secondary_object_id = sensor_data.get("secondary_object_id", "unknown")
        
        # In a real system, would query space object catalog
        # For now, generate mock data
        
        # Generate mock data for primary object
        primary_object = {
            "object_id": primary_object_id,
            "object_type": np.random.choice(["satellite", "rocket_body", "debris"]),
            "size_class": np.random.choice(["small", "medium", "large"]),
            "owner": np.random.choice(["USA", "Russia", "China", "ESA", "Commercial"]),
            "operational_status": np.random.choice(["active", "inactive", "unknown"]),
            "maneuverable": np.random.choice([True, False]),
            "mission_type": np.random.choice(["comms", "earth_obs", "navigation", "military", "unknown"])
        }
        
        # Generate mock data for secondary object
        secondary_object = {
            "object_id": secondary_object_id,
            "object_type": np.random.choice(["satellite", "rocket_body", "debris"]),
            "size_class": np.random.choice(["small", "medium", "large"]),
            "owner": np.random.choice(["USA", "Russia", "China", "ESA", "Commercial"]),
            "operational_status": np.random.choice(["active", "inactive", "unknown"]),
            "maneuverable": np.random.choice([True, False]),
            "mission_type": np.random.choice(["comms", "earth_obs", "navigation", "military", "unknown"])
        }
        
        # Analyze relationship between objects
        same_owner = primary_object["owner"] == secondary_object["owner"]
        both_active = (primary_object["operational_status"] == "active" and 
                        secondary_object["operational_status"] == "active")
        both_maneuverable = primary_object["maneuverable"] and secondary_object["maneuverable"]
        
        # Determine historical patterns
        # In real system, would query historical database
        historical_proximity = np.random.choice([True, False])
        repeated_pattern = historical_proximity and np.random.choice([True, False])
        
        return {
            "primary_object": primary_object,
            "secondary_object": secondary_object,
            "relationship": {
                "same_owner": same_owner,
                "both_active": both_active,
                "both_maneuverable": both_maneuverable,
                "historical_proximity": historical_proximity,
                "repeated_pattern": repeated_pattern
            },
            "high_value_asset": (
                primary_object["owner"] == "USA" and 
                primary_object["mission_type"] in ["military", "navigation"]
            ) or (
                secondary_object["owner"] == "USA" and 
                secondary_object["mission_type"] in ["military", "navigation"]
            )
        }
    
    async def _assess_conjunction_risk(self, event: Event,
                                   object_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess the risk of the conjunction.
        
        Args:
            event: The proximity event
            object_analysis: Results from object analysis
            
        Returns:
            Conjunction risk assessment
        """
        # Extract relevant data
        sensor_data = event.detection_data.sensor_data
        minimum_distance = sensor_data.get("minimum_distance", 0)
        relative_velocity = sensor_data.get("relative_velocity", 0)
        
        primary_object = object_analysis.get("primary_object", {})
        secondary_object = object_analysis.get("secondary_object", {})
        
        # In a real system, would use precise collision probability calculation
        # For now, use a simplified model
        
        # Estimate object sizes based on size class
        size_map = {"small": 0.5, "medium": 2.0, "large": 5.0}
        primary_size = size_map.get(primary_object.get("size_class", "small"), 0.5)
        secondary_size = size_map.get(secondary_object.get("size_class", "small"), 0.5)
        
        # Combined object radii
        combined_radius = primary_size + secondary_size
        
        # Calculate miss distance to combined radius ratio
        miss_ratio = minimum_distance / (combined_radius * 1000)  # Convert size to meters
        
        # Calculate collision probability using simple model
        # P = 1 / (miss_ratio^2)
        if miss_ratio > 0:
            collision_probability = min(1.0, 1 / (miss_ratio * miss_ratio))
        else:
            collision_probability = 1.0
            
        # Calculate potential impact energy (kinetic energy)
        # Estimate masses based on size (very rough approximation)
        primary_mass = primary_size**3 * 100  # kg, assuming average density
        secondary_mass = secondary_size**3 * 100  # kg
        
        # Reduced mass
        reduced_mass = (primary_mass * secondary_mass) / (primary_mass + secondary_mass)
        
        # Kinetic energy at impact (joules)
        impact_energy = 0.5 * reduced_mass * (relative_velocity**2)
        
        # Convert to TNT equivalent (1 kg TNT = 4.184e6 joules)
        tnt_equivalent = impact_energy / 4.184e6
        
        # Assess risk level
        risk_level = "low"
        if collision_probability > 0.1:
            risk_level = "high"
        elif collision_probability > 0.01:
            risk_level = "medium"
        
        # Determine if avoidance maneuver is recommended
        avoidance_recommended = collision_probability > 0.001
        
        # Which object should maneuver (if applicable)
        primary_should_maneuver = False
        secondary_should_maneuver = False
        
        if avoidance_recommended:
            primary_should_maneuver = primary_object.get("maneuverable", False)
            if not primary_should_maneuver:
                secondary_should_maneuver = secondary_object.get("maneuverable", False)
        
        return {
            "collision_probability": collision_probability,
            "miss_ratio": miss_ratio,
            "risk_level": risk_level,
            "impact_energy_joules": impact_energy,
            "tnt_equivalent_kg": tnt_equivalent,
            "avoidance_recommended": avoidance_recommended,
            "avoidance_options": {
                "primary_should_maneuver": primary_should_maneuver,
                "secondary_should_maneuver": secondary_should_maneuver,
                "both_can_maneuver": primary_object.get("maneuverable", False) and 
                                    secondary_object.get("maneuverable", False)
            }
        }
    
    async def assess_hostility(self, event: Event) -> Tuple[Dict[str, Any], ThreatLevel]:
        """
        Assess hostility of a proximity event.
        
        Factors:
        1. Proximity type
        2. Object characteristics and relationships
        3. Conjunction risk
        4. Historical patterns
        
        Args:
            event: The event to assess
            
        Returns:
            Tuple of (assessment_details, threat_level)
        """
        # Extract data from processing steps
        classification_step = next((s for s in event.processing_steps 
                                   if s.step_name == "proximity_classification"), None)
        object_step = next((s for s in event.processing_steps 
                           if s.step_name == "object_analysis"), None)
        risk_step = next((s for s in event.processing_steps 
                         if s.step_name == "conjunction_risk"), None)
        
        if not all([classification_step, object_step, risk_step]):
            return {"error": "Missing required processing steps"}, ThreatLevel.MODERATE
        
        classification_data = classification_step.output or {}
        object_data = object_step.output or {}
        risk_data = risk_step.output or {}
        
        # Extract key information
        primary_type = classification_data.get("primary_type", "unknown")
        extremely_close = classification_data.get("proximity_characteristics", {}).get("extremely_close", False)
        controlled = classification_data.get("proximity_characteristics", {}).get("controlled", False)
        
        primary_object = object_data.get("primary_object", {})
        secondary_object = object_data.get("secondary_object", {})
        relationship = object_data.get("relationship", {})
        high_value_asset = object_data.get("high_value_asset", False)
        
        risk_level = risk_data.get("risk_level", "low")
        collision_probability = risk_data.get("collision_probability", 0)
        
        # Assess hostility
        hostility_score = 0
        
        # Factor 1: Proximity type
        if primary_type == "intercept_trajectory":
            hostility_score += 3
        elif primary_type == "intentional_rendezvous" and not relationship.get("same_owner", False):
            hostility_score += 2
        elif primary_type == "fly_by" and controlled:
            hostility_score += 1
            
        # Factor 2: Extremely close approach
        if extremely_close:
            hostility_score += 1
            
        # Factor 3: High value asset involved
        if high_value_asset:
            hostility_score += 2
            
        # Factor 4: High collision risk with suspicious characteristics
        if risk_level == "high" and primary_type not in ["natural_conjunction"]:
            hostility_score += 2
            
        # Factor 5: Foreign object approaching US military asset
        if (primary_object.get("owner") == "USA" and primary_object.get("mission_type") == "military" and
            secondary_object.get("owner") not in ["USA", "Commercial", "ESA"]):
            hostility_score += 3
        elif (secondary_object.get("owner") == "USA" and secondary_object.get("mission_type") == "military" and
              primary_object.get("owner") not in ["USA", "Commercial", "ESA"]):
            hostility_score += 3
            
        # Factor 6: Repeated pattern without explanation
        if relationship.get("repeated_pattern", False) and not relationship.get("same_owner", False):
            hostility_score += 1
        
        # Determine threat level
        threat_level = ThreatLevel.NONE
        if hostility_score >= 6:
            threat_level = ThreatLevel.HIGH
        elif hostility_score >= 4:
            threat_level = ThreatLevel.MODERATE
        elif hostility_score >= 2:
            threat_level = ThreatLevel.LOW
        
        # Create assessment details
        assessment = {
            "hostility_score": hostility_score,
            "factors": {
                "proximity_type": primary_type,
                "extremely_close": extremely_close,
                "controlled_approach": controlled,
                "high_value_asset": high_value_asset,
                "risk_level": risk_level,
                "collision_probability": collision_probability,
                "same_owner": relationship.get("same_owner", False),
                "repeated_pattern": relationship.get("repeated_pattern", False)
            },
            "assessment_time": datetime.utcnow().isoformat()
        }
        
        return assessment, threat_level
    
    async def generate_coa(self, event: Event, threat_level: ThreatLevel) -> CourseOfAction:
        """
        Generate course of action for a proximity event.
        
        Args:
            event: The event to generate COA for
            threat_level: Assessed threat level
            
        Returns:
            Recommended course of action
        """
        # Extract data from processing steps
        classification_step = next((s for s in event.processing_steps 
                                   if s.step_name == "proximity_classification"), None)
        object_step = next((s for s in event.processing_steps 
                           if s.step_name == "object_analysis"), None)
        risk_step = next((s for s in event.processing_steps 
                         if s.step_name == "conjunction_risk"), None)
        
        classification_data = classification_step.output if classification_step else {}
        object_data = object_step.output if object_step else {}
        risk_data = risk_step.output if risk_step else {}
        
        # Extract key information
        primary_type = classification_data.get("primary_type", "unknown")
        primary_object = object_data.get("primary_object", {})
        secondary_object = object_data.get("secondary_object", {})
        avoidance_recommended = risk_data.get("avoidance_recommended", False)
        primary_should_maneuver = risk_data.get("avoidance_options", {}).get("primary_should_maneuver", False)
        secondary_should_maneuver = risk_data.get("avoidance_options", {}).get("secondary_should_maneuver", False)
        
        # Set COA priority based on threat level
        priority = 1  # Default low priority
        if threat_level == ThreatLevel.HIGH:
            priority = 5
        elif threat_level == ThreatLevel.MODERATE:
            priority = 3
        elif threat_level == ThreatLevel.LOW:
            priority = 2
        
        # Generate actions based on threat level and proximity type
        actions = ["Continue monitoring both objects"]
        
        # Add tracking actions
        actions.append("Increase tracking frequency during close approach period")
        
        # Add collision avoidance actions if applicable
        if avoidance_recommended:
            if primary_should_maneuver and primary_object.get("owner") == "USA":
                actions.append(f"Recommend avoidance maneuver for {primary_object.get('object_id')}")
            elif secondary_should_maneuver and secondary_object.get("owner") == "USA":
                actions.append(f"Recommend avoidance maneuver for {secondary_object.get('object_id')}")
            else:
                actions.append("Notify operators of collision risk")
        
        # Add specific actions based on classification
        if primary_type == "intentional_rendezvous" and not object_data.get("relationship", {}).get("same_owner", False):
            actions.append("Analyze approach trajectory for indications of intent")
            actions.append("Monitor for any RF or optical emissions between objects")
            
        elif primary_type == "intercept_trajectory":
            actions.append("Issue alert to spacecraft operators")
            actions.append("Prepare defensive options assessment")
            
        # Add threat-specific actions
        if threat_level == ThreatLevel.MODERATE:
            actions.append("Increase sensor resources allocated to monitoring the event")
            actions.append("Alert SSA Fusion Center of potential hostile activity")
            
        if threat_level == ThreatLevel.HIGH:
            actions.append("Initiate continuous monitoring of both objects")
            actions.append("Prepare diplomatic communication if foreign entities involved")
            actions.append("Evaluate all available defensive options")
            actions.append("Alert national leadership of potential space threat")
            
        # Create COA
        title = f"Proximity Response Plan: {threat_level.value.capitalize()} Threat"
        description = (
            f"Response plan for {primary_type} proximity event between "
            f"{primary_object.get('object_id')} and {secondary_object.get('object_id')}"
        )
        
        # Set expiration
        closest_approach_time = datetime.fromisoformat(
            event.detection_data.sensor_data.get("closest_approach_time", datetime.utcnow().isoformat())
        )
        expiration = closest_approach_time + timedelta(hours=48)
        
        return CourseOfAction(
            title=title,
            description=description,
            priority=priority,
            actions=actions,
            expiration=expiration
        )