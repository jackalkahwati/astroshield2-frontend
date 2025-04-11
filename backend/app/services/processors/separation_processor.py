"""Separation event processor."""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import numpy as np

from app.models.events import (
    Event, EventType, EventStatus, EventDetection,
    ThreatLevel, CourseOfAction, SeparationDetection
)
from app.services.event_processor_base import EventProcessorBase

logger = logging.getLogger(__name__)

class SeparationProcessor(EventProcessorBase):
    """Processor for separation events."""
    
    def __init__(self):
        super().__init__()
        self.event_type = EventType.SEPARATION
    
    async def detect_entry_criteria(self, data: Dict[str, Any]) -> Optional[EventDetection]:
        """
        Check if data meets entry criteria for separation events.
        
        Entry criteria:
        1. Detection of new object in proximity to parent object
        2. Relative velocity indicating separation
        3. Confidence above threshold
        
        Args:
            data: Sensor data
            
        Returns:
            EventDetection if criteria met, None otherwise
        """
        # Extract required fields
        if not all(key in data for key in ["parent_object_id", "child_object_id", "relative_velocity", "confidence"]):
            return None
        
        # Check if relative velocity indicates separation (e.g., > 0.1 m/s)
        if data["relative_velocity"] < 0.1:
            return None
        
        # Check confidence
        if data["confidence"] < 0.7:
            return None
        
        # Create detection
        return EventDetection(
            event_type=EventType.SEPARATION,
            object_id=data["parent_object_id"],  # Use parent ID as primary object
            detection_time=datetime.utcnow(),
            confidence=data["confidence"],
            sensor_data={
                "parent_object_id": data["parent_object_id"],
                "child_object_id": data["child_object_id"],
                "relative_velocity": data["relative_velocity"],
                "separation_time": data.get("separation_time", datetime.utcnow().isoformat()),
                "separation_distance": data.get("separation_distance", 0),
                "relative_trajectory": data.get("relative_trajectory", [0, 0, 0]),
                "sensor_id": data.get("sensor_id", "unknown")
            }
        )
    
    async def process_event(self, event: Event) -> Event:
        """
        Process a separation event.
        
        Processing steps:
        1. Update status to PROCESSING
        2. Classify separation type
        3. Analyze child object characteristics
        4. Predict child object trajectory
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
            # Step 1: Classify separation type
            separation_classification = await self._classify_separation(event)
            event = await self.add_processing_step(
                event=event,
                step_name="separation_classification",
                status="success",
                output=separation_classification
            )
            
            # Step 2: Analyze child object characteristics
            child_analysis = await self._analyze_child_object(event, separation_classification)
            event = await self.add_processing_step(
                event=event,
                step_name="child_object_analysis",
                status="success",
                output=child_analysis
            )
            
            # Step 3: Predict child object trajectory
            trajectory_prediction = await self._predict_trajectory(event, child_analysis)
            event = await self.add_processing_step(
                event=event,
                step_name="trajectory_prediction",
                status="success",
                output=trajectory_prediction
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
            logger.error(f"Error processing separation event: {str(e)}", exc_info=True)
            event.status = EventStatus.ERROR
            event = await self.add_processing_step(
                event=event,
                step_name="error",
                status="failed",
                error=str(e)
            )
            return event
    
    async def _classify_separation(self, event: Event) -> Dict[str, Any]:
        """
        Classify the type of separation event.
        
        Classification types:
        - Planned deployment
        - Debris shedding
        - Fragmentation
        - Breakup
        - Payload separation
        
        Args:
            event: The separation event
            
        Returns:
            Classification results
        """
        # Extract relevant data
        sensor_data = event.detection_data.sensor_data
        relative_velocity = sensor_data.get("relative_velocity", 0)
        separation_distance = sensor_data.get("separation_distance", 0)
        
        # Perform classification based on relative velocity and other factors
        classifications = {}
        
        # Planned deployments typically have low relative velocity (0.1-2 m/s)
        classifications["planned_deployment"] = 0.9 if 0.1 <= relative_velocity <= 2.0 else 0.1
        
        # Debris shedding typically has very low relative velocity
        classifications["debris_shedding"] = 0.8 if relative_velocity < 0.3 else 0.1
        
        # Fragmentation has higher relative velocity
        classifications["fragmentation"] = 0.9 if relative_velocity > 5.0 else 0.1
        
        # Breakup has very high relative velocity
        classifications["breakup"] = 0.9 if relative_velocity > 10.0 else 0.1
        
        # Payload separation is typically well-controlled with moderate velocity
        classifications["payload_separation"] = 0.9 if 1.0 <= relative_velocity <= 3.0 else 0.1
        
        # Determine primary classification
        primary_classification = max(classifications.items(), key=lambda x: x[1])
        
        return {
            "primary_type": primary_classification[0],
            "confidence": primary_classification[1],
            "classifications": classifications,
            "relative_velocity": relative_velocity,
            "separation_distance": separation_distance,
            "separation_characteristics": {
                "controlled": relative_velocity < 5.0,
                "violent": relative_velocity > 10.0,
                "multiple_objects": False  # In a real system, would be determined from sensor data
            }
        }
    
    async def _analyze_child_object(self, event: Event, 
                                 classification_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze characteristics of the child object.
        
        Args:
            event: The separation event
            classification_results: Results from separation classification
            
        Returns:
            Child object analysis
        """
        # Extract relevant data
        sensor_data = event.detection_data.sensor_data
        child_object_id = sensor_data.get("child_object_id", "unknown")
        primary_type = classification_results.get("primary_type", "unknown")
        
        # In a real system, would query space object catalog
        # For now, generate mock data
        
        # Estimate size based on separation type
        estimated_size = 0.0
        if primary_type == "planned_deployment":
            estimated_size = np.random.uniform(0.1, 3.0)  # Small satellite (0.1-3m)
        elif primary_type == "payload_separation":
            estimated_size = np.random.uniform(1.0, 5.0)  # Medium satellite (1-5m)
        elif primary_type == "debris_shedding":
            estimated_size = np.random.uniform(0.01, 0.2)  # Small debris (1-20cm)
        elif primary_type in ["fragmentation", "breakup"]:
            estimated_size = np.random.uniform(0.05, 0.5)  # Fragment (5-50cm)
            
        # Determine if object appears to have control capability
        has_control_capability = False
        if primary_type in ["planned_deployment", "payload_separation"]:
            has_control_capability = np.random.random() > 0.2  # 80% chance for controlled separations
            
        # Determine if object appears to have propulsion
        has_propulsion = False
        if has_control_capability:
            has_propulsion = np.random.random() > 0.3  # 70% chance if it has control capability
            
        # Determine if object appears to have communication capability
        has_communication = False
        if has_control_capability:
            has_communication = np.random.random() > 0.1  # 90% chance if it has control capability
        
        # Determine if object has known mission/function
        known_mission = False
        mission_type = "unknown"
        if primary_type in ["planned_deployment", "payload_separation"]:
            known_mission = np.random.random() > 0.3  # 70% chance for planned separations
            if known_mission:
                mission_type = np.random.choice([
                    "earth_observation", "communications", "navigation", 
                    "technology_demonstration", "science", "military"
                ])
                
        return {
            "child_object_id": child_object_id,
            "estimated_size_m": estimated_size,
            "estimated_mass_kg": estimated_size**3 * 100,  # Simple cubic scaling for mass
            "capabilities": {
                "control": has_control_capability,
                "propulsion": has_propulsion,
                "communication": has_communication
            },
            "mission": {
                "known": known_mission,
                "type": mission_type
            },
            "catalog_status": "uncatalogued",  # Initially uncatalogued
            "priority_for_tracking": "high" if has_propulsion else "medium"
        }
    
    async def _predict_trajectory(self, event: Event, 
                              child_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict trajectory of the child object.
        
        Args:
            event: The separation event
            child_analysis: Results from child object analysis
            
        Returns:
            Trajectory prediction
        """
        # Extract relevant data
        sensor_data = event.detection_data.sensor_data
        relative_velocity = sensor_data.get("relative_velocity", 0)
        relative_trajectory = sensor_data.get("relative_trajectory", [0, 0, 0])
        has_propulsion = child_analysis.get("capabilities", {}).get("propulsion", False)
        
        # In a real system, would use orbital mechanics to propagate trajectory
        # For now, generate mock data
        
        # Determine if trajectory crosses paths with any known objects
        crosses_path_with_objects = np.random.random() > 0.7  # 30% chance
        
        # Generate list of potential crossing objects
        crossing_objects = []
        if crosses_path_with_objects:
            num_crossing_objects = np.random.randint(1, 3)
            for i in range(num_crossing_objects):
                object_id = f"SAT-{np.random.randint(1000, 9999)}"
                time_to_crossing = np.random.uniform(1, 72)  # Hours
                distance_at_crossing = np.random.uniform(1, 50)  # km
                
                crossing_objects.append({
                    "object_id": object_id,
                    "time_to_crossing_hours": time_to_crossing,
                    "distance_at_crossing_km": distance_at_crossing
                })
                
        # Determine if trajectory threatens any critical assets
        threatens_critical_assets = False
        threatened_assets = []
        
        if crosses_path_with_objects:
            for obj in crossing_objects:
                if obj["distance_at_crossing_km"] < 10:  # If close approach
                    # 20% chance it's a critical asset if close approach
                    if np.random.random() > 0.8:
                        threatens_critical_assets = True
                        threatened_assets.append({
                            "object_id": obj["object_id"],
                            "asset_type": np.random.choice(["military", "civilian", "commercial"]),
                            "owner": np.random.choice(["USA", "Allied", "Other"])
                        })
        
        # Determine trajectory stability and predictability
        trajectory_predictable = not has_propulsion  # Unpredictable if it has propulsion
        prediction_confidence = 0.9 if trajectory_predictable else 0.5
        
        return {
            "initial_trajectory": {
                "relative_velocity": relative_velocity,
                "relative_direction": relative_trajectory
            },
            "prediction_window_hours": 72,
            "trajectory_predictable": trajectory_predictable,
            "prediction_confidence": prediction_confidence,
            "crosses_path_with_objects": crosses_path_with_objects,
            "crossing_objects": crossing_objects,
            "threatens_critical_assets": threatens_critical_assets,
            "threatened_assets": threatened_assets,
            "long_term_stability": {
                "stable_orbit": np.random.random() > 0.3,  # 70% chance of stable orbit
                "estimated_orbital_lifetime_days": np.random.randint(30, 3650)  # 30 days to 10 years
            }
        }
    
    async def assess_hostility(self, event: Event) -> Tuple[Dict[str, Any], ThreatLevel]:
        """
        Assess hostility of a separation event.
        
        Factors:
        1. Separation type and characteristics
        2. Child object capabilities
        3. Trajectory threats
        4. Prior notification
        
        Args:
            event: The event to assess
            
        Returns:
            Tuple of (assessment_details, threat_level)
        """
        # Extract data from processing steps
        classification_step = next((s for s in event.processing_steps 
                                   if s.step_name == "separation_classification"), None)
        child_step = next((s for s in event.processing_steps 
                          if s.step_name == "child_object_analysis"), None)
        trajectory_step = next((s for s in event.processing_steps 
                               if s.step_name == "trajectory_prediction"), None)
        
        if not all([classification_step, child_step, trajectory_step]):
            return {"error": "Missing required processing steps"}, ThreatLevel.MODERATE
        
        classification_data = classification_step.output or {}
        child_data = child_step.output or {}
        trajectory_data = trajectory_step.output or {}
        
        # Extract key information
        primary_type = classification_data.get("primary_type", "unknown")
        violent_separation = classification_data.get("separation_characteristics", {}).get("violent", False)
        
        has_propulsion = child_data.get("capabilities", {}).get("propulsion", False)
        known_mission = child_data.get("mission", {}).get("known", False)
        mission_type = child_data.get("mission", {}).get("type", "unknown")
        
        threatens_critical_assets = trajectory_data.get("threatens_critical_assets", False)
        threatened_assets = trajectory_data.get("threatened_assets", [])
        
        # Assess hostility
        hostility_score = 0
        
        # Factor 1: Separation type
        if primary_type in ["breakup", "fragmentation"]:
            hostility_score += 2
        elif primary_type == "debris_shedding":
            hostility_score += 1
            
        # Factor 2: Violent separation
        if violent_separation:
            hostility_score += 2
            
        # Factor 3: Unknown mission
        if has_propulsion and not known_mission:
            hostility_score += 2
        elif not known_mission:
            hostility_score += 1
            
        # Factor 4: Military mission
        if mission_type == "military":
            hostility_score += 1
            
        # Factor 5: Threatens critical assets
        if threatens_critical_assets:
            hostility_score += 3
            
            # Extra points if threatening US or allied assets
            for asset in threatened_assets:
                if asset.get("owner") in ["USA", "Allied"]:
                    hostility_score += 2
                    break
        
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
                "separation_type": primary_type,
                "violent_separation": violent_separation,
                "has_propulsion": has_propulsion,
                "known_mission": known_mission,
                "mission_type": mission_type,
                "threatens_critical_assets": threatens_critical_assets
            },
            "assessment_time": datetime.utcnow().isoformat()
        }
        
        return assessment, threat_level
    
    async def generate_coa(self, event: Event, threat_level: ThreatLevel) -> CourseOfAction:
        """
        Generate course of action for a separation event.
        
        Args:
            event: The event to generate COA for
            threat_level: Assessed threat level
            
        Returns:
            Recommended course of action
        """
        # Extract data from processing steps
        classification_step = next((s for s in event.processing_steps 
                                   if s.step_name == "separation_classification"), None)
        child_step = next((s for s in event.processing_steps 
                          if s.step_name == "child_object_analysis"), None)
        trajectory_step = next((s for s in event.processing_steps 
                               if s.step_name == "trajectory_prediction"), None)
        
        classification_data = classification_step.output if classification_step else {}
        child_data = child_step.output if child_step else {}
        trajectory_data = trajectory_step.output if trajectory_step else {}
        
        # Extract key information
        primary_type = classification_data.get("primary_type", "unknown")
        child_object_id = child_data.get("child_object_id", "unknown")
        threatens_critical_assets = trajectory_data.get("threatens_critical_assets", False)
        
        # Set COA priority based on threat level
        priority = 1  # Default low priority
        if threat_level == ThreatLevel.HIGH:
            priority = 5
        elif threat_level == ThreatLevel.MODERATE:
            priority = 3
        elif threat_level == ThreatLevel.LOW:
            priority = 2
        
        # Generate actions based on threat level and separation type
        actions = [f"Track and catalog new object {child_object_id}"]
        
        # Add tracking actions
        actions.append("Initiate space fence tracking for precise orbital determination")
        actions.append("Update space object catalog with new entry")
        
        # Add specific actions based on classification
        if primary_type in ["planned_deployment", "payload_separation"]:
            actions.append("Correlate with announced deployments/separations")
            actions.append("Monitor for initial operations/activation of new object")
            
        elif primary_type in ["breakup", "fragmentation"]:
            actions.append("Alert all spacecraft operators in vicinity")
            actions.append("Initiate debris field mapping and propagation")
            actions.append("Issue NOTAM for affected orbital regime")
            
        # Add threat-specific actions
        if threatens_critical_assets:
            actions.append("Alert operators of potentially threatened assets")
            actions.append("Prepare collision avoidance options for threatened assets")
            
            if threat_level >= ThreatLevel.MODERATE:
                actions.append("Increase surveillance frequency of new object")
                
        # Add additional actions based on threat level
        if threat_level == ThreatLevel.HIGH:
            actions.append("Initiate persistent surveillance of new object")
            actions.append("Prepare diplomatic inquiries if originating from foreign entity")
            actions.append("Evaluate defensive posture options")
            
        # Create COA
        title = f"Separation Response Plan: {threat_level.value.capitalize()} Threat"
        description = (
            f"Response plan for {primary_type} separation from object {event.object_id} "
            f"resulting in new object {child_object_id}"
        )
        
        # Set expiration
        expiration = datetime.utcnow() + timedelta(hours=48)
        
        return CourseOfAction(
            title=title,
            description=description,
            priority=priority,
            actions=actions,
            expiration=expiration
        )