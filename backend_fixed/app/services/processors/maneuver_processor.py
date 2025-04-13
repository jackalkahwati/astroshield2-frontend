"""Maneuver event processor."""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import numpy as np

from app.models.events import (
    Event, EventType, EventStatus, EventDetection,
    ThreatLevel, CourseOfAction, ManeuverDetection
)
from app.services.event_processor_base import EventProcessorBase

logger = logging.getLogger(__name__)

class ManeuverProcessor(EventProcessorBase):
    """Processor for maneuver events."""
    
    def __init__(self):
        super().__init__()
        self.event_type = EventType.MANEUVER
    
    async def detect_entry_criteria(self, data: Dict[str, Any]) -> Optional[EventDetection]:
        """
        Check if data meets entry criteria for maneuver events.
        
        Entry criteria:
        1. Change in velocity vector (delta-v) above threshold
        2. Confidence above threshold
        
        Args:
            data: Sensor data
            
        Returns:
            EventDetection if criteria met, None otherwise
        """
        # Extract required fields
        if not all(key in data for key in ["object_id", "delta_v", "confidence"]):
            return None
        
        # Check if delta-v is above threshold (e.g., 1 m/s)
        if data["delta_v"] < 1.0:
            return None
        
        # Check confidence
        if data["confidence"] < 0.7:
            return None
        
        # Extract additional fields
        direction = data.get("direction", [0, 0, 0])
        
        # Create detection
        return EventDetection(
            event_type=EventType.MANEUVER,
            object_id=data["object_id"],
            detection_time=datetime.utcnow(),
            confidence=data["confidence"],
            sensor_data={
                "delta_v": data["delta_v"],
                "direction": direction,
                "maneuver_time": data.get("timestamp", datetime.utcnow().isoformat()),
                "sensor_id": data.get("sensor_id", "unknown"),
                "initial_orbit": data.get("initial_orbit", {}),
                "final_orbit": data.get("final_orbit", {})
            }
        )
    
    async def process_event(self, event: Event) -> Event:
        """
        Process a maneuver event.
        
        Processing steps:
        1. Update status to PROCESSING
        2. Classify maneuver type
        3. Determine purpose assessment
        4. Identify potential targets
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
            # Step 1: Classify maneuver type
            maneuver_classification = await self._classify_maneuver(event)
            event = await self.add_processing_step(
                event=event,
                step_name="maneuver_classification",
                status="success",
                output=maneuver_classification
            )
            
            # Step 2: Determine purpose assessment
            purpose_assessment = await self._assess_purpose(event, maneuver_classification)
            event = await self.add_processing_step(
                event=event,
                step_name="purpose_assessment",
                status="success",
                output=purpose_assessment
            )
            
            # Step 3: Identify potential targets
            target_analysis = await self._analyze_potential_targets(event, purpose_assessment)
            event = await self.add_processing_step(
                event=event,
                step_name="target_analysis",
                status="success",
                output=target_analysis
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
            logger.error(f"Error processing maneuver event: {str(e)}", exc_info=True)
            event.status = EventStatus.ERROR
            event = await self.add_processing_step(
                event=event,
                step_name="error",
                status="failed",
                error=str(e)
            )
            return event
    
    async def _classify_maneuver(self, event: Event) -> Dict[str, Any]:
        """
        Classify the type of maneuver.
        
        Classification types:
        - Station-keeping
        - Orbit raising/lowering
        - Plane change
        - Phasing
        - Rendezvous
        - Avoidance
        - Deorbit
        
        Args:
            event: The maneuver event
            
        Returns:
            Classification results
        """
        # Extract relevant data
        sensor_data = event.detection_data.sensor_data
        delta_v = sensor_data.get("delta_v", 0)
        direction = sensor_data.get("direction", [0, 0, 0])
        initial_orbit = sensor_data.get("initial_orbit", {})
        final_orbit = sensor_data.get("final_orbit", {})
        
        # Calculate direction magnitudes (in-track, cross-track, radial)
        # This is a simplification - real implementation would use orbital mechanics
        in_track_component = abs(direction[0]) if len(direction) > 0 else 0
        cross_track_component = abs(direction[1]) if len(direction) > 1 else 0
        radial_component = abs(direction[2]) if len(direction) > 2 else 0
        
        # Normalize components
        total = in_track_component + cross_track_component + radial_component
        if total > 0:
            in_track_ratio = in_track_component / total
            cross_track_ratio = cross_track_component / total
            radial_ratio = radial_component / total
        else:
            in_track_ratio = cross_track_ratio = radial_ratio = 0
        
        # Perform classification based on delta-v magnitude and direction
        classifications = {}
        
        # Station-keeping typically has low delta-v
        classifications["station_keeping"] = 0.9 if delta_v < 1.0 else 0.1
        
        # Orbit raising/lowering has high radial component
        classifications["orbit_change"] = 0.9 if radial_ratio > 0.6 else 0.1
        
        # Plane change has high cross-track component
        classifications["plane_change"] = 0.9 if cross_track_ratio > 0.6 else 0.1
        
        # Phasing has high in-track component
        classifications["phasing"] = 0.9 if in_track_ratio > 0.7 else 0.1
        
        # Rendezvous is harder to classify without target information
        # For now, use a combination of factors
        classifications["rendezvous"] = 0.7 if (in_track_ratio > 0.4 and delta_v > 5.0) else 0.1
        
        # Avoidance maneuvers are typically sudden with moderate delta-v
        classifications["avoidance"] = 0.8 if (delta_v > 0.5 and delta_v < 10.0) else 0.2
        
        # Deorbit has high retrograde in-track component
        # This is a simplification - would need velocity vector direction
        classifications["deorbit"] = 0.9 if (delta_v > 50.0 and in_track_ratio > 0.7) else 0.1
        
        # Determine primary classification
        primary_classification = max(classifications.items(), key=lambda x: x[1])
        
        return {
            "primary_type": primary_classification[0],
            "confidence": primary_classification[1],
            "classifications": classifications,
            "delta_v": delta_v,
            "direction_components": {
                "in_track": in_track_ratio,
                "cross_track": cross_track_ratio,
                "radial": radial_ratio
            }
        }
    
    async def _assess_purpose(self, event: Event, classification_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess the purpose of the maneuver.
        
        Args:
            event: The maneuver event
            classification_results: Results from maneuver classification
            
        Returns:
            Purpose assessment
        """
        # Extract relevant data
        primary_type = classification_results.get("primary_type", "unknown")
        
        # Define purpose mappings
        purpose_mappings = {
            "station_keeping": "maintain_orbit",
            "orbit_change": "change_altitude",
            "plane_change": "change_inclination",
            "phasing": "adjust_position",
            "rendezvous": "approach_target",
            "avoidance": "avoid_collision",
            "deorbit": "end_of_life"
        }
        
        standard_purpose = purpose_mappings.get(primary_type, "unknown")
        
        # Calculate benign vs. suspicious score
        benign_score = 0.8
        suspicious_score = 0.2
        
        # Adjust scores based on maneuver type
        if primary_type in ["rendezvous"]:
            benign_score = 0.6
            suspicious_score = 0.4
        elif primary_type in ["avoidance", "station_keeping"]:
            benign_score = 0.9
            suspicious_score = 0.1
        
        # Check for unusual characteristics that might indicate suspicious purpose
        delta_v = classification_results.get("delta_v", 0)
        
        # Unusually large delta-v might be suspicious
        if delta_v > 100:
            benign_score *= 0.8
            suspicious_score = 1 - benign_score
        
        # Determine overall assessment
        if benign_score > suspicious_score:
            overall_assessment = "nominal"
        else:
            overall_assessment = "suspicious"
        
        return {
            "standard_purpose": standard_purpose,
            "benign_score": benign_score,
            "suspicious_score": suspicious_score,
            "overall_assessment": overall_assessment,
            "unusual_characteristics": {
                "high_delta_v": delta_v > 100,
                "rapid_execution": False  # Would be determined from actual maneuver timing
            }
        }
    
    async def _analyze_potential_targets(self, event: Event, purpose_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze potential targets of the maneuver.
        
        Args:
            event: The maneuver event
            purpose_assessment: Results from purpose assessment
            
        Returns:
            Target analysis results
        """
        # In a real implementation, this would query a space object catalog
        # For now, generate mock data
        
        # Only perform target analysis for certain purposes
        standard_purpose = purpose_assessment.get("standard_purpose", "unknown")
        if standard_purpose not in ["approach_target", "avoid_collision"]:
            return {
                "has_potential_targets": False,
                "target_count": 0,
                "targets": []
            }
        
        # Mock target generation
        target_count = np.random.randint(0, 3)
        targets = []
        
        for i in range(target_count):
            target_id = f"SAT-{np.random.randint(1000, 9999)}"
            approach_distance_km = np.random.uniform(0.1, 50.0)
            relative_velocity_km_s = np.random.uniform(0.01, 1.0)
            
            targets.append({
                "object_id": target_id,
                "approach_distance_km": approach_distance_km,
                "relative_velocity_km_s": relative_velocity_km_s,
                "estimated_closest_approach": (datetime.utcnow() + timedelta(hours=np.random.uniform(1, 24))).isoformat(),
                "object_type": np.random.choice(["satellite", "rocket_body", "debris"]),
                "owner": np.random.choice(["USA", "Russia", "China", "ESA", "Commercial"])
            })
        
        return {
            "has_potential_targets": target_count > 0,
            "target_count": target_count,
            "targets": targets,
            "closest_approach_km": min([t["approach_distance_km"] for t in targets]) if targets else None
        }
    
    async def assess_hostility(self, event: Event) -> Tuple[Dict[str, Any], ThreatLevel]:
        """
        Assess hostility of a maneuver event.
        
        Factors:
        1. Maneuver type and purpose
        2. Potential targets
        3. Historical behavior
        4. Delta-v characteristics
        
        Args:
            event: The event to assess
            
        Returns:
            Tuple of (assessment_details, threat_level)
        """
        # Extract data from processing steps
        classification_step = next((s for s in event.processing_steps 
                                   if s.step_name == "maneuver_classification"), None)
        purpose_step = next((s for s in event.processing_steps 
                            if s.step_name == "purpose_assessment"), None)
        target_step = next((s for s in event.processing_steps 
                           if s.step_name == "target_analysis"), None)
        
        if not classification_step or not purpose_step:
            return {"error": "Missing required processing steps"}, ThreatLevel.MODERATE
        
        classification_data = classification_step.output or {}
        purpose_data = purpose_step.output or {}
        target_data = target_step.output or {}
        
        # Extract key information
        primary_type = classification_data.get("primary_type", "unknown")
        standard_purpose = purpose_data.get("standard_purpose", "unknown")
        suspicious_score = purpose_data.get("suspicious_score", 0.2)
        has_potential_targets = target_data.get("has_potential_targets", False)
        closest_approach_km = target_data.get("closest_approach_km", 1000)
        
        # Assess hostility
        hostility_score = 0
        
        # Factor 1: Maneuver type
        if primary_type in ["rendezvous"]:
            hostility_score += 2
        elif primary_type in ["plane_change", "phasing"]:
            hostility_score += 1
            
        # Factor 2: Purpose assessment
        if suspicious_score > 0.6:
            hostility_score += 3
        elif suspicious_score > 0.3:
            hostility_score += 1
            
        # Factor 3: Target proximity
        if has_potential_targets:
            if closest_approach_km < 1:
                hostility_score += 3
            elif closest_approach_km < 5:
                hostility_score += 2
            elif closest_approach_km < 20:
                hostility_score += 1
                
        # Factor 4: Target ownership (if applicable)
        if has_potential_targets:
            for target in target_data.get("targets", []):
                if target.get("owner") in ["USA", "Allied"]:
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
                "maneuver_type": primary_type,
                "standard_purpose": standard_purpose,
                "suspicious_score": suspicious_score,
                "has_potential_targets": has_potential_targets,
                "closest_approach_km": closest_approach_km
            },
            "assessment_time": datetime.utcnow().isoformat()
        }
        
        return assessment, threat_level
    
    async def generate_coa(self, event: Event, threat_level: ThreatLevel) -> CourseOfAction:
        """
        Generate course of action for a maneuver event.
        
        Args:
            event: The event to generate COA for
            threat_level: Assessed threat level
            
        Returns:
            Recommended course of action
        """
        # Extract data from processing steps
        classification_step = next((s for s in event.processing_steps 
                                   if s.step_name == "maneuver_classification"), None)
        target_step = next((s for s in event.processing_steps 
                           if s.step_name == "target_analysis"), None)
        
        classification_data = classification_step.output if classification_step else {}
        target_data = target_step.output if target_step else {}
        
        # Extract key information
        primary_type = classification_data.get("primary_type", "unknown")
        targets = target_data.get("targets", [])
        has_potential_targets = target_data.get("has_potential_targets", False)
        
        # Set COA priority based on threat level
        priority = 1  # Default low priority
        if threat_level == ThreatLevel.HIGH:
            priority = 5
        elif threat_level == ThreatLevel.MODERATE:
            priority = 3
        elif threat_level == ThreatLevel.LOW:
            priority = 2
        
        # Generate actions based on threat level and maneuver type
        actions = ["Continue monitoring spacecraft"]
        
        if primary_type == "rendezvous" and has_potential_targets:
            actions.append("Increase tracking frequency for both objects")
            actions.append("Notify SSA fusion center of potential conjunction")
            
            if threat_level in [ThreatLevel.MODERATE, ThreatLevel.HIGH]:
                actions.append("Alert affected spacecraft operators")
                actions.append("Prepare defensive posture evaluation")
                
            if threat_level == ThreatLevel.HIGH:
                actions.append("Initiate targeted intelligence collection")
                actions.append("Prepare diplomatic channels for potential protest")
                actions.append("Evaluate potential counter-maneuvers")
                
        elif primary_type in ["orbit_change", "plane_change"]:
            actions.append("Update catalog with new orbital parameters")
            
            if threat_level in [ThreatLevel.MODERATE, ThreatLevel.HIGH]:
                actions.append("Assess new orbit for strategic implications")
                actions.append("Verify no conjunction threats in new orbit")
                
        elif primary_type == "avoidance":
            actions.append("Verify target of avoidance maneuver")
            actions.append("Ensure conjunction risk has been mitigated")
            
        # Create COA
        title = f"Maneuver Response Plan: {threat_level.value.capitalize()} Threat"
        description = (
            f"Response plan for {primary_type} maneuver by object {event.object_id}"
        )
        
        # Set expiration
        expiration = datetime.utcnow() + timedelta(hours=24)
        
        return CourseOfAction(
            title=title,
            description=description,
            priority=priority,
            actions=actions,
            expiration=expiration
        )