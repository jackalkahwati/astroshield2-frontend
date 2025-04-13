"""Attitude change event processor."""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import numpy as np

from app.models.events import (
    Event, EventType, EventStatus, EventDetection,
    ThreatLevel, CourseOfAction, AttitudeChangeDetection
)
from app.services.event_processor_base import EventProcessorBase

logger = logging.getLogger(__name__)

class AttitudeChangeProcessor(EventProcessorBase):
    """Processor for attitude change events."""
    
    def __init__(self):
        super().__init__()
        self.event_type = EventType.ATTITUDE_CHANGE
    
    async def detect_entry_criteria(self, data: Dict[str, Any]) -> Optional[EventDetection]:
        """
        Check if data meets entry criteria for attitude change events.
        
        Entry criteria:
        1. Change in attitude (orientation) above threshold
        2. Confidence above threshold
        
        Args:
            data: Sensor data
            
        Returns:
            EventDetection if criteria met, None otherwise
        """
        # Extract required fields
        if not all(key in data for key in ["object_id", "change_magnitude", "previous_attitude", "current_attitude", "confidence"]):
            return None
        
        # Check if change magnitude is above threshold (e.g., 5 degrees)
        if data["change_magnitude"] < 5.0:
            return None
        
        # Check confidence
        if data["confidence"] < 0.7:
            return None
        
        # Create detection
        return EventDetection(
            event_type=EventType.ATTITUDE_CHANGE,
            object_id=data["object_id"],
            detection_time=datetime.utcnow(),
            confidence=data["confidence"],
            sensor_data={
                "change_magnitude": data["change_magnitude"],
                "previous_attitude": data["previous_attitude"],
                "current_attitude": data["current_attitude"],
                "attitude_change_time": data.get("attitude_change_time", datetime.utcnow().isoformat()),
                "change_rate": data.get("change_rate", 0),
                "axis_of_rotation": data.get("axis_of_rotation", [0, 0, 0]),
                "sensor_id": data.get("sensor_id", "unknown")
            }
        )
    
    async def process_event(self, event: Event) -> Event:
        """
        Process an attitude change event.
        
        Processing steps:
        1. Update status to PROCESSING
        2. Classify attitude change type
        3. Analyze change dynamics
        4. Determine operational implications
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
            # Step 1: Classify attitude change type
            attitude_classification = await self._classify_attitude_change(event)
            event = await self.add_processing_step(
                event=event,
                step_name="attitude_classification",
                status="success",
                output=attitude_classification
            )
            
            # Step 2: Analyze change dynamics
            dynamics_analysis = await self._analyze_dynamics(event, attitude_classification)
            event = await self.add_processing_step(
                event=event,
                step_name="dynamics_analysis",
                status="success",
                output=dynamics_analysis
            )
            
            # Step 3: Determine operational implications
            operational_implications = await self._determine_implications(event, attitude_classification, dynamics_analysis)
            event = await self.add_processing_step(
                event=event,
                step_name="operational_implications",
                status="success",
                output=operational_implications
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
            logger.error(f"Error processing attitude change event: {str(e)}", exc_info=True)
            event.status = EventStatus.ERROR
            event = await self.add_processing_step(
                event=event,
                step_name="error",
                status="failed",
                error=str(e)
            )
            return event
    
    async def _classify_attitude_change(self, event: Event) -> Dict[str, Any]:
        """
        Classify the type of attitude change.
        
        Classification types:
        - Normal slew (mission-related pointing change)
        - Spin-up/down
        - Tumble onset
        - Scan pattern
        - Sensor pointing
        - Solar array adjustment
        
        Args:
            event: The attitude change event
            
        Returns:
            Classification results
        """
        # Extract relevant data
        sensor_data = event.detection_data.sensor_data
        change_magnitude = sensor_data.get("change_magnitude", 0)  # degrees
        change_rate = sensor_data.get("change_rate", 0)  # degrees/second
        previous_attitude = sensor_data.get("previous_attitude", [0, 0, 0, 0])  # quaternion
        current_attitude = sensor_data.get("current_attitude", [0, 0, 0, 0])  # quaternion
        axis_of_rotation = sensor_data.get("axis_of_rotation", [0, 0, 0])
        
        # Perform classification based on change characteristics
        classifications = {}
        
        # Normal slew typically has moderate change rate and specific target orientation
        classifications["normal_slew"] = 0.8 if (0.1 < change_rate < 2.0) else 0.1
        
        # Spin-up/down has change primarily around a single axis with increasing/decreasing rate
        is_single_axis_dominant = np.max(np.abs(axis_of_rotation)) > 0.8 if len(axis_of_rotation) > 0 else False
        classifications["spin_up"] = 0.8 if (is_single_axis_dominant and change_rate > 1.0) else 0.1
        classifications["spin_down"] = 0.8 if (is_single_axis_dominant and change_rate < -0.5) else 0.1
        
        # Tumble onset has high rate without a dominant axis
        is_tumbling = not is_single_axis_dominant and abs(change_rate) > 2.0
        classifications["tumble_onset"] = 0.9 if is_tumbling else 0.1
        
        # Scan pattern involves regular back-and-forth motion (hard to detect from a single change)
        classifications["scan_pattern"] = 0.3  # Low confidence without multiple observations
        
        # Sensor pointing typically involves precise, small adjustments
        classifications["sensor_pointing"] = 0.8 if (0.1 < change_rate < 0.5 and change_magnitude < 20) else 0.1
        
        # Solar array adjustment typically involves rotation around specific axis
        # For simplification, assume z-axis is often used for solar array adjustments
        is_likely_solar_adjustment = (
            is_single_axis_dominant and 
            np.argmax(np.abs(axis_of_rotation)) == 2 and  # Z-axis dominant
            change_magnitude < 30
        )
        classifications["solar_array_adjustment"] = 0.8 if is_likely_solar_adjustment else 0.1
        
        # Determine primary classification
        primary_classification = max(classifications.items(), key=lambda x: x[1])
        
        return {
            "primary_type": primary_classification[0],
            "confidence": primary_classification[1],
            "classifications": classifications,
            "change_magnitude_deg": change_magnitude,
            "change_rate_deg_s": change_rate,
            "change_characteristics": {
                "is_high_rate": change_rate > 2.0,
                "is_controlled": 0.01 < change_rate < 2.0,
                "is_single_axis": is_single_axis_dominant,
                "dominant_axis": np.argmax(np.abs(axis_of_rotation)) if is_single_axis_dominant else None,
                "is_tumbling": is_tumbling
            }
        }
    
    async def _analyze_dynamics(self, event: Event, 
                            classification_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the dynamics of the attitude change.
        
        Args:
            event: The attitude change event
            classification_results: Results from attitude classification
            
        Returns:
            Dynamics analysis
        """
        # Extract relevant data
        sensor_data = event.detection_data.sensor_data
        change_magnitude = sensor_data.get("change_magnitude", 0)
        change_rate = sensor_data.get("change_rate", 0)
        primary_type = classification_results.get("primary_type", "unknown")
        is_controlled = classification_results.get("change_characteristics", {}).get("is_controlled", True)
        is_tumbling = classification_results.get("change_characteristics", {}).get("is_tumbling", False)
        
        # In a real system, would use physical models for attitude dynamics
        # For now, generate simplified analysis
        
        # Estimate energy required for the change
        # This is a simplification - real calculation would use moment of inertia
        estimated_energy = change_magnitude * change_rate * 0.1  # arbitrary units
        
        # Determine if change appears controlled or uncontrolled
        # Controlled changes have smooth profiles and target specific orientations
        
        # Estimate time to complete the change
        if change_rate > 0:
            est_time_to_complete = change_magnitude / change_rate  # seconds
        else:
            est_time_to_complete = None
            
        # Determine if attitude is stable after change
        # In a real system, would analyze post-change attitude data over time
        is_stable_after = not is_tumbling and primary_type not in ["tumble_onset", "spin_up"]
        
        # Determine if consistent with nominal operations
        is_consistent_with_nominal = primary_type in ["normal_slew", "sensor_pointing", "solar_array_adjustment"]
        
        # Estimate control authority remaining
        # In tumbling cases or uncontrolled changes, control authority may be compromised
        control_authority_remaining = 1.0  # full
        if is_tumbling:
            control_authority_remaining = 0.1  # minimal
        elif not is_controlled:
            control_authority_remaining = 0.5  # partial
            
        # Determine if change is within operational envelope
        # This would normally be determined by spacecraft specifications
        is_within_envelope = (
            is_controlled and 
            not is_tumbling and 
            change_rate < 3.0 and
            change_magnitude < 180
        )
        
        return {
            "estimated_energy": estimated_energy,
            "estimated_time_to_complete_s": est_time_to_complete,
            "stability_assessment": {
                "is_stable_after": is_stable_after,
                "is_consistent_with_nominal": is_consistent_with_nominal,
                "control_authority_remaining": control_authority_remaining,
                "is_within_operational_envelope": is_within_envelope,
                "recovery_likelihood": 0.9 if is_stable_after else (0.5 if is_controlled else 0.1)
            },
            "attitude_control_system": {
                "appears_functional": is_controlled and not is_tumbling,
                "performance_nominal": is_within_envelope,
                "fuel_consumption_estimate": "high" if change_magnitude > 90 else "moderate" if change_magnitude > 30 else "low"
            }
        }
    
    async def _determine_implications(self, event: Event, 
                                  classification_results: Dict[str, Any],
                                  dynamics_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine operational implications of the attitude change.
        
        Args:
            event: The attitude change event
            classification_results: Results from attitude classification
            dynamics_analysis: Results from dynamics analysis
            
        Returns:
            Operational implications
        """
        # Extract relevant data
        object_id = event.object_id
        primary_type = classification_results.get("primary_type", "unknown")
        is_tumbling = classification_results.get("change_characteristics", {}).get("is_tumbling", False)
        is_stable_after = dynamics_analysis.get("stability_assessment", {}).get("is_stable_after", True)
        is_within_envelope = dynamics_analysis.get("stability_assessment", {}).get("is_within_operational_envelope", True)
        
        # In a real system, would query spacecraft database for capabilities and mission
        # For now, generate mock data
        
        # Generate mock spacecraft data
        spacecraft_data = {
            "object_id": object_id,
            "object_type": np.random.choice(["satellite", "space_station", "unknown"]),
            "owner": np.random.choice(["USA", "Russia", "China", "ESA", "Commercial"]),
            "operational_status": np.random.choice(["active", "inactive", "unknown"]),
            "mission_type": np.random.choice(["comms", "earth_obs", "navigation", "military", "unknown"]),
            "capabilities": {
                "imaging": np.random.choice([True, False]),
                "communications": np.random.choice([True, False]),
                "signals_intelligence": np.random.choice([True, False]),
                "radar": np.random.choice([True, False])
            }
        }
        
        # Determine impact on spacecraft functions
        functional_impacts = {}
        
        # Power generation impact
        if is_tumbling:
            functional_impacts["power_generation"] = "severe"
        elif not is_stable_after:
            functional_impacts["power_generation"] = "moderate"
        elif primary_type == "solar_array_adjustment":
            functional_impacts["power_generation"] = "improved"
        else:
            functional_impacts["power_generation"] = "nominal"
            
        # Communications impact
        if is_tumbling:
            functional_impacts["communications"] = "severe"
        elif not is_stable_after:
            functional_impacts["communications"] = "moderate"
        elif primary_type == "normal_slew" and spacecraft_data["capabilities"]["communications"]:
            functional_impacts["communications"] = "changed_target"
        else:
            functional_impacts["communications"] = "nominal"
            
        # Imaging/sensing impact
        if is_tumbling:
            functional_impacts["sensing"] = "severe"
        elif not is_stable_after:
            functional_impacts["sensing"] = "moderate"
        elif primary_type in ["normal_slew", "sensor_pointing"] and any([
            spacecraft_data["capabilities"]["imaging"],
            spacecraft_data["capabilities"]["signals_intelligence"],
            spacecraft_data["capabilities"]["radar"]
        ]):
            functional_impacts["sensing"] = "changed_target"
        else:
            functional_impacts["sensing"] = "nominal"
            
        # Determine mission impact
        mission_impact = "none"
        if is_tumbling:
            mission_impact = "mission_failure"
        elif not is_stable_after:
            mission_impact = "significant_degradation"
        elif not is_within_envelope:
            mission_impact = "minor_degradation"
        elif primary_type in ["normal_slew", "sensor_pointing"] and spacecraft_data["mission_type"] in ["earth_obs", "military"]:
            mission_impact = "new_targeting"
            
        # Determine potential new targets if applicable
        potential_targets = []
        if mission_impact == "new_targeting":
            # In a real system, would analyze pointing direction against ground/space targets
            # For demo, generate random possibilities
            if np.random.random() > 0.7:  # 30% chance of having identified targets
                target_type = "ground" if np.random.random() > 0.5 else "space"
                
                if target_type == "ground":
                    potential_targets.append({
                        "type": "ground",
                        "region": np.random.choice(["North America", "Europe", "Asia", "Middle East", "Pacific"])
                    })
                else:
                    potential_targets.append({
                        "type": "space",
                        "object_type": np.random.choice(["satellite", "space_station", "debris"])
                    })
        
        return {
            "spacecraft_data": spacecraft_data,
            "functional_impacts": functional_impacts,
            "mission_impact": mission_impact,
            "potential_targets": potential_targets,
            "operational_assessment": {
                "likely_operational_change": mission_impact in ["new_targeting"],
                "likely_anomaly": mission_impact in ["minor_degradation", "significant_degradation", "mission_failure"],
                "likely_end_of_life": mission_impact == "mission_failure" and not is_within_envelope
            }
        }
    
    async def assess_hostility(self, event: Event) -> Tuple[Dict[str, Any], ThreatLevel]:
        """
        Assess hostility of an attitude change event.
        
        Factors:
        1. Type of attitude change
        2. Operational implications
        3. Targeting changes
        4. Spacecraft characteristics
        
        Args:
            event: The event to assess
            
        Returns:
            Tuple of (assessment_details, threat_level)
        """
        # Extract data from processing steps
        classification_step = next((s for s in event.processing_steps 
                                   if s.step_name == "attitude_classification"), None)
        dynamics_step = next((s for s in event.processing_steps 
                             if s.step_name == "dynamics_analysis"), None)
        implications_step = next((s for s in event.processing_steps 
                                 if s.step_name == "operational_implications"), None)
        
        if not all([classification_step, dynamics_step, implications_step]):
            return {"error": "Missing required processing steps"}, ThreatLevel.MODERATE
        
        classification_data = classification_step.output or {}
        dynamics_data = dynamics_step.output or {}
        implications_data = implications_step.output or {}
        
        # Extract key information
        primary_type = classification_data.get("primary_type", "unknown")
        is_controlled = classification_data.get("change_characteristics", {}).get("is_controlled", True)
        
        is_stable_after = dynamics_data.get("stability_assessment", {}).get("is_stable_after", True)
        is_consistent_with_nominal = dynamics_data.get("stability_assessment", {}).get("is_consistent_with_nominal", True)
        
        spacecraft_data = implications_data.get("spacecraft_data", {})
        mission_impact = implications_data.get("mission_impact", "none")
        potential_targets = implications_data.get("potential_targets", [])
        
        # Assess hostility
        hostility_score = 0
        
        # Factor 1: Change type and characteristics
        if primary_type in ["sensor_pointing"] and not is_consistent_with_nominal:
            hostility_score += 1
            
        # Factor 2: Operational implications
        if mission_impact == "new_targeting":
            hostility_score += 2
            
        # Factor 3: Potential targets
        for target in potential_targets:
            if target.get("type") == "space" and spacecraft_data.get("mission_type") == "military":
                hostility_score += 2
            elif target.get("type") == "ground" and target.get("region") in ["North America", "Europe"] and spacecraft_data.get("owner") not in ["USA", "ESA", "Commercial"]:
                hostility_score += 3
                
        # Factor 4: Spacecraft characteristics
        is_military = spacecraft_data.get("mission_type") == "military"
        is_foreign = spacecraft_data.get("owner") not in ["USA", "Allied", "Commercial"]
        has_intelligence_capability = spacecraft_data.get("capabilities", {}).get("signals_intelligence", False)
        
        if is_military and is_foreign and has_intelligence_capability and mission_impact == "new_targeting":
            hostility_score += 3
        elif is_military and is_foreign and mission_impact == "new_targeting":
            hostility_score += 2
            
        # Factor 5: Anomalous behavior
        if not is_consistent_with_nominal and is_controlled and is_military and is_foreign:
            hostility_score += 2
            
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
                "change_type": primary_type,
                "is_controlled": is_controlled,
                "is_stable_after": is_stable_after,
                "is_consistent_with_nominal": is_consistent_with_nominal,
                "mission_impact": mission_impact,
                "has_potential_targets": len(potential_targets) > 0,
                "is_military": is_military,
                "is_foreign": is_foreign,
                "has_intelligence_capability": has_intelligence_capability
            },
            "assessment_time": datetime.utcnow().isoformat()
        }
        
        return assessment, threat_level
    
    async def generate_coa(self, event: Event, threat_level: ThreatLevel) -> CourseOfAction:
        """
        Generate course of action for an attitude change event.
        
        Args:
            event: The event to generate COA for
            threat_level: Assessed threat level
            
        Returns:
            Recommended course of action
        """
        # Extract data from processing steps
        classification_step = next((s for s in event.processing_steps 
                                   if s.step_name == "attitude_classification"), None)
        dynamics_step = next((s for s in event.processing_steps 
                             if s.step_name == "dynamics_analysis"), None)
        implications_step = next((s for s in event.processing_steps 
                                 if s.step_name == "operational_implications"), None)
        
        classification_data = classification_step.output if classification_step else {}
        dynamics_data = dynamics_step.output or {}
        implications_data = implications_step.output or {}
        
        # Extract key information
        object_id = event.object_id
        primary_type = classification_data.get("primary_type", "unknown")
        is_tumbling = classification_data.get("change_characteristics", {}).get("is_tumbling", False)
        
        is_stable_after = dynamics_data.get("stability_assessment", {}).get("is_stable_after", True)
        
        spacecraft_data = implications_data.get("spacecraft_data", {})
        mission_impact = implications_data.get("mission_impact", "none")
        functional_impacts = implications_data.get("functional_impacts", {})
        potential_targets = implications_data.get("potential_targets", [])
        
        # Set COA priority based on threat level
        priority = 1  # Default low priority
        if threat_level == ThreatLevel.HIGH:
            priority = 5
        elif threat_level == ThreatLevel.MODERATE:
            priority = 3
        elif threat_level == ThreatLevel.LOW:
            priority = 2
        
        # Generate actions based on threat level and change characteristics
        actions = [f"Continue monitoring attitude of {object_id}"]
        
        # Add tracking actions for all cases
        actions.append("Update space object catalog with new attitude state")
        
        # Add specific actions based on classification
        if is_tumbling:
            actions.append("Monitor for signs of recovery or further degradation")
            actions.append("Assess probability of debris generation")
            
            if spacecraft_data.get("owner") == "USA":
                actions.append("Alert spacecraft operators to attempt recovery")
                
        elif primary_type == "sensor_pointing" or mission_impact == "new_targeting":
            actions.append("Analyze new pointing direction and potential targets")
            
            if len(potential_targets) > 0:
                target_desc = ", ".join([
                    f"{t.get('type')} target in {t.get('region')}" if t.get('type') == "ground" else
                    f"{t.get('object_type')} in space" for t in potential_targets
                ])
                actions.append(f"Monitor interactions with potential {target_desc}")
                
        # Add functional impact actions
        if functional_impacts.get("communications") in ["severe", "moderate"]:
            actions.append("Monitor for changes in communication patterns")
            
        if functional_impacts.get("sensing") == "changed_target":
            actions.append("Correlate with collection requirements and known targets")
            
        # Add threat-specific actions
        if threat_level == ThreatLevel.MODERATE:
            actions.append("Increase collection priority for this target")
            
            if spacecraft_data.get("capabilities", {}).get("signals_intelligence", False):
                actions.append("Monitor for changes in signals intelligence collection")
                
        if threat_level == ThreatLevel.HIGH:
            actions.append("Alert space operations center of suspicious attitude change")
            actions.append("Evaluate defensive posture for potentially targeted assets")
            
            if spacecraft_data.get("mission_type") == "military" and spacecraft_data.get("owner") not in ["USA", "Allied"]:
                actions.append("Initiate continuous monitoring of spacecraft activities")
                actions.append("Prepare diplomatic inquiries regarding spacecraft behavior")
                
        # Create COA
        title = f"Attitude Change Response Plan: {threat_level.value.capitalize()} Threat"
        description = (
            f"Response plan for {primary_type} attitude change of {object_id} "
            f"({spacecraft_data.get('owner', 'unknown')})"
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