"""Link change event processor."""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import numpy as np

from app.models.events import (
    Event, EventType, EventStatus, EventDetection,
    ThreatLevel, CourseOfAction, LinkChangeDetection
)
from app.services.event_processor_base import EventProcessorBase

logger = logging.getLogger(__name__)

class LinkChangeProcessor(EventProcessorBase):
    """Processor for link change events."""
    
    def __init__(self):
        super().__init__()
        self.event_type = EventType.LINK_CHANGE
    
    async def detect_entry_criteria(self, data: Dict[str, Any]) -> Optional[EventDetection]:
        """
        Check if data meets entry criteria for link change events.
        
        Entry criteria:
        1. Detection of change in communications state
        2. Confidence above threshold
        
        Args:
            data: Sensor data
            
        Returns:
            EventDetection if criteria met, None otherwise
        """
        # Extract required fields
        if not all(key in data for key in ["object_id", "link_type", "previous_state", "current_state", "confidence"]):
            return None
        
        # Check if states are different
        if data["previous_state"] == data["current_state"]:
            return None
        
        # Check confidence
        if data["confidence"] < 0.7:
            return None
        
        # Create detection
        return EventDetection(
            event_type=EventType.LINK_CHANGE,
            object_id=data["object_id"],
            detection_time=datetime.utcnow(),
            confidence=data["confidence"],
            sensor_data={
                "link_type": data["link_type"],
                "previous_state": data["previous_state"],
                "current_state": data["current_state"],
                "link_change_time": data.get("link_change_time", datetime.utcnow().isoformat()),
                "frequency_band": data.get("frequency_band", "unknown"),
                "signal_characteristics": data.get("signal_characteristics", {}),
                "sensor_id": data.get("sensor_id", "unknown")
            }
        )
    
    async def process_event(self, event: Event) -> Event:
        """
        Process a link change event.
        
        Processing steps:
        1. Update status to PROCESSING
        2. Analyze link change type
        3. Assess operational impact
        4. Identify communication pattern
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
            # Step 1: Analyze link change type
            link_analysis = await self._analyze_link_change(event)
            event = await self.add_processing_step(
                event=event,
                step_name="link_analysis",
                status="success",
                output=link_analysis
            )
            
            # Step 2: Assess operational impact
            operational_impact = await self._assess_operational_impact(event, link_analysis)
            event = await self.add_processing_step(
                event=event,
                step_name="operational_impact",
                status="success",
                output=operational_impact
            )
            
            # Step 3: Identify communication pattern
            comm_pattern = await self._identify_comm_pattern(event, link_analysis)
            event = await self.add_processing_step(
                event=event,
                step_name="comm_pattern",
                status="success",
                output=comm_pattern
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
            logger.error(f"Error processing link change event: {str(e)}", exc_info=True)
            event.status = EventStatus.ERROR
            event = await self.add_processing_step(
                event=event,
                step_name="error",
                status="failed",
                error=str(e)
            )
            return event
    
    async def _analyze_link_change(self, event: Event) -> Dict[str, Any]:
        """
        Analyze the type of link change.
        
        Classification types:
        - Signal activation
        - Signal deactivation
        - Modulation change
        - Frequency shift
        - Encryption change
        - Power increase/decrease
        
        Args:
            event: The link change event
            
        Returns:
            Analysis results
        """
        # Extract relevant data
        sensor_data = event.detection_data.sensor_data
        link_type = sensor_data.get("link_type", "unknown")
        previous_state = sensor_data.get("previous_state", "inactive")
        current_state = sensor_data.get("current_state", "active")
        frequency_band = sensor_data.get("frequency_band", "unknown")
        signal_characteristics = sensor_data.get("signal_characteristics", {})
        
        # Determine basic change type
        change_type = "unknown"
        if previous_state == "inactive" and current_state == "active":
            change_type = "activation"
        elif previous_state == "active" and current_state == "inactive":
            change_type = "deactivation"
        else:
            change_type = "modification"
            
        # Check for specific changes in signal characteristics
        specific_changes = []
        
        if "modulation" in signal_characteristics:
            prev_mod = signal_characteristics.get("previous_modulation")
            curr_mod = signal_characteristics.get("current_modulation")
            if prev_mod != curr_mod:
                specific_changes.append("modulation_change")
                
        if "frequency" in signal_characteristics:
            prev_freq = signal_characteristics.get("previous_frequency")
            curr_freq = signal_characteristics.get("current_frequency")
            if prev_freq != curr_freq:
                specific_changes.append("frequency_shift")
                
        if "encryption" in signal_characteristics:
            prev_enc = signal_characteristics.get("previous_encryption")
            curr_enc = signal_characteristics.get("current_encryption")
            if prev_enc != curr_enc:
                specific_changes.append("encryption_change")
                
        if "power" in signal_characteristics:
            prev_power = signal_characteristics.get("previous_power")
            curr_power = signal_characteristics.get("current_power")
            if prev_power < curr_power:
                specific_changes.append("power_increase")
            elif prev_power > curr_power:
                specific_changes.append("power_decrease")
                
        # Determine if change was scheduled/expected
        # In a real system, would check against scheduled communication plans
        # For now, generate random assessment
        is_scheduled = np.random.random() > 0.3  # 70% of link changes are scheduled
        
        # Analyze timing patterns
        # In a real system, would analyze historical communication patterns
        time_of_day = datetime.utcnow().hour
        is_typical_time = 9 <= time_of_day <= 17  # Business hours
        
        return {
            "basic_change_type": change_type,
            "specific_changes": specific_changes,
            "link_type": link_type,
            "frequency_band": frequency_band,
            "assessment": {
                "is_scheduled": is_scheduled,
                "is_typical_time": is_typical_time,
                "is_standard_protocol": np.random.random() > 0.2,  # 80% follow standard protocols
                "is_encrypted": "encryption_change" in specific_changes or 
                               signal_characteristics.get("current_encryption", False)
            }
        }
    
    async def _assess_operational_impact(self, event: Event, 
                                     link_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess the operational impact of the link change.
        
        Args:
            event: The link change event
            link_analysis: Results from link change analysis
            
        Returns:
            Operational impact assessment
        """
        # Extract relevant data
        sensor_data = event.detection_data.sensor_data
        object_id = event.object_id
        change_type = link_analysis.get("basic_change_type", "unknown")
        
        # In a real system, would query the satellite catalog for object information
        # For now, generate mock assessment
        
        # Generate mock spacecraft data
        spacecraft_data = {
            "object_id": object_id,
            "object_type": np.random.choice(["satellite", "space_station", "unknown"]),
            "owner": np.random.choice(["USA", "Russia", "China", "ESA", "Commercial"]),
            "operational_status": np.random.choice(["active", "inactive", "unknown"]),
            "mission_type": np.random.choice(["comms", "earth_obs", "navigation", "military", "unknown"])
        }
        
        # Assess impact based on change type and spacecraft data
        operational_significance = "nominal"
        mission_impact = "none"
        
        # Deactivation of primary link for active spacecraft is high impact
        if change_type == "deactivation" and spacecraft_data["operational_status"] == "active":
            if sensor_data.get("link_type") == "primary":
                operational_significance = "anomalous"
                mission_impact = "severe"
            else:
                operational_significance = "noteworthy"
                mission_impact = "moderate"
                
        # Activation of link for inactive spacecraft is noteworthy
        elif change_type == "activation" and spacecraft_data["operational_status"] == "inactive":
            operational_significance = "anomalous"
            mission_impact = "significant"
            
        # Encryption changes are noteworthy
        elif "encryption_change" in link_analysis.get("specific_changes", []):
            operational_significance = "noteworthy"
            mission_impact = "moderate"
            
        # Determine if change affects telemetry, command, or payload data
        affected_functions = []
        if sensor_data.get("link_type") == "telem":
            affected_functions.append("telemetry")
        elif sensor_data.get("link_type") == "cmd":
            affected_functions.append("command")
        elif sensor_data.get("link_type") == "payload":
            affected_functions.append("payload_data")
            
        # Determine if redundancy exists
        has_redundancy = np.random.random() > 0.3  # 70% chance of redundant systems
        
        return {
            "spacecraft_data": spacecraft_data,
            "operational_significance": operational_significance,
            "mission_impact": mission_impact,
            "affected_functions": affected_functions,
            "has_redundancy": has_redundancy,
            "estimated_duration": "unknown" if change_type == "deactivation" else "ongoing",
            "recovery_options": [] if mission_impact == "none" else ["activate_backup", "reset_communications"]
        }
    
    async def _identify_comm_pattern(self, event: Event,
                                 link_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify communication patterns and anomalies.
        
        Args:
            event: The link change event
            link_analysis: Results from link change analysis
            
        Returns:
            Communication pattern analysis
        """
        # Extract relevant data
        sensor_data = event.detection_data.sensor_data
        change_type = link_analysis.get("basic_change_type", "unknown")
        is_scheduled = link_analysis.get("assessment", {}).get("is_scheduled", False)
        
        # In a real system, would analyze historical communication patterns
        # For now, generate mock analysis
        
        # Generate random historical pattern
        historical_pattern = np.random.choice([
            "regular_scheduled", "on_demand", "continuous", 
            "intermittent", "event_triggered", "unknown"
        ])
        
        # Determine if current change matches historical pattern
        matches_historical_pattern = np.random.random() > 0.2  # 80% match historical pattern
        
        # Identify potential ground stations
        # In a real system, would determine this based on orbit and known ground facilities
        potential_ground_stations = []
        
        # Add 1-3 random ground stations
        num_stations = np.random.randint(1, 4)
        all_stations = [
            {"name": "Goldstone", "operator": "USA"},
            {"name": "Madrid", "operator": "ESA"},
            {"name": "Canberra", "operator": "USA"},
            {"name": "Weilheim", "operator": "Germany"},
            {"name": "Kaena Point", "operator": "USA"},
            {"name": "Dongara", "operator": "Australia"},
            {"name": "Fucino", "operator": "Italy"},
            {"name": "Bear Lakes", "operator": "Russia"},
            {"name": "Jiamusi", "operator": "China"},
            {"name": "Svalbard", "operator": "Norway"}
        ]
        
        potential_ground_stations = np.random.choice(all_stations, num_stations, replace=False).tolist()
        
        # Determine if any anomalies are present
        anomalies = []
        
        if not is_scheduled and historical_pattern in ["regular_scheduled", "continuous"]:
            anomalies.append("unexpected_timing")
            
        if change_type == "deactivation" and historical_pattern == "continuous":
            anomalies.append("unexpected_deactivation")
            
        if "frequency_shift" in link_analysis.get("specific_changes", []) and historical_pattern != "unknown":
            anomalies.append("unexpected_frequency_change")
            
        if "encryption_change" in link_analysis.get("specific_changes", []):
            anomalies.append("encryption_protocol_change")
            
        # Determine if pattern suggests automation or manual operation
        automated_probability = 0.8 if is_scheduled and matches_historical_pattern else 0.4
        
        # Determine if pattern suggests command sequence
        command_sequence_probability = 0.7 if change_type == "activation" and len(anomalies) == 0 else 0.3
        
        return {
            "historical_pattern": historical_pattern,
            "matches_historical_pattern": matches_historical_pattern,
            "potential_ground_stations": potential_ground_stations,
            "anomalies": anomalies,
            "automation_assessment": {
                "automated_probability": automated_probability,
                "likely_automated": automated_probability > 0.5
            },
            "command_assessment": {
                "command_sequence_probability": command_sequence_probability,
                "likely_command_sequence": command_sequence_probability > 0.5
            }
        }
    
    async def assess_hostility(self, event: Event) -> Tuple[Dict[str, Any], ThreatLevel]:
        """
        Assess hostility of a link change event.
        
        Factors:
        1. Link change type and characteristics
        2. Operational context
        3. Communication patterns
        4. Anomalies
        
        Args:
            event: The event to assess
            
        Returns:
            Tuple of (assessment_details, threat_level)
        """
        # Extract data from processing steps
        link_analysis_step = next((s for s in event.processing_steps 
                                  if s.step_name == "link_analysis"), None)
        impact_step = next((s for s in event.processing_steps 
                           if s.step_name == "operational_impact"), None)
        pattern_step = next((s for s in event.processing_steps 
                            if s.step_name == "comm_pattern"), None)
        
        if not all([link_analysis_step, impact_step, pattern_step]):
            return {"error": "Missing required processing steps"}, ThreatLevel.MODERATE
        
        link_analysis_data = link_analysis_step.output or {}
        impact_data = impact_step.output or {}
        pattern_data = pattern_step.output or {}
        
        # Extract key information
        change_type = link_analysis_data.get("basic_change_type", "unknown")
        specific_changes = link_analysis_data.get("specific_changes", [])
        is_scheduled = link_analysis_data.get("assessment", {}).get("is_scheduled", False)
        is_encrypted = link_analysis_data.get("assessment", {}).get("is_encrypted", False)
        
        spacecraft_data = impact_data.get("spacecraft_data", {})
        operational_significance = impact_data.get("operational_significance", "nominal")
        mission_impact = impact_data.get("mission_impact", "none")
        
        matches_historical_pattern = pattern_data.get("matches_historical_pattern", True)
        anomalies = pattern_data.get("anomalies", [])
        
        # Assess hostility
        hostility_score = 0
        
        # Factor 1: Change type
        if change_type == "activation" and operational_significance == "anomalous":
            hostility_score += 2
        elif change_type == "deactivation" and operational_significance == "anomalous":
            hostility_score += 1
            
        # Factor 2: Specific changes
        if "encryption_change" in specific_changes:
            hostility_score += 1
            
        # Factor 3: Scheduling and patterns
        if not is_scheduled and not matches_historical_pattern:
            hostility_score += 2
            
        # Factor 4: Operational significance
        if operational_significance == "anomalous":
            hostility_score += 2
        elif operational_significance == "noteworthy":
            hostility_score += 1
            
        # Factor 5: Mission type and owner
        is_military = spacecraft_data.get("mission_type") == "military"
        is_foreign = spacecraft_data.get("owner") not in ["USA", "Allied", "Commercial"]
        
        if is_military and is_foreign and len(anomalies) > 0:
            hostility_score += 3
        elif is_military and len(anomalies) > 0:
            hostility_score += 2
            
        # Factor 6: Anomalies
        if "unexpected_deactivation" in anomalies and is_military:
            hostility_score += 2
        elif "unexpected_frequency_change" in anomalies and is_military:
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
                "change_type": change_type,
                "specific_changes": specific_changes,
                "is_scheduled": is_scheduled,
                "is_encrypted": is_encrypted,
                "operational_significance": operational_significance,
                "mission_impact": mission_impact,
                "matches_historical_pattern": matches_historical_pattern,
                "anomalies": anomalies,
                "is_military": is_military,
                "is_foreign": is_foreign
            },
            "assessment_time": datetime.utcnow().isoformat()
        }
        
        return assessment, threat_level
    
    async def generate_coa(self, event: Event, threat_level: ThreatLevel) -> CourseOfAction:
        """
        Generate course of action for a link change event.
        
        Args:
            event: The event to generate COA for
            threat_level: Assessed threat level
            
        Returns:
            Recommended course of action
        """
        # Extract data from processing steps
        link_analysis_step = next((s for s in event.processing_steps 
                                  if s.step_name == "link_analysis"), None)
        impact_step = next((s for s in event.processing_steps 
                           if s.step_name == "operational_impact"), None)
        pattern_step = next((s for s in event.processing_steps 
                            if s.step_name == "comm_pattern"), None)
        
        link_analysis_data = link_analysis_step.output if link_analysis_step else {}
        impact_data = impact_step.output or {}
        pattern_data = pattern_step.output or {}
        
        # Extract key information
        object_id = event.object_id
        change_type = link_analysis_data.get("basic_change_type", "unknown")
        link_type = link_analysis_data.get("link_type", "unknown")
        
        spacecraft_data = impact_data.get("spacecraft_data", {})
        operational_significance = impact_data.get("operational_significance", "nominal")
        mission_impact = impact_data.get("mission_impact", "none")
        
        anomalies = pattern_data.get("anomalies", [])
        
        # Set COA priority based on threat level
        priority = 1  # Default low priority
        if threat_level == ThreatLevel.HIGH:
            priority = 5
        elif threat_level == ThreatLevel.MODERATE:
            priority = 3
        elif threat_level == ThreatLevel.LOW:
            priority = 2
        
        # Generate actions based on threat level and change type
        actions = [f"Continue monitoring communications of {object_id}"]
        
        # Add recording action for all cases
        actions.append(f"Record signal characteristics of {link_type} link")
        
        # Add appropriate actions based on change type
        if change_type == "activation":
            actions.append("Analyze signal content and metadata")
            
            if operational_significance in ["noteworthy", "anomalous"]:
                actions.append("Correlate with known operational schedules")
                
        elif change_type == "deactivation":
            actions.append("Monitor for reactivation or alternative links")
            
            if operational_significance == "anomalous":
                actions.append("Check for correlation with orbital events/maneuvers")
                
        elif change_type == "modification":
            actions.append("Compare new signal characteristics with previous baseline")
            
        # Add anomaly-specific actions
        if "unexpected_timing" in anomalies:
            actions.append("Analyze timing pattern for indications of intent")
            
        if "encryption_protocol_change" in anomalies:
            actions.append("Attempt to characterize new encryption protocol")
            
        # Add threat-specific actions
        if threat_level == ThreatLevel.MODERATE:
            actions.append("Increase collection priority for this target")
            actions.append("Alert space operations center of communication anomaly")
            
            if spacecraft_data.get("mission_type") == "military":
                actions.append("Assess potential operational impacts on mission")
                
        if threat_level == ThreatLevel.HIGH:
            actions.append("Initiate continuous monitoring of all frequencies")
            actions.append("Alert leadership of potential hostile activity")
            actions.append("Assess defensive posture and mitigation options")
            
            if spacecraft_data.get("owner") not in ["USA", "Allied", "Commercial"]:
                actions.append("Prepare diplomatic inquiries regarding communication changes")
                
        # Create COA
        title = f"Link Change Response Plan: {threat_level.value.capitalize()} Threat"
        description = (
            f"Response plan for {change_type} of {link_type} communications link "
            f"on {object_id} ({spacecraft_data.get('owner', 'unknown')})"
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