"""Reentry event processor."""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import numpy as np

from app.models.events import (
    Event, EventType, EventStatus, EventDetection,
    ThreatLevel, CourseOfAction, ReentryDetection
)
from app.services.event_processor_base import EventProcessorBase
from app.services.trajectory_service import TrajectoryService

logger = logging.getLogger(__name__)

class ReentryProcessor(EventProcessorBase):
    """Processor for reentry events."""
    
    def __init__(self):
        super().__init__()
        self.event_type = EventType.REENTRY
    
    async def detect_entry_criteria(self, data: Dict[str, Any]) -> Optional[EventDetection]:
        """
        Check if data meets entry criteria for reentry events.
        
        Entry criteria:
        1. Predicted altitude reaches below threshold (e.g., 100 km)
        2. Confidence above threshold
        
        Args:
            data: Sensor data
            
        Returns:
            EventDetection if criteria met, None otherwise
        """
        # Extract required fields
        if not all(key in data for key in ["object_id", "altitude_km", "prediction_confidence"]):
            return None
        
        # Check if altitude is below threshold
        if data["altitude_km"] > 100:  # Karman line
            return None
        
        # Check confidence
        if data["prediction_confidence"] < 0.7:
            return None
        
        # Create detection
        return EventDetection(
            event_type=EventType.REENTRY,
            object_id=data["object_id"],
            detection_time=datetime.utcnow(),
            confidence=data["prediction_confidence"],
            sensor_data={
                "altitude_km": data["altitude_km"],
                "predicted_location": data.get("predicted_location", {"lat": 0, "lon": 0}),
                "predicted_time": data.get("predicted_time", datetime.utcnow().isoformat()),
                "velocity_km_s": data.get("velocity_km_s", 0)
            }
        )
    
    async def process_event(self, event: Event) -> Event:
        """
        Process a reentry event.
        
        Processing steps:
        1. Update status to PROCESSING
        2. Perform trajectory analysis
        3. Determine impact prediction
        4. Identify populations at risk
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
            # Step 1: Perform trajectory analysis
            trajectory_results = await self._analyze_trajectory(event)
            event = await self.add_processing_step(
                event=event,
                step_name="trajectory_analysis",
                status="success",
                output=trajectory_results
            )
            
            # Step 2: Identify populations at risk
            population_results = await self._identify_populations_at_risk(
                trajectory_results["impact_location"], 
                trajectory_results["uncertainty_radius_km"]
            )
            event = await self.add_processing_step(
                event=event,
                step_name="population_risk_assessment",
                status="success",
                output=population_results
            )
            
            # Step 3: Assess hostility
            hostility_assessment, threat_level = await self.assess_hostility(event)
            event = await self.add_processing_step(
                event=event,
                step_name="hostility_assessment",
                status="success",
                output={"threat_level": threat_level.value}
            )
            
            # Step 4: Generate COA
            coa = await self.generate_coa(event, threat_level)
            
            # Complete the event
            event = await self.complete_event(
                event=event,
                threat_level=threat_level,
                hostility_assessment=hostility_assessment,
                coa=coa
            )
            
            return event
            
        except Exception as e:
            logger.error(f"Error processing reentry event: {str(e)}", exc_info=True)
            event.status = EventStatus.ERROR
            event = await self.add_processing_step(
                event=event,
                step_name="error",
                status="failed",
                error=str(e)
            )
            return event
    
    async def _analyze_trajectory(self, event: Event) -> Dict[str, Any]:
        """
        Analyze the trajectory of the reentering object.
        
        Args:
            event: The reentry event
            
        Returns:
            Trajectory analysis results
        """
        # In a real implementation, this would use the trajectory service
        # and actual physics calculations
        
        # Extract data from event
        sensor_data = event.detection_data.sensor_data
        
        # Mock trajectory analysis
        impact_time = datetime.utcnow() + timedelta(minutes=30)
        
        # Generate a realistic impact location near the predicted location
        predicted_location = sensor_data.get("predicted_location", {"lat": 0, "lon": 0})
        impact_location = {
            "lat": predicted_location["lat"] + np.random.normal(0, 0.5),
            "lon": predicted_location["lon"] + np.random.normal(0, 0.5)
        }
        
        return {
            "impact_time": impact_time.isoformat(),
            "impact_location": impact_location,
            "impact_velocity_km_s": sensor_data.get("velocity_km_s", 7.8) * 0.9,
            "uncertainty_radius_km": 20.0,
            "time_to_impact_minutes": 30,
            "survivable_fragments": np.random.randint(0, 5)
        }
    
    async def _identify_populations_at_risk(self, impact_location: Dict[str, float], 
                                          uncertainty_radius_km: float) -> Dict[str, Any]:
        """
        Identify populations at risk from the reentry.
        
        Args:
            impact_location: Predicted impact location
            uncertainty_radius_km: Uncertainty radius in km
            
        Returns:
            Population risk assessment
        """
        # Mock implementation - in real system would query population database
        
        # Generate random population density based on location
        # Ocean locations would have ~0 population
        lat, lon = impact_location["lat"], impact_location["lon"]
        
        # Over ocean check (very simplified)
        is_ocean = abs(lat) < 60 and (lon < -60 or lon > 60)
        
        if is_ocean:
            population_density = 0
            region_type = "ocean"
        else:
            population_density = np.random.lognormal(3, 1)  # people per sq km
            region_type = np.random.choice(["urban", "suburban", "rural"], 
                                           p=[0.2, 0.3, 0.5])
        
        # Calculate impact area
        impact_area_sq_km = np.pi * uncertainty_radius_km**2
        
        # Estimate population at risk
        population_at_risk = int(population_density * impact_area_sq_km)
        
        return {
            "region_type": region_type,
            "population_density": population_density,
            "impact_area_sq_km": impact_area_sq_km,
            "population_at_risk": population_at_risk,
            "critical_infrastructure": region_type == "urban"
        }
    
    async def assess_hostility(self, event: Event) -> Tuple[Dict[str, Any], ThreatLevel]:
        """
        Assess hostility of a reentry event.
        
        Factors:
        1. Object characteristics (size, origin)
        2. Trajectory predictability
        3. Timing and location of reentry
        4. Population centers at risk
        5. Advanced warning provided
        
        Args:
            event: The event to assess
            
        Returns:
            Tuple of (assessment_details, threat_level)
        """
        # Extract data from processing steps
        trajectory_step = next((s for s in event.processing_steps 
                               if s.step_name == "trajectory_analysis"), None)
        population_step = next((s for s in event.processing_steps 
                               if s.step_name == "population_risk_assessment"), None)
        
        if not trajectory_step or not population_step:
            return {"error": "Missing required processing steps"}, ThreatLevel.MODERATE
        
        trajectory_data = trajectory_step.output or {}
        population_data = population_step.output or {}
        
        # Assess factors
        survivable_fragments = trajectory_data.get("survivable_fragments", 0)
        population_at_risk = population_data.get("population_at_risk", 0)
        critical_infrastructure = population_data.get("critical_infrastructure", False)
        region_type = population_data.get("region_type", "ocean")
        
        # Perform assessment
        hostility_score = 0
        
        # Factor 1: Survivable fragments
        if survivable_fragments > 3:
            hostility_score += 2
        elif survivable_fragments > 0:
            hostility_score += 1
        
        # Factor 2: Population at risk
        if population_at_risk > 100000:
            hostility_score += 3
        elif population_at_risk > 10000:
            hostility_score += 2
        elif population_at_risk > 1000:
            hostility_score += 1
        
        # Factor 3: Critical infrastructure
        if critical_infrastructure:
            hostility_score += 2
        
        # Factor 4: Region type
        if region_type == "urban":
            hostility_score += 2
        elif region_type == "suburban":
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
                "survivable_fragments": survivable_fragments,
                "population_at_risk": population_at_risk,
                "critical_infrastructure": critical_infrastructure,
                "region_type": region_type
            },
            "assessment_time": datetime.utcnow().isoformat()
        }
        
        return assessment, threat_level
    
    async def generate_coa(self, event: Event, threat_level: ThreatLevel) -> CourseOfAction:
        """
        Generate course of action for a reentry event.
        
        Args:
            event: The event to generate COA for
            threat_level: Assessed threat level
            
        Returns:
            Recommended course of action
        """
        # Extract data from processing steps
        trajectory_step = next((s for s in event.processing_steps 
                               if s.step_name == "trajectory_analysis"), None)
        population_step = next((s for s in event.processing_steps 
                               if s.step_name == "population_risk_assessment"), None)
        
        trajectory_data = trajectory_step.output if trajectory_step else {}
        population_data = population_step.output if population_step else {}
        
        # Extract key information
        impact_time = trajectory_data.get("impact_time", datetime.utcnow().isoformat())
        impact_location = trajectory_data.get("impact_location", {"lat": 0, "lon": 0})
        population_at_risk = population_data.get("population_at_risk", 0)
        
        # Set COA priority based on threat level
        priority = 1  # Default low priority
        if threat_level == ThreatLevel.HIGH:
            priority = 5
        elif threat_level == ThreatLevel.MODERATE:
            priority = 4
        elif threat_level == ThreatLevel.LOW:
            priority = 3
        
        # Generate actions based on threat level and population
        actions = ["Continue monitoring reentry trajectory"]
        
        if threat_level in [ThreatLevel.MODERATE, ThreatLevel.HIGH]:
            actions.extend([
                f"Issue notification to air traffic control for region near {impact_location['lat']:.2f}, {impact_location['lon']:.2f}",
                "Activate space debris monitoring network"
            ])
        
        if threat_level == ThreatLevel.HIGH:
            actions.extend([
                f"Issue emergency alert to civilian authorities in affected region",
                "Dispatch recovery/containment teams to predicted impact site",
                "Initiate diplomatic channels if object is foreign"
            ])
        
        if population_at_risk > 1000:
            actions.append("Prepare evacuation recommendations for affected areas")
        
        # Create COA
        title = f"Reentry Response Plan: {threat_level.value.capitalize()} Threat"
        description = (
            f"Response plan for reentry of object {event.object_id} "
            f"with predicted impact at {impact_time}."
        )
        
        # Expiration should be after impact
        impact_dt = datetime.fromisoformat(impact_time) if isinstance(impact_time, str) else impact_time
        expiration = impact_dt + timedelta(hours=2)
        
        return CourseOfAction(
            title=title,
            description=description,
            priority=priority,
            actions=actions,
            expiration=expiration
        )