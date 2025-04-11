"""Launch event processor."""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import numpy as np

from app.models.events import (
    Event, EventType, EventStatus, EventDetection,
    ThreatLevel, CourseOfAction, LaunchDetection
)
from app.services.event_processor_base import EventProcessorBase

logger = logging.getLogger(__name__)

class LaunchProcessor(EventProcessorBase):
    """Processor for launch events."""
    
    def __init__(self):
        super().__init__()
        self.event_type = EventType.LAUNCH
    
    async def detect_entry_criteria(self, data: Dict[str, Any]) -> Optional[EventDetection]:
        """
        Check if data meets entry criteria for launch events.
        
        Entry criteria:
        1. Detection of a new object in ascent trajectory
        2. Confidence above threshold
        
        Args:
            data: Sensor data
            
        Returns:
            EventDetection if criteria met, None otherwise
        """
        # Extract required fields
        if not all(key in data for key in ["object_id", "launch_site", "initial_trajectory", "confidence"]):
            return None
        
        # Check confidence
        if data["confidence"] < 0.7:
            return None
        
        # Create detection
        return EventDetection(
            event_type=EventType.LAUNCH,
            object_id=data["object_id"],
            detection_time=datetime.utcnow(),
            confidence=data["confidence"],
            sensor_data={
                "launch_site": data["launch_site"],
                "initial_trajectory": data["initial_trajectory"],
                "launch_time": data.get("launch_time", datetime.utcnow().isoformat()),
                "velocity": data.get("velocity", 0),
                "altitude": data.get("altitude", 0),
                "sensor_id": data.get("sensor_id", "unknown"),
                "thermal_signature": data.get("thermal_signature", {})
            }
        )
    
    async def process_event(self, event: Event) -> Event:
        """
        Process a launch event.
        
        Processing steps:
        1. Update status to PROCESSING
        2. Classify launch type
        3. Estimate payload characteristics
        4. Predict orbit
        5. Analyze launch site
        6. Assess hostility
        7. Generate COA
        8. Update status to COMPLETED
        
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
            # Step 1: Classify launch type
            launch_classification = await self._classify_launch(event)
            event = await self.add_processing_step(
                event=event,
                step_name="launch_classification",
                status="success",
                output=launch_classification
            )
            
            # Step 2: Estimate payload characteristics
            payload_estimation = await self._estimate_payload(event, launch_classification)
            event = await self.add_processing_step(
                event=event,
                step_name="payload_estimation",
                status="success",
                output=payload_estimation
            )
            
            # Step 3: Predict orbit
            orbit_prediction = await self._predict_orbit(event, launch_classification)
            event = await self.add_processing_step(
                event=event,
                step_name="orbit_prediction",
                status="success",
                output=orbit_prediction
            )
            
            # Step 4: Analyze launch site
            launch_site_analysis = await self._analyze_launch_site(event)
            event = await self.add_processing_step(
                event=event,
                step_name="launch_site_analysis",
                status="success",
                output=launch_site_analysis
            )
            
            # Step 5: Assess hostility
            hostility_assessment, threat_level = await self.assess_hostility(event)
            event = await self.add_processing_step(
                event=event,
                step_name="hostility_assessment",
                status="success",
                output={"threat_level": threat_level.value}
            )
            
            # Step 6: Generate COA
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
            logger.error(f"Error processing launch event: {str(e)}", exc_info=True)
            event.status = EventStatus.ERROR
            event = await self.add_processing_step(
                event=event,
                step_name="error",
                status="failed",
                error=str(e)
            )
            return event
    
    async def _classify_launch(self, event: Event) -> Dict[str, Any]:
        """
        Classify the type of launch.
        
        Classification types:
        - Orbital (LEO, MEO, GEO, HEO)
        - Suborbital
        - Ballistic missile
        - Sounding rocket
        
        Args:
            event: The launch event
            
        Returns:
            Classification results
        """
        # Extract relevant data
        sensor_data = event.detection_data.sensor_data
        initial_trajectory = sensor_data.get("initial_trajectory", [0, 0, 0])
        velocity = sensor_data.get("velocity", 0)
        altitude = sensor_data.get("altitude", 0)
        thermal_signature = sensor_data.get("thermal_signature", {})
        
        # Perform classification based on trajectory and velocity
        classifications = {}
        
        # Orbital launches need to reach orbital velocity (approximately > 7.8 km/s)
        is_orbital = velocity > 7000
        
        if is_orbital:
            # Classify orbital regime based on trajectory and velocity
            # LEO: 160-2000km, MEO: 2000-35786km, GEO: 35786km, HEO: high eccentricity
            classifications["orbital_leo"] = 0.8 if velocity < 9000 else 0.1
            classifications["orbital_meo"] = 0.8 if 9000 <= velocity < 10000 else 0.1
            classifications["orbital_geo"] = 0.8 if velocity >= 10000 else 0.1
            classifications["orbital_heo"] = 0.5  # Hard to determine without full trajectory
        else:
            # Suborbital classifications
            classifications["suborbital"] = 0.8 if velocity > 3000 else 0.1
            classifications["ballistic_missile"] = 0.7 if (velocity > 3000 and altitude < 150000) else 0.1
            classifications["sounding_rocket"] = 0.8 if (velocity < 3000 and altitude > 50000) else 0.1
        
        # Determine primary classification
        primary_classification = max(classifications.items(), key=lambda x: x[1])
        
        # Estimate launch vehicle type based on thermal signature and velocity
        launch_vehicle_type = "unknown"
        if "heat_signature" in thermal_signature:
            heat_sig = thermal_signature["heat_signature"]
            if is_orbital and heat_sig > 5000:
                launch_vehicle_type = "heavy_lift"
            elif is_orbital:
                launch_vehicle_type = "medium_lift"
            elif velocity > 3000:
                launch_vehicle_type = "ballistic_missile"
            else:
                launch_vehicle_type = "small_rocket"
        
        return {
            "primary_type": primary_classification[0],
            "confidence": primary_classification[1],
            "classifications": classifications,
            "is_orbital": is_orbital,
            "launch_vehicle_type": launch_vehicle_type,
            "launch_characteristics": {
                "initial_velocity_m_s": velocity,
                "initial_altitude_m": altitude,
                "heading": np.arctan2(initial_trajectory[1], initial_trajectory[0]) if len(initial_trajectory) >= 2 else 0,
                "launch_energy": velocity**2 * 0.5,  # Simplified kinetic energy calculation
                "multi_stage": thermal_signature.get("multi_stage", False)
            }
        }
    
    async def _estimate_payload(self, event: Event, 
                            classification_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate characteristics of the payload.
        
        Args:
            event: The launch event
            classification_results: Results from launch classification
            
        Returns:
            Payload estimation
        """
        # Extract relevant data
        primary_type = classification_results.get("primary_type", "unknown")
        launch_vehicle_type = classification_results.get("launch_vehicle_type", "unknown")
        is_orbital = classification_results.get("is_orbital", False)
        
        # In a real system, would use physics models and historical data
        # For now, generate mock estimations
        
        # Estimate payload mass based on launch vehicle type
        estimated_payload_mass = 0
        if launch_vehicle_type == "heavy_lift":
            estimated_payload_mass = np.random.uniform(5000, 20000)  # kg
        elif launch_vehicle_type == "medium_lift":
            estimated_payload_mass = np.random.uniform(1000, 5000)  # kg
        elif launch_vehicle_type == "ballistic_missile":
            estimated_payload_mass = np.random.uniform(500, 2000)  # kg
        elif launch_vehicle_type == "small_rocket":
            estimated_payload_mass = np.random.uniform(50, 500)  # kg
            
        # Estimate payload type based on classification
        payload_type_probabilities = {}
        
        if primary_type.startswith("orbital"):
            payload_type_probabilities = {
                "satellite": 0.7,
                "space_station_module": 0.1,
                "multiple_satellites": 0.15,
                "unknown": 0.05
            }
        elif primary_type == "ballistic_missile":
            payload_type_probabilities = {
                "conventional_warhead": 0.6,
                "test_payload": 0.3,
                "unknown": 0.1
            }
        else:  # Suborbital, sounding rocket
            payload_type_probabilities = {
                "scientific_instruments": 0.6,
                "technology_demonstration": 0.3,
                "unknown": 0.1
            }
            
        # Select payload type based on probabilities
        payload_types = list(payload_type_probabilities.keys())
        payload_probs = list(payload_type_probabilities.values())
        payload_type = np.random.choice(payload_types, p=payload_probs)
        
        # Determine if multiple payloads
        multiple_payloads = False
        if payload_type == "multiple_satellites":
            multiple_payloads = True
            estimated_payload_count = np.random.randint(2, 10)
        else:
            estimated_payload_count = 1
            
        # Estimate payload capabilities
        capabilities = {}
        
        if is_orbital:
            capabilities = {
                "propulsion": np.random.random() > 0.2,  # 80% chance of propulsion for orbital
                "communication": np.random.random() > 0.1,  # 90% chance of communication for orbital
                "sensors": np.random.random() > 0.3,  # 70% chance of sensors for orbital
                "power_generation": True  # All orbital payloads need power
            }
        else:
            capabilities = {
                "propulsion": np.random.random() > 0.7,  # 30% chance of propulsion for suborbital
                "communication": np.random.random() > 0.4,  # 60% chance of communication for suborbital
                "sensors": np.random.random() > 0.2,  # 80% chance of sensors for suborbital
                "power_generation": np.random.random() > 0.3  # 70% chance of power for suborbital
            }
            
        return {
            "estimated_payload_mass_kg": estimated_payload_mass,
            "payload_type": payload_type,
            "multiple_payloads": multiple_payloads,
            "estimated_payload_count": estimated_payload_count,
            "estimated_capabilities": capabilities,
            "military_payload_probability": 0.8 if payload_type == "conventional_warhead" else 
                                           (0.4 if primary_type.startswith("orbital") else 0.1),
            "confidence": 0.7  # Moderate confidence in payload estimation
        }
    
    async def _predict_orbit(self, event: Event, 
                         classification_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict the target orbit of the launch.
        
        Args:
            event: The launch event
            classification_results: Results from launch classification
            
        Returns:
            Orbit prediction
        """
        # Extract relevant data
        sensor_data = event.detection_data.sensor_data
        initial_trajectory = sensor_data.get("initial_trajectory", [0, 0, 0])
        launch_site = sensor_data.get("launch_site", {"lat": 0, "lon": 0})
        primary_type = classification_results.get("primary_type", "unknown")
        is_orbital = classification_results.get("is_orbital", False)
        
        # Mock orbit prediction - in reality would use orbital mechanics
        
        # Return minimal prediction for non-orbital launches
        if not is_orbital:
            return {
                "is_orbital": False,
                "apogee_km": np.random.uniform(100, 500),
                "impact_location": {
                    "lat": launch_site["lat"] + np.random.uniform(-10, 10),
                    "lon": launch_site["lon"] + np.random.uniform(-10, 10)
                },
                "confidence": 0.7
            }
        
        # Generate orbital parameters based on classification
        orbital_parameters = {}
        
        if primary_type == "orbital_leo":
            orbital_parameters = {
                "semi_major_axis_km": np.random.uniform(6700, 8000),
                "eccentricity": np.random.uniform(0, 0.01),
                "inclination_deg": np.random.uniform(0, 90),
                "period_minutes": np.random.uniform(90, 120)
            }
        elif primary_type == "orbital_meo":
            orbital_parameters = {
                "semi_major_axis_km": np.random.uniform(8000, 25000),
                "eccentricity": np.random.uniform(0, 0.1),
                "inclination_deg": np.random.uniform(0, 90),
                "period_minutes": np.random.uniform(120, 720)
            }
        elif primary_type == "orbital_geo":
            orbital_parameters = {
                "semi_major_axis_km": 42164,  # GEO
                "eccentricity": np.random.uniform(0, 0.001),
                "inclination_deg": np.random.uniform(0, 1),
                "period_minutes": 1436  # 24 hours
            }
        elif primary_type == "orbital_heo":
            orbital_parameters = {
                "semi_major_axis_km": np.random.uniform(8000, 40000),
                "eccentricity": np.random.uniform(0.2, 0.8),
                "inclination_deg": np.random.uniform(40, 90),
                "period_minutes": np.random.uniform(120, 1000)
            }
            
        # Calculate derived orbital parameters
        semi_major_axis = orbital_parameters["semi_major_axis_km"]
        eccentricity = orbital_parameters["eccentricity"]
        
        # Earth radius in km
        earth_radius = 6371
        
        # Calculate perigee and apogee
        perigee_km = semi_major_axis * (1 - eccentricity) - earth_radius
        apogee_km = semi_major_axis * (1 + eccentricity) - earth_radius
        
        # Determine orbit classification
        orbit_class = "LEO"
        if apogee_km > 35000:
            orbit_class = "GEO/HEO"
        elif apogee_km > 2000:
            orbit_class = "MEO"
            
        # Generate ground track information
        # In reality would calculate from orbital elements
        ground_track = []
        inclination = orbital_parameters["inclination_deg"]
        
        if inclination < 10:  # Near equatorial
            ground_track_type = "equatorial"
        elif abs(inclination - 90) < 10:  # Near polar
            ground_track_type = "polar"
        else:
            ground_track_type = "inclined"
            
        return {
            "is_orbital": True,
            "orbit_class": orbit_class,
            "orbital_parameters": orbital_parameters,
            "perigee_km": perigee_km,
            "apogee_km": apogee_km,
            "ground_track_type": ground_track_type,
            "confidence": 0.8,
            "operational_orbit_reached": False,  # Assuming just after launch detection
            "estimated_time_to_orbit_minutes": np.random.uniform(10, 30)
        }
    
    async def _analyze_launch_site(self, event: Event) -> Dict[str, Any]:
        """
        Analyze the launch site.
        
        Args:
            event: The launch event
            
        Returns:
            Launch site analysis
        """
        # Extract relevant data
        sensor_data = event.detection_data.sensor_data
        launch_site = sensor_data.get("launch_site", {"lat": 0, "lon": 0})
        
        # In a real system, would query a database of known launch sites
        # For now, generate mock analysis
        
        # Check if launch site is known
        # Create a simplified database of known launch sites
        known_sites = [
            {"name": "Cape Canaveral", "lat": 28.4, "lon": -80.6, "country": "USA", "commercial": False},
            {"name": "Kennedy Space Center", "lat": 28.6, "lon": -80.6, "country": "USA", "commercial": False},
            {"name": "Vandenberg", "lat": 34.7, "lon": -120.6, "country": "USA", "commercial": False},
            {"name": "Baikonur", "lat": 45.9, "lon": 63.3, "country": "Russia", "commercial": False},
            {"name": "Jiuquan", "lat": 40.9, "lon": 100.3, "country": "China", "commercial": False},
            {"name": "Tanegashima", "lat": 30.4, "lon": 131.0, "country": "Japan", "commercial": False},
            {"name": "Kourou", "lat": 5.2, "lon": -52.8, "country": "ESA", "commercial": False},
            {"name": "Satish Dhawan", "lat": 13.7, "lon": 80.2, "country": "India", "commercial": False},
            {"name": "Wallops", "lat": 37.8, "lon": -75.5, "country": "USA", "commercial": True},
            {"name": "Rocket Lab LC1", "lat": -39.3, "lon": 177.8, "country": "New Zealand", "commercial": True}
        ]
        
        # Find the closest known site
        closest_site = None
        min_distance = float('inf')
        
        for site in known_sites:
            # Calculate approximate distance using simplified formula
            distance = ((launch_site["lat"] - site["lat"])**2 + 
                         (launch_site["lon"] - site["lon"])**2)**0.5
            
            if distance < min_distance:
                min_distance = distance
                closest_site = site
        
        # Check if launch site is close enough to a known site
        # 0.1 degrees is roughly 11 km at the equator
        is_known_site = min_distance < 0.1
        site_info = closest_site if is_known_site else None
        
        # Determine if mobile launch
        # Assume mobile launch if not near known site and not over water
        is_over_water = self._is_over_water(launch_site["lat"], launch_site["lon"])
        is_mobile_launch = not is_known_site and not is_over_water
        
        # Determine country of origin
        country_of_origin = site_info["country"] if is_known_site else "Unknown"
        
        # Determine if launch was announced
        # In reality, would check against database of announced launches
        is_announced = np.random.random() > 0.2  # 80% of launches are announced
        if not is_known_site:
            is_announced = np.random.random() > 0.7  # Only 30% of unknown site launches are announced
            
        return {
            "launch_site": {
                "latitude": launch_site["lat"],
                "longitude": launch_site["lon"],
                "is_known_site": is_known_site,
                "site_name": site_info["name"] if is_known_site else "Unknown",
                "country": country_of_origin
            },
            "launch_circumstances": {
                "is_from_established_facility": is_known_site,
                "is_mobile_launch": is_mobile_launch,
                "is_maritime_launch": is_over_water and not is_known_site,
                "is_announced": is_announced,
                "is_commercial": is_known_site and site_info["commercial"]
            },
            "confidence": 0.9 if is_known_site else 0.6
        }
    
    def _is_over_water(self, lat: float, lon: float) -> bool:
        """
        Determine if coordinates are over water.
        
        This is a very simplified approximation. In a real system,
        would use a geographic database.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            True if coordinates are likely over water
        """
        # Very rough approximation of major landmasses
        # For demonstration only - real implementation would use proper geo data
        
        # Check if in bounds of major continents (simplified)
        in_north_america = (-170 < lon < -50) and (15 < lat < 75)
        in_south_america = (-90 < lon < -30) and (-60 < lat < 15)
        in_europe_africa = (-20 < lon < 60) and (-40 < lat < 70)
        in_asia_australia = (60 < lon < 180) and (-50 < lat < 75)
        
        return not (in_north_america or in_south_america or in_europe_africa or in_asia_australia)
    
    async def assess_hostility(self, event: Event) -> Tuple[Dict[str, Any], ThreatLevel]:
        """
        Assess hostility of a launch event.
        
        Factors:
        1. Launch type and payload characteristics
        2. Origin and announcement status
        3. Trajectory and orbit
        4. Historical context
        
        Args:
            event: The event to assess
            
        Returns:
            Tuple of (assessment_details, threat_level)
        """
        # Extract data from processing steps
        classification_step = next((s for s in event.processing_steps 
                                   if s.step_name == "launch_classification"), None)
        payload_step = next((s for s in event.processing_steps 
                            if s.step_name == "payload_estimation"), None)
        orbit_step = next((s for s in event.processing_steps 
                          if s.step_name == "orbit_prediction"), None)
        site_step = next((s for s in event.processing_steps 
                         if s.step_name == "launch_site_analysis"), None)
        
        if not all([classification_step, payload_step, orbit_step, site_step]):
            return {"error": "Missing required processing steps"}, ThreatLevel.MODERATE
        
        classification_data = classification_step.output or {}
        payload_data = payload_step.output or {}
        orbit_data = orbit_step.output or {}
        site_data = site_step.output or {}
        
        # Extract key information
        primary_type = classification_data.get("primary_type", "unknown")
        is_orbital = classification_data.get("is_orbital", False)
        
        payload_type = payload_data.get("payload_type", "unknown")
        military_payload_probability = payload_data.get("military_payload_probability", 0.0)
        
        orbit_class = orbit_data.get("orbit_class", "LEO") if is_orbital else "N/A"
        
        country_of_origin = site_data.get("launch_site", {}).get("country", "Unknown")
        is_announced = site_data.get("launch_circumstances", {}).get("is_announced", False)
        is_mobile_launch = site_data.get("launch_circumstances", {}).get("is_mobile_launch", False)
        
        # Assess hostility
        hostility_score = 0
        
        # Factor 1: Launch type
        if primary_type == "ballistic_missile":
            hostility_score += 4
        elif primary_type == "suborbital" and not is_announced:
            hostility_score += 2
            
        # Factor 2: Payload assessment
        if payload_type == "conventional_warhead":
            hostility_score += 4
        elif military_payload_probability > 0.6:
            hostility_score += 2
        elif military_payload_probability > 0.3:
            hostility_score += 1
            
        # Factor 3: Launch origin
        if country_of_origin in ["Russia", "China", "North Korea", "Iran"] and military_payload_probability > 0.3:
            hostility_score += 2
        elif country_of_origin == "Unknown" and not is_announced:
            hostility_score += 2
            
        # Factor 4: Launch circumstances
        if not is_announced:
            hostility_score += 1
        if is_mobile_launch and not is_announced:
            hostility_score += 2
            
        # Factor 5: Orbit type (for orbital launches)
        if is_orbital and orbit_class in ["MEO", "GEO/HEO"] and military_payload_probability > 0.5:
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
                "launch_type": primary_type,
                "is_orbital": is_orbital,
                "payload_type": payload_type,
                "military_payload_probability": military_payload_probability,
                "country_of_origin": country_of_origin,
                "is_announced": is_announced,
                "is_mobile_launch": is_mobile_launch,
                "orbit_class": orbit_class
            },
            "assessment_time": datetime.utcnow().isoformat()
        }
        
        return assessment, threat_level
    
    async def generate_coa(self, event: Event, threat_level: ThreatLevel) -> CourseOfAction:
        """
        Generate course of action for a launch event.
        
        Args:
            event: The event to generate COA for
            threat_level: Assessed threat level
            
        Returns:
            Recommended course of action
        """
        # Extract data from processing steps
        classification_step = next((s for s in event.processing_steps 
                                   if s.step_name == "launch_classification"), None)
        payload_step = next((s for s in event.processing_steps 
                            if s.step_name == "payload_estimation"), None)
        orbit_step = next((s for s in event.processing_steps 
                          if s.step_name == "orbit_prediction"), None)
        site_step = next((s for s in event.processing_steps 
                         if s.step_name == "launch_site_analysis"), None)
        
        classification_data = classification_step.output if classification_step else {}
        payload_data = payload_step.output or {}
        orbit_data = orbit_step.output or {}
        site_data = site_step.output or {}
        
        # Extract key information
        primary_type = classification_data.get("primary_type", "unknown")
        is_orbital = classification_data.get("is_orbital", False)
        
        payload_type = payload_data.get("payload_type", "unknown")
        military_payload_probability = payload_data.get("military_payload_probability", 0.0)
        
        country_of_origin = site_data.get("launch_site", {}).get("country", "Unknown")
        is_announced = site_data.get("launch_circumstances", {}).get("is_announced", False)
        
        # Set COA priority based on threat level
        priority = 1  # Default low priority
        if threat_level == ThreatLevel.HIGH:
            priority = 5
        elif threat_level == ThreatLevel.MODERATE:
            priority = 3
        elif threat_level == ThreatLevel.LOW:
            priority = 2
        
        # Generate actions based on threat level and launch type
        actions = ["Continue tracking object"]
        
        # Add tracking actions
        if is_orbital:
            actions.append("Catalog new space object and determine final orbital parameters")
            actions.append("Determine payload type through visual and RF signature analysis")
        else:
            actions.append("Track trajectory to impact/endpoint")
            
        # Add specific actions based on classification
        if primary_type == "ballistic_missile":
            actions.append("Alert BMEWS and missile defense assets")
            actions.append("Determine probable target region")
            actions.append("Assess payload type (conventional/test)")
            
            if threat_level in [ThreatLevel.MODERATE, ThreatLevel.HIGH]:
                actions.append("Place missile defense assets on elevated alert")
                
            if threat_level == ThreatLevel.HIGH:
                actions.append("Notify national leadership of potential hostile launch")
                actions.append("Prepare strategic response options")
                
        elif is_orbital and military_payload_probability > 0.5:
            actions.append("Determine payload capabilities through persistent tracking")
            actions.append("Assess potential military applications")
            
            if threat_level in [ThreatLevel.MODERATE, ThreatLevel.HIGH]:
                actions.append("Increase space situational awareness for orbital region")
                actions.append("Alert operators of potentially threatened assets")
                
        # Add diplomatic actions for certain scenarios
        if threat_level in [ThreatLevel.MODERATE, ThreatLevel.HIGH] and not is_announced:
            actions.append(f"Prepare diplomatic inquiry to {country_of_origin}")
            
        # Add intelligence actions
        if military_payload_probability > 0.3 and not is_announced:
            actions.append("Increase intelligence collection on spacecraft capabilities")
            
        # Create COA
        title = f"Launch Response Plan: {threat_level.value.capitalize()} Threat"
        description = (
            f"Response plan for {primary_type} launch from {country_of_origin} "
            f"detected at {event.detection_data.detection_time}"
        )
        
        # Set expiration
        # For orbital objects, longer timeframe; for ballistic, shorter
        if is_orbital:
            expiration = datetime.utcnow() + timedelta(days=7)
        else:
            expiration = datetime.utcnow() + timedelta(hours=24)
        
        return CourseOfAction(
            title=title,
            description=description,
            priority=priority,
            actions=actions,
            expiration=expiration
        )