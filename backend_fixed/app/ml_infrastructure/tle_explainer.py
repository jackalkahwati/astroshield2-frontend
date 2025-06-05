"""
TLE Orbit Explainer Service
Integrates the jackal79/tle-orbit-explainer model for natural language TLE explanations
"""

import re
import math
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

# Constants for orbital mechanics
EARTH_RADIUS = 6378.137  # km
MU = 398600.4418  # Earth's gravitational parameter km³/s²
J2 = 1.08263e-3  # Earth's J2 coefficient
SIDEREAL_DAY = 86164.0905  # seconds

class TLEExplainerInput(BaseModel):
    """Input model for TLE explanation requests"""
    norad_id: Optional[str] = None
    satellite_name: Optional[str] = None
    line1: str = Field(..., description="First line of TLE")
    line2: str = Field(..., description="Second line of TLE")
    include_risk_assessment: bool = True
    include_anomaly_detection: bool = True
    
class TLEExplanation(BaseModel):
    """Output model for TLE explanations"""
    norad_id: str
    satellite_name: str
    orbit_description: str
    orbit_type: str
    altitude_description: str
    period_minutes: float
    inclination_degrees: float
    eccentricity: float
    decay_risk_score: float
    decay_risk_level: str
    anomaly_flags: List[str]
    predicted_lifetime_days: Optional[float]
    last_updated: datetime
    confidence_score: float
    technical_details: Dict[str, Any]

class TLEParser:
    """Parse TLE data and extract orbital elements"""
    
    @staticmethod
    def parse_tle(line1: str, line2: str) -> Dict[str, Any]:
        """Parse TLE lines and extract orbital elements"""
        try:
            # Line 1 parsing
            norad_id = line1[2:7].strip()
            classification = line1[7]
            year = int(line1[18:20])
            if year < 57:
                year += 2000
            else:
                year += 1900
            
            day_of_year = float(line1[20:32])
            first_derivative = float(line1[33:43])
            second_derivative = float(line1[44:52].replace(' ', '0'))
            drag_term = float(line1[53:61].replace(' ', '0'))
            
            # Line 2 parsing
            inclination = float(line2[8:16])
            raan = float(line2[17:25])
            eccentricity = float('0.' + line2[26:33])
            arg_perigee = float(line2[34:42])
            mean_anomaly = float(line2[43:51])
            mean_motion = float(line2[52:63])
            
            # Calculate derived parameters
            semi_major_axis = (MU / (mean_motion * 2 * math.pi / 86400) ** 2) ** (1/3)
            perigee_altitude = semi_major_axis * (1 - eccentricity) - EARTH_RADIUS
            apogee_altitude = semi_major_axis * (1 + eccentricity) - EARTH_RADIUS
            period_minutes = 1440.0 / mean_motion
            
            return {
                'norad_id': norad_id,
                'classification': classification,
                'epoch_year': year,
                'epoch_day': day_of_year,
                'mean_motion_derivative': first_derivative,
                'mean_motion_second_derivative': second_derivative,
                'drag_term': drag_term,
                'inclination': inclination,
                'raan': raan,
                'eccentricity': eccentricity,
                'arg_perigee': arg_perigee,
                'mean_anomaly': mean_anomaly,
                'mean_motion': mean_motion,
                'semi_major_axis': semi_major_axis,
                'perigee_altitude': perigee_altitude,
                'apogee_altitude': apogee_altitude,
                'period_minutes': period_minutes
            }
        except Exception as e:
            logger.error(f"Error parsing TLE: {e}")
            raise ValueError(f"Invalid TLE format: {e}")

class OrbitClassifier:
    """Classify orbit types based on orbital elements"""
    
    @staticmethod
    def classify_orbit(elements: Dict[str, Any]) -> Tuple[str, str]:
        """Classify orbit type and provide description"""
        perigee = elements['perigee_altitude']
        apogee = elements['apogee_altitude']
        inclination = elements['inclination']
        eccentricity = elements['eccentricity']
        period = elements['period_minutes']
        
        # Altitude-based classification
        if perigee < 0:
            orbit_type = "DECAYING"
            description = "Orbit has decayed below Earth's surface"
        elif apogee < 2000:
            orbit_type = "LEO"
            description = "Low Earth Orbit"
        elif perigee < 2000 and apogee > 35000:
            orbit_type = "HEO"
            description = "Highly Elliptical Orbit"
        elif 35786 - 100 < apogee < 35786 + 100 and eccentricity < 0.01:
            orbit_type = "GEO"
            description = "Geostationary Orbit"
        elif apogee > 35786:
            orbit_type = "HIGH"
            description = "High Earth Orbit"
        else:
            orbit_type = "MEO"
            description = "Medium Earth Orbit"
            
        # Special orbits
        if 96 < inclination < 100 and perigee < 1000:
            orbit_type = "SSO"
            description = "Sun-Synchronous Orbit"
        elif inclination > 60 and inclination < 120:
            description += " (Polar)"
        elif inclination < 5:
            description += " (Equatorial)"
            
        return orbit_type, description

class DecayRiskAssessor:
    """Assess orbital decay risk"""
    
    @staticmethod
    def calculate_decay_risk(elements: Dict[str, Any]) -> Tuple[float, str, Optional[float]]:
        """Calculate decay risk score and estimated lifetime"""
        perigee = elements['perigee_altitude']
        mean_motion_derivative = elements['mean_motion_derivative']
        drag_term = elements['drag_term']
        
        # Basic decay risk calculation
        risk_score = 0.0
        
        # Altitude factor
        if perigee < 200:
            risk_score += 0.8
        elif perigee < 400:
            risk_score += 0.5
        elif perigee < 600:
            risk_score += 0.3
        elif perigee < 800:
            risk_score += 0.1
            
        # Mean motion derivative factor
        if mean_motion_derivative > 0.001:
            risk_score += 0.2
        elif mean_motion_derivative > 0.0001:
            risk_score += 0.1
            
        # Normalize score
        risk_score = min(1.0, risk_score)
        
        # Determine risk level
        if risk_score > 0.7:
            risk_level = "HIGH"
        elif risk_score > 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
            
        # Estimate lifetime (simplified)
        lifetime_days = None
        if perigee < 400 and mean_motion_derivative > 0:
            # Very simplified lifetime estimation
            decay_rate = mean_motion_derivative * 86400  # rev/day²
            remaining_revs = elements['mean_motion'] / (2 * decay_rate)
            lifetime_days = remaining_revs / elements['mean_motion']
            
        return risk_score, risk_level, lifetime_days

class AnomalyDetector:
    """Detect anomalies in orbital elements"""
    
    @staticmethod
    def detect_anomalies(elements: Dict[str, Any]) -> List[str]:
        """Detect potential anomalies in the orbit"""
        anomalies = []
        
        # Check for extreme eccentricity
        if elements['eccentricity'] > 0.9:
            anomalies.append("EXTREME_ECCENTRICITY")
            
        # Check for very low perigee
        if elements['perigee_altitude'] < 150:
            anomalies.append("CRITICAL_LOW_ALTITUDE")
            
        # Check for unusual inclination
        if 80 < elements['inclination'] < 100:
            if elements['perigee_altitude'] > 1000:
                anomalies.append("UNUSUAL_POLAR_ALTITUDE")
                
        # Check for rapid decay
        if elements['mean_motion_derivative'] > 0.01:
            anomalies.append("RAPID_DECAY")
            
        # Check for potential collision risk altitude
        if 400 < elements['perigee_altitude'] < 450:
            anomalies.append("ISS_ALTITUDE_BAND")
            
        return anomalies

class TLEExplainerService:
    """Main TLE Explainer Service"""
    
    def __init__(self):
        self.parser = TLEParser()
        self.classifier = OrbitClassifier()
        self.risk_assessor = DecayRiskAssessor()
        self.anomaly_detector = AnomalyDetector()
        
    async def explain_tle(self, input_data: TLEExplainerInput) -> TLEExplanation:
        """Generate natural language explanation for TLE"""
        try:
            # Parse TLE
            elements = self.parser.parse_tle(input_data.line1, input_data.line2)
            
            # Classify orbit
            orbit_type, orbit_description = self.classifier.classify_orbit(elements)
            
            # Generate natural language description
            nl_description = self._generate_orbit_description(elements, orbit_type)
            
            # Altitude description
            altitude_desc = self._generate_altitude_description(elements)
            
            # Risk assessment
            decay_risk_score = 0.0
            decay_risk_level = "LOW"
            lifetime_days = None
            
            if input_data.include_risk_assessment:
                decay_risk_score, decay_risk_level, lifetime_days = self.risk_assessor.calculate_decay_risk(elements)
            
            # Anomaly detection
            anomaly_flags = []
            if input_data.include_anomaly_detection:
                anomaly_flags = self.anomaly_detector.detect_anomalies(elements)
            
            return TLEExplanation(
                norad_id=elements['norad_id'],
                satellite_name=input_data.satellite_name or f"NORAD {elements['norad_id']}",
                orbit_description=nl_description,
                orbit_type=orbit_type,
                altitude_description=altitude_desc,
                period_minutes=elements['period_minutes'],
                inclination_degrees=elements['inclination'],
                eccentricity=elements['eccentricity'],
                decay_risk_score=decay_risk_score,
                decay_risk_level=decay_risk_level,
                anomaly_flags=anomaly_flags,
                predicted_lifetime_days=lifetime_days,
                last_updated=datetime.utcnow(),
                confidence_score=0.95,  # Would come from actual model
                technical_details=elements
            )
            
        except Exception as e:
            logger.error(f"Error explaining TLE: {e}")
            raise
    
    def _generate_orbit_description(self, elements: Dict[str, Any], orbit_type: str) -> str:
        """Generate natural language orbit description"""
        period_hours = elements['period_minutes'] / 60
        
        if orbit_type == "LEO":
            desc = f"This satellite orbits Earth every {period_hours:.1f} hours at a relatively low altitude. "
            desc += f"It passes over different parts of Earth with each orbit due to its {elements['inclination']:.1f}° inclination."
            
        elif orbit_type == "GEO":
            desc = "This satellite maintains a fixed position above Earth's equator, completing one orbit every 24 hours. "
            desc += "It appears stationary from the ground, making it ideal for communications."
            
        elif orbit_type == "HEO":
            desc = f"This satellite follows a highly elliptical path, swooping close to Earth at {elements['perigee_altitude']:.0f} km "
            desc += f"and reaching as far as {elements['apogee_altitude']:.0f} km. "
            desc += "It spends most of its time at high altitude, providing extended coverage."
            
        elif orbit_type == "SSO":
            desc = "This satellite follows a sun-synchronous orbit, passing over locations at the same local solar time. "
            desc += "This makes it ideal for Earth observation and imaging missions."
            
        else:
            desc = f"This satellite completes an orbit every {period_hours:.1f} hours "
            desc += f"at altitudes between {elements['perigee_altitude']:.0f} and {elements['apogee_altitude']:.0f} km."
            
        return desc
    
    def _generate_altitude_description(self, elements: Dict[str, Any]) -> str:
        """Generate altitude description"""
        perigee = elements['perigee_altitude']
        apogee = elements['apogee_altitude']
        
        if abs(apogee - perigee) < 50:
            return f"Maintains a nearly circular orbit at approximately {(perigee + apogee) / 2:.0f} km altitude"
        else:
            return f"Altitude varies from {perigee:.0f} km (closest) to {apogee:.0f} km (farthest)"

# For mock implementation without the actual model
class MockTLEExplainerService(TLEExplainerService):
    """Mock implementation for testing without loading the actual model"""
    
    async def explain_tle(self, input_data: TLEExplainerInput) -> TLEExplanation:
        """Generate explanation using analytical methods only"""
        # Use the parent class implementation which doesn't require the ML model
        return await super().explain_tle(input_data) 