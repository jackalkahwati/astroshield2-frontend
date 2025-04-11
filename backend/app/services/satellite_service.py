from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import random
import uuid
from pydantic import BaseModel

class OrbitParameters(BaseModel):
    altitude: float
    inclination: float
    eccentricity: float
    period: float
    semi_major_axis: float
    raan: float  # Right Ascension of the Ascending Node
    arg_perigee: float  # Argument of Perigee

class SatelliteTelemetry(BaseModel):
    battery_level: float
    temperature_internal: float
    temperature_external: float
    signal_strength: float
    data_rate: float
    memory_usage: float
    attitude: Dict[str, float]
    last_update: datetime

class SatelliteStatus(BaseModel):
    id: str
    name: str
    status: str
    owner: str
    launch_date: datetime
    last_contact: datetime
    orbit_parameters: OrbitParameters
    telemetry: SatelliteTelemetry
    capabilities: Dict[str, Any]
    tags: List[str]

class StateVector(BaseModel):
    timestamp: datetime
    position: Dict[str, float]
    velocity: Dict[str, float]
    covariance: Optional[List[List[float]]] = None

class SatelliteTrajectory(BaseModel):
    satellite_id: str
    start_time: datetime
    end_time: datetime
    points: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None

class SatelliteService:
    """
    Service for managing satellite information and telemetry.
    This implementation provides mock data but follows the structure that would
    be used for actual implementations.
    """
    def __init__(self):
        # Mock database of satellites
        self._satellites = {
            "sat-001": self._generate_satellite("sat-001", "AstroShield-1", "operational", 500.5),
            "sat-002": self._generate_satellite("sat-002", "AstroShield-2", "operational", 510.2),
            "sat-003": self._generate_satellite("sat-003", "AstroShield-3", "maintenance", 495.8),
            "sat-004": self._generate_satellite("sat-004", "AstroShield-4", "operational", 515.0)
        }
    
    def _generate_satellite(self, sat_id: str, name: str, status: str, altitude: float) -> Dict[str, Any]:
        """Generate a mock satellite entry with realistic data"""
        now = datetime.utcnow()
        launch_date = now - timedelta(days=random.randint(90, 365))
        
        # Generate orbit parameters based on altitude
        orbit_period = 90 + (altitude / 100)  # Simplified calculation: higher orbit = longer period
        
        return {
            "id": sat_id,
            "name": name,
            "status": status,
            "owner": "AstroShield Corp",
            "launch_date": launch_date,
            "last_contact": now - timedelta(minutes=random.randint(5, 60)),
            "orbit_parameters": {
                "altitude": altitude,
                "inclination": 45.0 + random.uniform(-5.0, 5.0),
                "eccentricity": 0.001 + random.uniform(0.0, 0.002),
                "period": orbit_period,
                "semi_major_axis": 6378.0 + altitude,  # Earth radius + altitude
                "raan": random.uniform(0.0, 360.0),
                "arg_perigee": random.uniform(0.0, 360.0)
            },
            "telemetry": {
                "battery_level": random.uniform(75.0, 98.0),
                "temperature_internal": random.uniform(18.0, 25.0),
                "temperature_external": random.uniform(-20.0, 20.0),
                "signal_strength": random.uniform(85.0, 99.0),
                "data_rate": random.uniform(10.0, 50.0),
                "memory_usage": random.uniform(30.0, 70.0),
                "attitude": {
                    "roll": random.uniform(-1.0, 1.0),
                    "pitch": random.uniform(-1.0, 1.0),
                    "yaw": random.uniform(-1.0, 1.0)
                },
                "last_update": now - timedelta(minutes=random.randint(1, 15))
            },
            "capabilities": {
                "imaging": status == "operational",
                "communication_relay": status == "operational",
                "science_payload": status == "operational",
                "propulsion": {
                    "available": status == "operational",
                    "type": "electric",
                    "fuel_remaining": random.uniform(75.0, 95.0)
                }
            },
            "tags": ["operational" if status == "operational" else "maintenance", 
                     "leo", "earth-observation"]
        }
    
    async def get_satellites(self, status: Optional[str] = None, owner: Optional[str] = None) -> List[SatelliteStatus]:
        """
        Get list of all satellites, optionally filtered by status and owner
        """
        satellites = list(self._satellites.values())
        
        # Apply filters if provided
        if status:
            satellites = [s for s in satellites if s["status"] == status]
        
        if owner:
            satellites = [s for s in satellites if s["owner"] == owner]
            
        # Convert to Pydantic models
        return [SatelliteStatus(**s) for s in satellites]
    
    async def get_satellite(self, satellite_id: str) -> Optional[SatelliteStatus]:
        """
        Get specific satellite by ID
        """
        if satellite_id not in self._satellites:
            return None
            
        return SatelliteStatus(**self._satellites[satellite_id])
    
    async def get_satellite_telemetry(self, satellite_id: str, time_range: Optional[Dict[str, datetime]] = None) -> List[Dict[str, Any]]:
        """
        Get telemetry data for a satellite over a time period
        """
        if satellite_id not in self._satellites:
            return []
            
        # Generate mock telemetry data
        now = datetime.utcnow()
        
        # Default to last 24 hours if no time range specified
        start_time = time_range.get("start", now - timedelta(hours=24)) if time_range else now - timedelta(hours=24)
        end_time = time_range.get("end", now) if time_range else now
        
        # Generate hourly data points
        data_points = []
        current_time = start_time
        
        while current_time <= end_time:
            data_points.append({
                "timestamp": current_time.isoformat(),
                "battery_level": random.uniform(75.0, 98.0),
                "temperature_internal": random.uniform(18.0, 25.0),
                "temperature_external": random.uniform(-20.0, 20.0),
                "signal_strength": random.uniform(85.0, 99.0),
                "data_rate": random.uniform(10.0, 50.0),
                "memory_usage": random.uniform(30.0, 70.0),
                "attitude": {
                    "roll": random.uniform(-1.0, 1.0),
                    "pitch": random.uniform(-1.0, 1.0),
                    "yaw": random.uniform(-1.0, 1.0)
                }
            })
            current_time += timedelta(hours=1)
            
        return data_points
    
    async def get_satellite_trajectory(self, satellite_id: str, start_time: datetime, end_time: datetime) -> Optional[SatelliteTrajectory]:
        """
        Get predicted trajectory for a satellite over a time period
        """
        if satellite_id not in self._satellites:
            return None
            
        satellite = self._satellites[satellite_id]
        
        # Generate trajectory points (simplified - in reality would use orbital mechanics)
        points = []
        current_time = start_time
        
        # Simplified orbit calculation (circular orbit approximation)
        altitude = satellite["orbit_parameters"]["altitude"]
        period_minutes = satellite["orbit_parameters"]["period"]
        orbit_radius = 6378.0 + altitude  # Earth radius + altitude in km
        orbit_circumference = 2 * 3.14159 * orbit_radius
        speed = orbit_circumference / (period_minutes * 60)  # km/s
        
        # Generate trajectory points at 5-minute intervals
        while current_time <= end_time:
            # Calculate position based on time (simplified circular orbit)
            time_in_orbit = (current_time - start_time).total_seconds()
            angle = (time_in_orbit / (period_minutes * 60)) * 360.0  # Angle in degrees
            
            # Convert to Cartesian coordinates (simplified)
            x = orbit_radius * (0.5 + 0.5 * (angle % 360) / 360)  # Just to create some variation
            y = orbit_radius * (0.5 + 0.5 * ((angle + 120) % 360) / 360)
            z = orbit_radius * (0.5 + 0.5 * ((angle + 240) % 360) / 360)
            
            # Velocity vector (simplified) - should be tangential to the orbit
            vx = -speed * (0.5 + 0.5 * ((angle + 90) % 360) / 360)
            vy = -speed * (0.5 + 0.5 * ((angle + 210) % 360) / 360)
            vz = -speed * (0.5 + 0.5 * ((angle + 330) % 360) / 360)
            
            points.append({
                "timestamp": current_time.isoformat(),
                "position": {"x": x, "y": y, "z": z},
                "velocity": {"x": vx, "y": vy, "z": vz}
            })
            
            current_time += timedelta(minutes=5)
            
        return SatelliteTrajectory(
            satellite_id=satellite_id,
            start_time=start_time,
            end_time=end_time,
            points=points,
            metadata={
                "orbit_type": "LEO",
                "calculation_method": "simplified_circular",
                "period_minutes": period_minutes
            }
        )
    
    async def get_state_vector(self, satellite_id: str, timestamp: Optional[datetime] = None) -> Optional[StateVector]:
        """
        Get state vector (position and velocity) for a satellite at a specific time
        """
        if satellite_id not in self._satellites:
            return None
            
        # Use current time if not specified
        if not timestamp:
            timestamp = datetime.utcnow()
            
        satellite = self._satellites[satellite_id]
        
        # Simplified orbit calculation (circular orbit approximation)
        altitude = satellite["orbit_parameters"]["altitude"]
        period_minutes = satellite["orbit_parameters"]["period"]
        orbit_radius = 6378.0 + altitude  # Earth radius + altitude in km
        orbit_circumference = 2 * 3.14159 * orbit_radius
        speed = orbit_circumference / (period_minutes * 60)  # km/s
        
        # Calculate position based on time (simplified circular orbit)
        # This is a very simplified model, real implementations would use SGP4 or similar
        time_in_orbit = timestamp.timestamp() % (period_minutes * 60)
        angle = (time_in_orbit / (period_minutes * 60)) * 360.0  # Angle in degrees
        
        # Convert to Cartesian coordinates (simplified)
        x = orbit_radius * (0.5 + 0.5 * (angle % 360) / 360)
        y = orbit_radius * (0.5 + 0.5 * ((angle + 120) % 360) / 360)
        z = orbit_radius * (0.5 + 0.5 * ((angle + 240) % 360) / 360)
        
        # Velocity vector (simplified) - should be tangential to the orbit
        vx = -speed * (0.5 + 0.5 * ((angle + 90) % 360) / 360)
        vy = -speed * (0.5 + 0.5 * ((angle + 210) % 360) / 360)
        vz = -speed * (0.5 + 0.5 * ((angle + 330) % 360) / 360)
        
        # Generate a simple diagonal covariance matrix
        covariance = [
            [0.01, 0, 0, 0, 0, 0],
            [0, 0.01, 0, 0, 0, 0],
            [0, 0, 0.01, 0, 0, 0],
            [0, 0, 0, 0.001, 0, 0],
            [0, 0, 0, 0, 0.001, 0],
            [0, 0, 0, 0, 0, 0.001]
        ]
        
        return StateVector(
            timestamp=timestamp,
            position={"x": x, "y": y, "z": z},
            velocity={"x": vx, "y": vy, "z": vz},
            covariance=covariance
        )