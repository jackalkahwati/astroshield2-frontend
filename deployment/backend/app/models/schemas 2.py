from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum
import uuid


class OrbitType(str, Enum):
    """
    Orbit types for satellites.
    
    - LEO: Low Earth Orbit (160-2000 km)
    - MEO: Medium Earth Orbit (2000-35786 km)
    - GEO: Geostationary/Geosynchronous Earth Orbit (35786 km)
    - HEO: Highly Elliptical Orbit
    """
    LEO = "LEO"
    MEO = "MEO"
    GEO = "GEO"
    HEO = "HEO"


class OperationalStatus(str, Enum):
    """Operational status of a satellite."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DECOMMISSIONED = "decommissioned"
    UNKNOWN = "unknown"


class OrbitalParameters(BaseModel):
    """
    Orbital parameters defining a satellite's orbit.
    
    These parameters use the standard Keplerian elements for defining an orbit.
    """
    semi_major_axis: float = Field(..., 
        description="Semi-major axis of the orbit in kilometers",
        example=7000.0,
        gt=0)
    eccentricity: float = Field(..., 
        description="Eccentricity of the orbit (0 = circular, 0-1 = elliptical)",
        example=0.0001,
        ge=0, lt=1)
    inclination: float = Field(..., 
        description="Inclination of the orbit in degrees",
        example=51.6,
        ge=0, le=180)
    raan: float = Field(..., 
        description="Right Ascension of the Ascending Node (RAAN) in degrees",
        example=235.7,
        ge=0, lt=360)
    argument_of_perigee: float = Field(..., 
        description="Argument of perigee in degrees",
        example=90.0,
        ge=0, lt=360)
    mean_anomaly: float = Field(..., 
        description="Mean anomaly in degrees",
        example=0.0,
        ge=0, lt=360)
    epoch: datetime = Field(..., 
        description="Epoch time for the orbital elements",
        example="2023-05-15T12:00:00Z")
    
    class Config:
        schema_extra = {
            "example": {
                "semi_major_axis": 7000.0,
                "eccentricity": 0.0001,
                "inclination": 51.6,
                "raan": 235.7,
                "argument_of_perigee": 90.0,
                "mean_anomaly": 0.0,
                "epoch": "2023-05-15T12:00:00Z"
            }
        }


class SatelliteBase(BaseModel):
    """Base model for satellite data."""
    name: str = Field(..., 
        description="Satellite name",
        example="AstroShield-1")
    norad_id: Optional[str] = Field(None, 
        description="NORAD Catalog Number (SATCAT)",
        example="43657")
    international_designator: Optional[str] = Field(None, 
        description="International Designator (COSPAR ID)",
        example="2018-099A")
    orbit_type: OrbitType = Field(..., 
        description="Type of orbit (LEO, MEO, GEO, HEO)")
    launch_date: Optional[datetime] = Field(None, 
        description="Launch date of the satellite",
        example="2023-05-15T00:00:00Z")
    operational_status: OperationalStatus = Field(OperationalStatus.UNKNOWN, 
        description="Current operational status")
    owner: Optional[str] = Field(None, 
        description="Owner or operator of the satellite",
        example="AstroShield Inc.")
    mission: Optional[str] = Field(None, 
        description="Primary mission of the satellite",
        example="Space Domain Awareness")


class SatelliteCreate(SatelliteBase):
    """Model for creating a new satellite"""
    orbital_parameters: OrbitalParameters = Field(..., 
        description="Orbital parameters of the satellite")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "AstroShield-1",
                "norad_id": "43657",
                "international_designator": "2018-099A",
                "orbit_type": "LEO",
                "launch_date": "2023-05-15T00:00:00Z",
                "operational_status": "active",
                "owner": "AstroShield Inc.",
                "mission": "Space Domain Awareness",
                "orbital_parameters": {
                    "semi_major_axis": 7000.0,
                    "eccentricity": 0.0001,
                    "inclination": 51.6,
                    "raan": 235.7,
                    "argument_of_perigee": 90.0,
                    "mean_anomaly": 0.0,
                    "epoch": "2023-05-15T12:00:00Z"
                }
            }
        }


class Satellite(SatelliteBase):
    """Model for satellite data retrieved from database"""
    id: str = Field(..., 
        description="Unique identifier of the satellite",
        example="sat-001")
    orbital_parameters: OrbitalParameters
    created_at: datetime = Field(..., 
        description="Timestamp when the satellite was created")
    updated_at: datetime = Field(..., 
        description="Timestamp when the satellite was last updated")
    
    class Config:
        orm_mode = True
        schema_extra = {
            "example": {
                "id": "sat-001",
                "name": "AstroShield-1",
                "norad_id": "43657",
                "international_designator": "2018-099A",
                "orbit_type": "LEO",
                "launch_date": "2023-05-15T00:00:00Z",
                "operational_status": "active",
                "owner": "AstroShield Inc.",
                "mission": "Space Domain Awareness",
                "orbital_parameters": {
                    "semi_major_axis": 7000.0,
                    "eccentricity": 0.0001,
                    "inclination": 51.6,
                    "raan": 235.7,
                    "argument_of_perigee": 90.0,
                    "mean_anomaly": 0.0,
                    "epoch": "2023-05-15T12:00:00Z"
                },
                "created_at": "2023-05-15T12:00:00Z",
                "updated_at": "2023-05-15T12:00:00Z"
            }
        }


class SatelliteUpdate(BaseModel):
    """Model for updating satellite information"""
    name: Optional[str] = Field(None, description="Satellite name")
    norad_id: Optional[str] = Field(None, description="NORAD Catalog Number (SATCAT)")
    international_designator: Optional[str] = Field(None, description="International Designator (COSPAR ID)")
    orbit_type: Optional[OrbitType] = Field(None, description="Type of orbit")
    launch_date: Optional[datetime] = Field(None, description="Launch date of the satellite")
    operational_status: Optional[OperationalStatus] = Field(None, description="Current operational status")
    owner: Optional[str] = Field(None, description="Owner or operator of the satellite")
    mission: Optional[str] = Field(None, description="Primary mission of the satellite")
    orbital_parameters: Optional[OrbitalParameters] = Field(None, description="Orbital parameters of the satellite")


class DirectionVector(BaseModel):
    """3D Vector for maneuver direction"""
    x: float = Field(..., description="X component", example=0.1)
    y: float = Field(..., description="Y component", example=0.0)
    z: float = Field(..., description="Z component", example=-0.1)
    
    class Config:
        schema_extra = {
            "example": {
                "x": 0.1,
                "y": 0.0,
                "z": -0.1
            }
        }


class ManeuverType(str, Enum):
    """Types of satellite maneuvers"""
    COLLISION_AVOIDANCE = "collision_avoidance"
    STATION_KEEPING = "station_keeping"
    ORBIT_RAISING = "orbit_raising"
    ORBIT_LOWERING = "orbit_lowering"
    INCLINATION_CHANGE = "inclination_change"
    DEORBIT = "deorbit"
    OTHER = "other"


class ManeuverStatus(str, Enum):
    """Status of a maneuver"""
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class ManeuverParameters(BaseModel):
    """Parameters defining a satellite maneuver"""
    delta_v: float = Field(..., 
        description="Delta-v (change in velocity) in km/s",
        example=0.02,
        gt=0)
    burn_duration: float = Field(..., 
        description="Burn duration in seconds",
        example=15.0,
        gt=0)
    direction: DirectionVector = Field(..., 
        description="Direction vector for the maneuver")
    target_orbit: Optional[Dict[str, float]] = Field(None, 
        description="Target orbital parameters after the maneuver")
    
    class Config:
        schema_extra = {
            "example": {
                "delta_v": 0.02,
                "burn_duration": 15.0,
                "direction": {
                    "x": 0.1,
                    "y": 0.0,
                    "z": -0.1
                },
                "target_orbit": {
                    "altitude": 500.2,
                    "inclination": 45.0,
                    "eccentricity": 0.001
                }
            }
        }


class ManeuverRequest(BaseModel):
    """Request model for creating a maneuver"""
    satellite_id: str = Field(..., 
        description="ID of the satellite to maneuver",
        example="sat-001")
    type: ManeuverType = Field(..., 
        description="Type of maneuver to perform")
    scheduled_start_time: datetime = Field(..., 
        description="Scheduled start time for the maneuver",
        example="2023-06-15T20:00:00Z")
    parameters: ManeuverParameters = Field(..., 
        description="Parameters for the maneuver")
    
    class Config:
        schema_extra = {
            "example": {
                "satellite_id": "sat-001",
                "type": "collision_avoidance",
                "scheduled_start_time": "2023-06-15T20:00:00Z",
                "parameters": {
                    "delta_v": 0.02,
                    "burn_duration": 15.0,
                    "direction": {
                        "x": 0.1,
                        "y": 0.0,
                        "z": -0.1
                    },
                    "target_orbit": {
                        "altitude": 500.2,
                        "inclination": 45.0,
                        "eccentricity": 0.001
                    }
                }
            }
        }


class ManeuverResources(BaseModel):
    """Resource information for a satellite's maneuver capabilities"""
    fuel_remaining: float = Field(..., 
        description="Percentage of fuel remaining",
        example=85.5,
        ge=0, le=100)
    power_available: float = Field(..., 
        description="Percentage of power available for maneuvers",
        example=90.0,
        ge=0, le=100)
    thruster_status: str = Field(..., 
        description="Status of the satellite's thrusters",
        example="nominal")
    
    class Config:
        schema_extra = {
            "example": {
                "fuel_remaining": 85.5,
                "power_available": 90.0,
                "thruster_status": "nominal"
            }
        }


class ManeuverStatus(BaseModel):
    """Full maneuver information with status"""
    id: str = Field(..., 
        description="Unique identifier of the maneuver",
        example="mnv-001")
    satellite_id: str = Field(..., 
        description="ID of the satellite being maneuvered",
        example="sat-001")
    status: str = Field(..., 
        description="Current status of the maneuver",
        example="completed")
    type: str = Field(..., 
        description="Type of maneuver",
        example="collision_avoidance")
    start_time: datetime = Field(..., 
        description="Scheduled/actual start time",
        example="2023-06-15T20:00:00Z")
    end_time: Optional[datetime] = Field(None, 
        description="Actual end time (if completed)")
    resources: ManeuverResources = Field(..., 
        description="Resource usage information")
    parameters: ManeuverParameters = Field(..., 
        description="Parameters for the maneuver")
    created_by: str = Field(..., 
        description="User who created the maneuver",
        example="user@example.com")
    created_at: datetime = Field(..., 
        description="Timestamp when the maneuver was created")
    updated_at: Optional[datetime] = Field(None, 
        description="Timestamp when the maneuver was last updated")
    
    class Config:
        orm_mode = True
        schema_extra = {
            "example": {
                "id": "mnv-001",
                "satellite_id": "sat-001",
                "status": "completed",
                "type": "collision_avoidance",
                "start_time": "2023-06-15T20:00:00Z",
                "end_time": "2023-06-15T20:15:00Z",
                "resources": {
                    "fuel_remaining": 85.5,
                    "power_available": 90.0,
                    "thruster_status": "nominal"
                },
                "parameters": {
                    "delta_v": 0.02,
                    "burn_duration": 15.0,
                    "direction": {
                        "x": 0.1,
                        "y": 0.0,
                        "z": -0.1
                    },
                    "target_orbit": {
                        "altitude": 500.2,
                        "inclination": 45.0,
                        "eccentricity": 0.001
                    }
                },
                "created_by": "user@example.com",
                "created_at": "2023-06-14T12:00:00Z",
                "updated_at": "2023-06-15T20:15:00Z"
            }
        } 