from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

class ObjectStabilityData(BaseModel):
    """Model for object stability metrics"""
    timestamp: datetime
    position: List[float] = Field(..., description="[x, y, z] position in km")
    velocity: List[float] = Field(..., description="[vx, vy, vz] velocity in km/s")
    orbital_elements: Dict[str, float] = Field(..., description="Keplerian orbital elements")
    stability_metrics: Dict[str, float] = Field(..., description="Stability metrics like eccentricity variation")

class ManeuverData(BaseModel):
    """Model for maneuver detection data"""
    timestamp: datetime
    delta_v: List[float] = Field(..., description="[dvx, dvy, dvz] velocity change in m/s")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in maneuver detection")
    type: str = Field(..., description="Type of maneuver detected")
    sensor_coverage: Dict[str, Any] = Field(..., description="Sensor coverage during maneuver")

class SignatureData(BaseModel):
    """Model for object signature data"""
    class SignatureType(str, Enum):
        RF = "rf"
        OPTICAL = "optical"
        RADAR = "radar"
        
    timestamp: datetime
    type: SignatureType
    magnitude: float = Field(..., description="Signal strength or visual magnitude")
    frequency: Optional[float] = Field(None, description="RF frequency in MHz if applicable")
    bandwidth: Optional[float] = Field(None, description="RF bandwidth in MHz if applicable")
    cross_section: Optional[float] = Field(None, description="Radar cross section in m² if applicable")
    confidence: float = Field(..., ge=0, le=1)

class BehaviorData(BaseModel):
    """Model for object behavior analysis"""
    timestamp: datetime
    pattern_of_life: Dict[str, Any] = Field(..., description="Normal behavior patterns")
    anomalies: List[Dict[str, Any]] = Field(default_factory=list, description="Detected anomalies")
    proximity_events: List[Dict[str, Any]] = Field(default_factory=list, description="Close approach events")
    imaging_opportunities: List[Dict[str, Any]] = Field(default_factory=list, description="Potential imaging passes")

class PhysicalCharacteristics(BaseModel):
    """Model for object physical characteristics"""
    timestamp: datetime
    area_to_mass_ratio: float = Field(..., gt=0, description="Area-to-mass ratio in m²/kg")
    size_estimate: Dict[str, float] = Field(..., description="Estimated dimensions in meters")
    attitude_state: Optional[Dict[str, Any]] = Field(None, description="Attitude and tumble state if known")
    subsatellite_count: int = Field(0, ge=0, description="Number of detected subsatellites")

class ComplianceData(BaseModel):
    """Model for regulatory compliance data"""
    timestamp: datetime
    itu_filing: Optional[Dict[str, Any]] = Field(None, description="ITU filing details if available")
    fcc_filing: Optional[Dict[str, Any]] = Field(None, description="FCC filing details if available")
    un_registry: bool = Field(..., description="Present in UN registry")
    launch_data: Dict[str, Any] = Field(..., description="Launch site and vehicle information")
    regulatory_violations: List[Dict[str, Any]] = Field(default_factory=list)

class CCDMAnalysisResult(BaseModel):
    """Comprehensive model for CCDM analysis results"""
    object_id: str
    analysis_timestamp: datetime
    stability: List[ObjectStabilityData] = Field(default_factory=list)
    maneuvers: List[ManeuverData] = Field(default_factory=list)
    signatures: List[SignatureData] = Field(default_factory=list)
    behavior: BehaviorData
    characteristics: PhysicalCharacteristics
    compliance: ComplianceData
    confidence_score: float = Field(..., ge=0, le=1, description="Overall confidence in analysis")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        schema_extra = {
            "example": {
                "object_id": "2023-001A",
                "analysis_timestamp": "2023-01-01T00:00:00Z",
                "stability": [{
                    "timestamp": "2023-01-01T00:00:00Z",
                    "position": [42164.0, 0.0, 0.0],
                    "velocity": [-3.075, 0.0, 0.0],
                    "orbital_elements": {
                        "semi_major_axis": 42164.0,
                        "eccentricity": 0.0,
                        "inclination": 0.0
                    },
                    "stability_metrics": {
                        "eccentricity_variation": 0.001
                    }
                }],
                "maneuvers": [{
                    "timestamp": "2023-01-02T00:00:00Z",
                    "delta_v": [10.0, 0.0, 0.0],
                    "confidence": 0.95,
                    "type": "station_keeping",
                    "sensor_coverage": {
                        "optical": True,
                        "radar": False
                    }
                }],
                "signatures": [{
                    "timestamp": "2023-01-01T00:00:00Z",
                    "type": "rf",
                    "magnitude": -10.0,
                    "frequency": 14500.0,
                    "bandwidth": 36.0,
                    "confidence": 0.9
                }],
                "behavior": {
                    "timestamp": "2023-01-01T00:00:00Z",
                    "pattern_of_life": {
                        "station_keeping_frequency": "daily",
                        "typical_maneuver_size": 10.0
                    },
                    "anomalies": [],
                    "proximity_events": [],
                    "imaging_opportunities": []
                },
                "characteristics": {
                    "timestamp": "2023-01-01T00:00:00Z",
                    "area_to_mass_ratio": 0.02,
                    "size_estimate": {
                        "length": 5.0,
                        "width": 3.0,
                        "height": 2.0
                    },
                    "attitude_state": {
                        "spin_rate": 0.1,
                        "attitude_stable": True
                    },
                    "subsatellite_count": 0
                },
                "compliance": {
                    "timestamp": "2023-01-01T00:00:00Z",
                    "itu_filing": {
                        "filing_id": "ABC123",
                        "status": "active"
                    },
                    "fcc_filing": {
                        "filing_id": "XYZ789",
                        "status": "active"
                    },
                    "un_registry": True,
                    "launch_data": {
                        "site": "Cape Canaveral",
                        "vehicle": "Falcon 9"
                    },
                    "regulatory_violations": []
                },
                "confidence_score": 0.95,
                "metadata": {
                    "analyst": "automated_system",
                    "analysis_version": "1.0.0"
                }
            }
        }
