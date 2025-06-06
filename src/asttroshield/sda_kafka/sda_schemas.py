"""
SDA-Specific Message Schemas
Official schemas for SDA Kafka message bus integration
Based on SDA Welders Arc GitLab repository schemas
"""

import json
from typing import Optional, List, Any, Dict
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict

try:
    from pydantic import BaseModel, Field, validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False


if PYDANTIC_AVAILABLE:
    class SDAManeuverDetected(BaseModel):
        """
        SDA SS4 maneuvers-detected schema
        Official schema from SDA Welders Arc GitLab repository
        """
        # Required fields
        source: str = Field(description="Source system identifier")
        satNo: str = Field(description="Satellite catalog number")
        
        # Optional timestamp fields
        createdAt: Optional[datetime] = Field(None, description="Record creation timestamp")
        eventStartTime: Optional[datetime] = Field(None, description="Maneuver start time")
        eventStopTime: Optional[datetime] = Field(None, description="Maneuver stop time")
        
        # Pre-maneuver state
        preCov: Optional[List[List[float]]] = Field(None, description="Pre-maneuver covariance matrix")
        prePosX: Optional[float] = Field(None, description="Pre-maneuver position X (km)")
        prePosY: Optional[float] = Field(None, description="Pre-maneuver position Y (km)")
        prePosZ: Optional[float] = Field(None, description="Pre-maneuver position Z (km)")
        preVelX: Optional[float] = Field(None, description="Pre-maneuver velocity X (km/s)")
        preVelY: Optional[float] = Field(None, description="Pre-maneuver velocity Y (km/s)")
        preVelZ: Optional[float] = Field(None, description="Pre-maneuver velocity Z (km/s)")
        
        # Post-maneuver state
        postCov: Optional[List[List[float]]] = Field(None, description="Post-maneuver covariance matrix")
        postPosX: Optional[float] = Field(None, description="Post-maneuver position X (km)")
        postPosY: Optional[float] = Field(None, description="Post-maneuver position Y (km)")
        postPosZ: Optional[float] = Field(None, description="Post-maneuver position Z (km)")
        postVelX: Optional[float] = Field(None, description="Post-maneuver velocity X (km/s)")
        postVelY: Optional[float] = Field(None, description="Post-maneuver velocity Y (km/s)")
        postVelZ: Optional[float] = Field(None, description="Post-maneuver velocity Z (km/s)")
        
        @validator('createdAt', 'eventStartTime', 'eventStopTime', pre=True)
        def validate_datetime(cls, v):
            if isinstance(v, str):
                try:
                    return datetime.fromisoformat(v.replace('Z', '+00:00'))
                except ValueError:
                    return datetime.fromisoformat(v)
            return v
        
        @validator('preCov', 'postCov', pre=True)
        def validate_covariance(cls, v):
            if v is not None and not isinstance(v, list):
                raise ValueError("Covariance matrix must be a list of lists")
            return v
        
        class Config:
            json_encoders = {
                datetime: lambda v: v.isoformat()
            }
    
    class SDALaunchDetected(BaseModel):
        """
        SDA SS5 launch-detected schema
        For launch detection events
        """
        source: str = Field(description="Source system identifier")
        launchSite: Optional[str] = Field(None, description="Launch site identifier")
        vehicleId: Optional[str] = Field(None, description="Launch vehicle identifier")
        payloadId: Optional[str] = Field(None, description="Payload identifier")
        launchTime: Optional[datetime] = Field(None, description="Launch timestamp")
        confidence: Optional[float] = Field(None, ge=0, le=1, description="Detection confidence")
        
        @validator('launchTime', pre=True)
        def validate_datetime(cls, v):
            if isinstance(v, str):
                try:
                    return datetime.fromisoformat(v.replace('Z', '+00:00'))
                except ValueError:
                    return datetime.fromisoformat(v)
            return v
        
        class Config:
            json_encoders = {
                datetime: lambda v: v.isoformat()
            }
    
    class SDATLEUpdate(BaseModel):
        """
        SDA SS1 TLE update schema
        For TLE data updates and orbital analysis
        """
        source: str = Field(description="Source system identifier")
        satelliteId: str = Field(description="Satellite identifier")
        catalogNumber: Optional[str] = Field(None, description="NORAD catalog number")
        
        # TLE data
        line1: Optional[str] = Field(None, description="TLE line 1")
        line2: Optional[str] = Field(None, description="TLE line 2")
        epoch: Optional[datetime] = Field(None, description="TLE epoch")
        
        # Orbital elements
        inclination: Optional[float] = Field(None, description="Inclination (degrees)")
        eccentricity: Optional[float] = Field(None, description="Eccentricity")
        meanMotion: Optional[float] = Field(None, description="Mean motion (rev/day)")
        
        # Quality metrics
        accuracy: Optional[float] = Field(None, ge=0, le=1, description="TLE accuracy score")
        confidence: Optional[float] = Field(None, ge=0, le=1, description="Analysis confidence")
        
        @validator('epoch', pre=True)
        def validate_datetime(cls, v):
            if isinstance(v, str):
                try:
                    return datetime.fromisoformat(v.replace('Z', '+00:00'))
                except ValueError:
                    return datetime.fromisoformat(v)
            return v
        
        class Config:
            json_encoders = {
                datetime: lambda v: v.isoformat()
            }
    
    class SDALaunchIntentAssessment(BaseModel):
        """
        SDA SS5 launch intent assessment schema
        For assessing the intent and threat level of detected launches
        """
        source: str = Field(description="Source system identifier")
        launchId: str = Field(description="Launch event identifier")
        
        # Intent assessment
        intentCategory: Optional[str] = Field(None, description="Intent category (benign, surveillance, hostile)")
        threatLevel: Optional[str] = Field(None, description="Threat level (low, medium, high, critical)")
        hostilityScore: Optional[float] = Field(None, ge=0, le=1, description="Hostility score 0-1")
        confidence: Optional[float] = Field(None, ge=0, le=1, description="Assessment confidence")
        
        # Target analysis
        potentialTargets: Optional[List[str]] = Field(None, description="List of potential target assets")
        targetType: Optional[str] = Field(None, description="Type of target (satellite, station, debris)")
        
        # Threat indicators
        threatIndicators: Optional[List[str]] = Field(None, description="List of threat indicators")
        asatCapability: Optional[bool] = Field(None, description="Anti-satellite capability assessment")
        coplanarThreat: Optional[bool] = Field(None, description="Coplanar threat assessment")
        
        # Assessment metadata
        assessmentTime: Optional[datetime] = Field(None, description="Assessment timestamp")
        analystId: Optional[str] = Field(None, description="Analyst identifier")
        
        @validator('assessmentTime', pre=True)
        def validate_datetime(cls, v):
            if isinstance(v, str):
                try:
                    return datetime.fromisoformat(v.replace('Z', '+00:00'))
                except ValueError:
                    return datetime.fromisoformat(v)
            return v
        
        class Config:
            json_encoders = {
                datetime: lambda v: v.isoformat()
            }
    
    class SDAPezWezPrediction(BaseModel):
        """
        SDA SS5 PEZ-WEZ prediction schema
        For Probability of Engagement Zone - Weapon Engagement Zone predictions
        """
        source: str = Field(description="Source system identifier")
        threatId: str = Field(description="Threat object identifier")
        
        # PEZ-WEZ parameters
        weaponType: Optional[str] = Field(None, description="Weapon type (kkv, grappler, rf, eo)")
        pezRadius: Optional[float] = Field(None, description="PEZ radius in km")
        wezRadius: Optional[float] = Field(None, description="WEZ radius in km")
        
        # Engagement predictions
        engagementProbability: Optional[float] = Field(None, ge=0, le=1, description="Engagement probability")
        timeToEngagement: Optional[float] = Field(None, description="Time to engagement in seconds")
        engagementWindow: Optional[List[datetime]] = Field(None, description="Engagement time window [start, end]")
        
        # Target information
        targetAssets: Optional[List[str]] = Field(None, description="List of assets at risk")
        primaryTarget: Optional[str] = Field(None, description="Primary target identifier")
        
        # Prediction metadata
        predictionTime: Optional[datetime] = Field(None, description="Prediction timestamp")
        validityPeriod: Optional[float] = Field(None, description="Prediction validity in hours")
        confidence: Optional[float] = Field(None, ge=0, le=1, description="Prediction confidence")
        
        @validator('predictionTime', pre=True)
        def validate_datetime(cls, v):
            if isinstance(v, str):
                try:
                    return datetime.fromisoformat(v.replace('Z', '+00:00'))
                except ValueError:
                    return datetime.fromisoformat(v)
            return v
        
        class Config:
            json_encoders = {
                datetime: lambda v: v.isoformat()
            }
    
    class SDAASATAssessment(BaseModel):
        """
        SDA SS5 ASAT (Anti-Satellite) assessment schema
        For assessing anti-satellite weapon capabilities and threats
        """
        source: str = Field(description="Source system identifier")
        threatId: str = Field(description="Threat object/launch identifier")
        
        # ASAT assessment
        asatType: Optional[str] = Field(None, description="ASAT type (kinetic, directed_energy, cyber, jamming)")
        asatCapability: Optional[bool] = Field(None, description="ASAT capability confirmed")
        threatLevel: Optional[str] = Field(None, description="Threat level (low, medium, high, imminent)")
        
        # Target analysis
        targetedAssets: Optional[List[str]] = Field(None, description="List of threatened assets")
        orbitRegimesThreatened: Optional[List[str]] = Field(None, description="Orbit regimes under threat")
        interceptCapability: Optional[bool] = Field(None, description="Intercept capability assessment")
        
        # Technical assessment
        maxReachAltitude: Optional[float] = Field(None, description="Maximum reach altitude in km")
        effectiveRange: Optional[float] = Field(None, description="Effective range in km")
        launchToImpact: Optional[float] = Field(None, description="Launch to impact time in minutes")
        
        # Assessment metadata
        assessmentTime: Optional[datetime] = Field(None, description="Assessment timestamp")
        confidence: Optional[float] = Field(None, ge=0, le=1, description="Assessment confidence")
        intelligence_sources: Optional[List[str]] = Field(None, description="Intelligence sources used")
        
        @validator('assessmentTime', pre=True)
        def validate_datetime(cls, v):
            if isinstance(v, str):
                try:
                    return datetime.fromisoformat(v.replace('Z', '+00:00'))
                except ValueError:
                    return datetime.fromisoformat(v)
            return v
        
        class Config:
            json_encoders = {
                datetime: lambda v: v.isoformat()
            }
    
    class SDASS2StateVector(BaseModel):
        """
        SDA SS2 State Vector schema
        For RSO state vectors with covariance data
        """
        source: str = Field(description="Source system identifier")
        objectId: str = Field(description="RSO object identifier")
        
        # State vector components
        position: List[float] = Field(description="Position vector [x, y, z] in km")
        velocity: List[float] = Field(description="Velocity vector [vx, vy, vz] in km/s")
        epoch: datetime = Field(description="State epoch timestamp")
        
        # Covariance matrix (6x6 for position and velocity)
        covariance: Optional[List[List[float]]] = Field(None, description="6x6 covariance matrix")
        
        # Metadata
        coordinateFrame: Optional[str] = Field("ITRF", description="Coordinate reference frame")
        dataSource: Optional[str] = Field(None, description="Data source or sensor")
        qualityMetric: Optional[float] = Field(None, ge=0, le=1, description="State quality metric")
        propagatedFrom: Optional[datetime] = Field(None, description="Original observation time")
        
        @validator('epoch', 'propagatedFrom', pre=True)
        def validate_datetime(cls, v):
            if isinstance(v, str):
                try:
                    return datetime.fromisoformat(v.replace('Z', '+00:00'))
                except ValueError:
                    return datetime.fromisoformat(v)
            return v
        
        @validator('position', 'velocity', pre=True)
        def validate_vectors(cls, v):
            if len(v) != 3:
                raise ValueError("Position and velocity vectors must have 3 components")
            return v
        
        @validator('covariance', pre=True)
        def validate_covariance(cls, v):
            if v is not None:
                if len(v) != 6 or any(len(row) != 6 for row in v):
                    raise ValueError("Covariance matrix must be 6x6")
            return v
        
        class Config:
            json_encoders = {
                datetime: lambda v: v.isoformat()
            }
    
    class SDASS2Elset(BaseModel):
        """
        SDA SS2 Elset (TLE) schema
        For Two-Line Element sets and orbital parameters
        """
        source: str = Field(description="Source system identifier")
        objectId: str = Field(description="RSO object identifier")
        
        # TLE data
        catalogNumber: Optional[str] = Field(None, description="NORAD catalog number")
        classification: Optional[str] = Field("U", description="Classification (U/C/S)")
        intlDesignator: Optional[str] = Field(None, description="International designator")
        epoch: datetime = Field(description="Element set epoch")
        meanMotion: float = Field(description="Mean motion (revs/day)")
        eccentricity: float = Field(description="Eccentricity")
        inclination: float = Field(description="Inclination (degrees)")
        argOfPerigee: float = Field(description="Argument of perigee (degrees)")
        raan: float = Field(description="Right ascension of ascending node (degrees)")
        meanAnomaly: float = Field(description="Mean anomaly (degrees)")
        
        # TLE lines (if available)
        line1: Optional[str] = Field(None, description="TLE line 1")
        line2: Optional[str] = Field(None, description="TLE line 2")
        
        # Quality and metadata
        rcsSize: Optional[str] = Field(None, description="RCS size category (SMALL/MEDIUM/LARGE)")
        objectType: Optional[str] = Field(None, description="Object type classification")
        dataSource: Optional[str] = Field(None, description="Data source")
        qualityMetric: Optional[float] = Field(None, ge=0, le=1, description="Elset quality metric")
        
        @validator('epoch', pre=True)
        def validate_datetime(cls, v):
            if isinstance(v, str):
                try:
                    return datetime.fromisoformat(v.replace('Z', '+00:00'))
                except ValueError:
                    return datetime.fromisoformat(v)
            return v
        
        class Config:
            json_encoders = {
                datetime: lambda v: v.isoformat()
            }
    
    class SDASS6ResponseRecommendation(BaseModel):
        """
        SDA SS6 Response Recommendation schema
        For course of action recommendations and tactics
        """
        source: str = Field(description="Source system identifier")
        threatId: str = Field(description="Threat identifier")
        responseId: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Response recommendation ID")
        
        # Threat context
        threatType: str = Field(description="Type of threat (launch, on-orbit, maneuver, etc.)")
        threatLevel: str = Field(description="Threat level (low, medium, high, critical)")
        threatenedAssets: List[str] = Field(description="List of assets under threat")
        
        # Response recommendations
        primaryCOA: str = Field(description="Primary course of action recommendation")
        alternateCOAs: Optional[List[str]] = Field(None, description="Alternative courses of action")
        tacticsAndProcedures: Optional[List[str]] = Field(None, description="Recommended tactics and procedures")
        
        # Timing and priority
        priority: str = Field(description="Response priority (immediate, urgent, routine)")
        timeToImplement: Optional[float] = Field(None, description="Estimated implementation time (minutes)")
        effectiveWindow: Optional[List[datetime]] = Field(None, description="Effective response window [start, end]")
        
        # Confidence and rationale
        confidence: float = Field(ge=0, le=1, description="Recommendation confidence")
        rationale: Optional[str] = Field(None, description="Reasoning for recommendation")
        riskAssessment: Optional[str] = Field(None, description="Risk assessment summary")
        
        # Metadata
        recommendationTime: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
        analystId: Optional[str] = Field(None, description="Analyst or system identifier")
        
        @validator('recommendationTime', pre=True)
        def validate_datetime(cls, v):
            if isinstance(v, str):
                try:
                    return datetime.fromisoformat(v.replace('Z', '+00:00'))
                except ValueError:
                    return datetime.fromisoformat(v)
            return v
        
        class Config:
            json_encoders = {
                datetime: lambda v: v.isoformat()
            }
    
    class SDASS0WeatherData(BaseModel):
        """
        SDA SS0 Weather Data schema
        For environmental and weather data ingestion
        """
        source: str = Field(description="Source system identifier (e.g., EarthCast)")
        dataType: str = Field(description="Type of weather data")
        
        # Spatial coverage
        latitude: Optional[float] = Field(None, ge=-90, le=90, description="Latitude (degrees)")
        longitude: Optional[float] = Field(None, ge=-180, le=180, description="Longitude (degrees)")
        altitude: Optional[float] = Field(None, description="Altitude (km)")
        resolution: Optional[float] = Field(None, description="Spatial resolution (km)")
        
        # Temporal information
        timestamp: datetime = Field(description="Data timestamp")
        validTime: Optional[datetime] = Field(None, description="Data valid time")
        forecastTime: Optional[datetime] = Field(None, description="Forecast time (if applicable)")
        
        # Data values
        value: Optional[float] = Field(None, description="Primary data value")
        values: Optional[List[float]] = Field(None, description="Array of data values")
        units: Optional[str] = Field(None, description="Data units")
        
        # Quality indicators
        qualityFlag: Optional[str] = Field(None, description="Data quality flag")
        confidence: Optional[float] = Field(None, ge=0, le=1, description="Data confidence")
        
        @validator('timestamp', 'validTime', 'forecastTime', pre=True)
        def validate_datetime(cls, v):
            if isinstance(v, str):
                try:
                    return datetime.fromisoformat(v.replace('Z', '+00:00'))
                except ValueError:
                    return datetime.fromisoformat(v)
            return v
        
        class Config:
            json_encoders = {
                datetime: lambda v: v.isoformat()
            }

else:
    # Fallback implementations without Pydantic
    @dataclass
    class SDAManeuverDetected:
        """SDA SS4 maneuvers-detected schema (fallback)"""
        source: str
        satNo: str
        createdAt: Optional[str] = None
        eventStartTime: Optional[str] = None
        eventStopTime: Optional[str] = None
        preCov: Optional[List[List[float]]] = None
        prePosX: Optional[float] = None
        prePosY: Optional[float] = None
        prePosZ: Optional[float] = None
        preVelX: Optional[float] = None
        preVelY: Optional[float] = None
        preVelZ: Optional[float] = None
        postCov: Optional[List[List[float]]] = None
        postPosX: Optional[float] = None
        postPosY: Optional[float] = None
        postPosZ: Optional[float] = None
        postVelX: Optional[float] = None
        postVelY: Optional[float] = None
        postVelZ: Optional[float] = None
        
        def json(self):
            return json.dumps(asdict(self), default=str)
    
    @dataclass
    class SDALaunchDetected:
        """SDA SS5 launch-detected schema (fallback)"""
        source: str
        launchSite: Optional[str] = None
        vehicleId: Optional[str] = None
        payloadId: Optional[str] = None
        launchTime: Optional[str] = None
        confidence: Optional[float] = None
        
        def json(self):
            return json.dumps(asdict(self), default=str)
    
    @dataclass
    class SDATLEUpdate:
        """SDA SS1 TLE update schema (fallback)"""
        source: str
        satelliteId: str
        catalogNumber: Optional[str] = None
        line1: Optional[str] = None
        line2: Optional[str] = None
        epoch: Optional[str] = None
        inclination: Optional[float] = None
        eccentricity: Optional[float] = None
        meanMotion: Optional[float] = None
        accuracy: Optional[float] = None
        confidence: Optional[float] = None
        
        def json(self):
            return json.dumps(asdict(self), default=str)
    
    @dataclass
    class SDALaunchIntentAssessment:
        """SDA SS5 launch intent assessment schema (fallback)"""
        source: str
        launchId: str
        intentCategory: Optional[str] = None
        threatLevel: Optional[str] = None
        hostilityScore: Optional[float] = None
        confidence: Optional[float] = None
        potentialTargets: Optional[List[str]] = None
        targetType: Optional[str] = None
        threatIndicators: Optional[List[str]] = None
        asatCapability: Optional[bool] = None
        coplanarThreat: Optional[bool] = None
        assessmentTime: Optional[str] = None
        analystId: Optional[str] = None
        
        def json(self):
            return json.dumps(asdict(self), default=str)
    
    @dataclass
    class SDAPezWezPrediction:
        """SDA SS5 PEZ-WEZ prediction schema (fallback)"""
        source: str
        threatId: str
        weaponType: Optional[str] = None
        pezRadius: Optional[float] = None
        wezRadius: Optional[float] = None
        engagementProbability: Optional[float] = None
        timeToEngagement: Optional[float] = None
        engagementWindow: Optional[List[str]] = None
        targetAssets: Optional[List[str]] = None
        primaryTarget: Optional[str] = None
        predictionTime: Optional[str] = None
        validityPeriod: Optional[float] = None
        confidence: Optional[float] = None
        
        def json(self):
            return json.dumps(asdict(self), default=str)
    
    @dataclass
    class SDAASATAssessment:
        """SDA SS5 ASAT assessment schema (fallback)"""
        source: str
        threatId: str
        asatType: Optional[str] = None
        asatCapability: Optional[bool] = None
        threatLevel: Optional[str] = None
        targetedAssets: Optional[List[str]] = None
        orbitRegimesThreatened: Optional[List[str]] = None
        interceptCapability: Optional[bool] = None
        maxReachAltitude: Optional[float] = None
        effectiveRange: Optional[float] = None
        launchToImpact: Optional[float] = None
        assessmentTime: Optional[str] = None
        confidence: Optional[float] = None
        intelligence_sources: Optional[List[str]] = None
        
        def json(self):
            return json.dumps(asdict(self), default=str)
    
    @dataclass
    class SDASS2StateVector:
        """SDA SS2 State Vector schema (fallback)"""
        source: str
        objectId: str
        position: List[float]
        velocity: List[float]
        epoch: str
        covariance: Optional[List[List[float]]] = None
        coordinateFrame: Optional[str] = "ITRF"
        dataSource: Optional[str] = None
        qualityMetric: Optional[float] = None
        propagatedFrom: Optional[str] = None
        
        def json(self):
            return json.dumps(asdict(self), default=str)
    
    @dataclass  
    class SDASS2Elset:
        """SDA SS2 Elset schema (fallback)"""
        source: str
        objectId: str
        epoch: str
        meanMotion: float
        eccentricity: float
        inclination: float
        argOfPerigee: float
        raan: float
        meanAnomaly: float
        catalogNumber: Optional[str] = None
        classification: Optional[str] = "U"
        intlDesignator: Optional[str] = None
        line1: Optional[str] = None
        line2: Optional[str] = None
        rcsSize: Optional[str] = None
        objectType: Optional[str] = None
        dataSource: Optional[str] = None
        qualityMetric: Optional[float] = None
        
        def json(self):
            return json.dumps(asdict(self), default=str)
    
    @dataclass
    class SDASS6ResponseRecommendation:
        """SDA SS6 Response Recommendation schema (fallback)"""
        source: str
        threatId: str
        responseId: str
        threatType: str
        threatLevel: str
        threatenedAssets: List[str]
        primaryCOA: str
        priority: str
        confidence: float
        alternateCOAs: Optional[List[str]] = None
        tacticsAndProcedures: Optional[List[str]] = None
        timeToImplement: Optional[float] = None
        effectiveWindow: Optional[List[str]] = None
        rationale: Optional[str] = None
        riskAssessment: Optional[str] = None
        recommendationTime: Optional[str] = None
        analystId: Optional[str] = None
        
        def json(self):
            return json.dumps(asdict(self), default=str)
    
    @dataclass
    class SDASS0WeatherData:
        """SDA SS0 Weather Data schema (fallback)"""
        source: str
        dataType: str
        timestamp: str
        latitude: Optional[float] = None
        longitude: Optional[float] = None
        altitude: Optional[float] = None
        resolution: Optional[float] = None
        validTime: Optional[str] = None
        forecastTime: Optional[str] = None
        value: Optional[float] = None
        values: Optional[List[float]] = None
        units: Optional[str] = None
        qualityFlag: Optional[str] = None
        confidence: Optional[float] = None
        
        def json(self):
            return json.dumps(asdict(self), default=str)


class SDASchemaFactory:
    """Factory for creating SDA schema instances"""
    
    @staticmethod
    def create_maneuver_detected(
        satellite_id: str,
        source: str = "astroshield",
        pre_position: Optional[List[float]] = None,
        pre_velocity: Optional[List[float]] = None,
        post_position: Optional[List[float]] = None,
        post_velocity: Optional[List[float]] = None,
        event_start: Optional[datetime] = None,
        event_stop: Optional[datetime] = None,
        pre_covariance: Optional[List[List[float]]] = None,
        post_covariance: Optional[List[List[float]]] = None
    ) -> SDAManeuverDetected:
        """Create SDA maneuver detected message"""
        
        # Convert position/velocity lists to individual components
        pre_pos_x = pre_position[0] if pre_position and len(pre_position) > 0 else None
        pre_pos_y = pre_position[1] if pre_position and len(pre_position) > 1 else None
        pre_pos_z = pre_position[2] if pre_position and len(pre_position) > 2 else None
        
        pre_vel_x = pre_velocity[0] if pre_velocity and len(pre_velocity) > 0 else None
        pre_vel_y = pre_velocity[1] if pre_velocity and len(pre_velocity) > 1 else None
        pre_vel_z = pre_velocity[2] if pre_velocity and len(pre_velocity) > 2 else None
        
        post_pos_x = post_position[0] if post_position and len(post_position) > 0 else None
        post_pos_y = post_position[1] if post_position and len(post_position) > 1 else None
        post_pos_z = post_position[2] if post_position and len(post_position) > 2 else None
        
        post_vel_x = post_velocity[0] if post_velocity and len(post_velocity) > 0 else None
        post_vel_y = post_velocity[1] if post_velocity and len(post_velocity) > 1 else None
        post_vel_z = post_velocity[2] if post_velocity and len(post_velocity) > 2 else None
        
        return SDAManeuverDetected(
            source=source,
            satNo=satellite_id,
            createdAt=datetime.now(timezone.utc),
            eventStartTime=event_start,
            eventStopTime=event_stop,
            preCov=pre_covariance,
            prePosX=pre_pos_x,
            prePosY=pre_pos_y,
            prePosZ=pre_pos_z,
            preVelX=pre_vel_x,
            preVelY=pre_vel_y,
            preVelZ=pre_vel_z,
            postCov=post_covariance,
            postPosX=post_pos_x,
            postPosY=post_pos_y,
            postPosZ=post_pos_z,
            postVelX=post_vel_x,
            postVelY=post_vel_y,
            postVelZ=post_vel_z
        )
    
    @staticmethod
    def create_launch_detected(
        source: str = "astroshield",
        launch_site: Optional[str] = None,
        vehicle_id: Optional[str] = None,
        payload_id: Optional[str] = None,
        launch_time: Optional[datetime] = None,
        confidence: Optional[float] = None
    ) -> SDALaunchDetected:
        """Create SDA launch detected message"""
        return SDALaunchDetected(
            source=source,
            launchSite=launch_site,
            vehicleId=vehicle_id,
            payloadId=payload_id,
            launchTime=launch_time or datetime.now(timezone.utc),
            confidence=confidence
        )
    
    @staticmethod
    def create_tle_update(
        satellite_id: str,
        source: str = "astroshield",
        catalog_number: Optional[str] = None,
        line1: Optional[str] = None,
        line2: Optional[str] = None,
        epoch: Optional[datetime] = None,
        orbital_elements: Optional[Dict[str, float]] = None,
        accuracy: Optional[float] = None,
        confidence: Optional[float] = None
    ) -> SDATLEUpdate:
        """Create SDA TLE update message"""
        elements = orbital_elements or {}
        
        return SDATLEUpdate(
            source=source,
            satelliteId=satellite_id,
            catalogNumber=catalog_number,
            line1=line1,
            line2=line2,
            epoch=epoch,
            inclination=elements.get('inclination'),
            eccentricity=elements.get('eccentricity'),
            meanMotion=elements.get('mean_motion'),
            accuracy=accuracy,
            confidence=confidence
        )
    
    @staticmethod
    def create_launch_intent_assessment(
        launch_id: str,
        source: str = "astroshield",
        intent_category: Optional[str] = None,
        threat_level: Optional[str] = None,
        hostility_score: Optional[float] = None,
        confidence: Optional[float] = None,
        potential_targets: Optional[List[str]] = None,
        target_type: Optional[str] = None,
        threat_indicators: Optional[List[str]] = None,
        asat_capability: Optional[bool] = None,
        coplanar_threat: Optional[bool] = None,
        analyst_id: Optional[str] = None
    ) -> SDALaunchIntentAssessment:
        """Create SDA SS5 launch intent assessment message"""
        return SDALaunchIntentAssessment(
            source=source,
            launchId=launch_id,
            intentCategory=intent_category,
            threatLevel=threat_level,
            hostilityScore=hostility_score,
            confidence=confidence,
            potentialTargets=potential_targets,
            targetType=target_type,
            threatIndicators=threat_indicators,
            asatCapability=asat_capability,
            coplanarThreat=coplanar_threat,
            assessmentTime=datetime.now(timezone.utc),
            analystId=analyst_id
        )
    
    @staticmethod
    def create_pez_wez_prediction(
        threat_id: str,
        source: str = "astroshield",
        weapon_type: Optional[str] = None,
        pez_radius: Optional[float] = None,
        wez_radius: Optional[float] = None,
        engagement_probability: Optional[float] = None,
        time_to_engagement: Optional[float] = None,
        engagement_window: Optional[List[datetime]] = None,
        target_assets: Optional[List[str]] = None,
        primary_target: Optional[str] = None,
        validity_period: Optional[float] = None,
        confidence: Optional[float] = None
    ) -> SDAPezWezPrediction:
        """Create SDA SS5 PEZ-WEZ prediction message"""
        return SDAPezWezPrediction(
            source=source,
            threatId=threat_id,
            weaponType=weapon_type,
            pezRadius=pez_radius,
            wezRadius=wez_radius,
            engagementProbability=engagement_probability,
            timeToEngagement=time_to_engagement,
            engagementWindow=engagement_window,
            targetAssets=target_assets,
            primaryTarget=primary_target,
            predictionTime=datetime.now(timezone.utc),
            validityPeriod=validity_period,
            confidence=confidence
        )
    
    @staticmethod
    def create_asat_assessment(
        threat_id: str,
        source: str = "astroshield",
        asat_type: Optional[str] = None,
        asat_capability: Optional[bool] = None,
        threat_level: Optional[str] = None,
        targeted_assets: Optional[List[str]] = None,
        orbit_regimes_threatened: Optional[List[str]] = None,
        intercept_capability: Optional[bool] = None,
        max_reach_altitude: Optional[float] = None,
        effective_range: Optional[float] = None,
        launch_to_impact: Optional[float] = None,
        confidence: Optional[float] = None,
        intelligence_sources: Optional[List[str]] = None
    ) -> SDAASATAssessment:
        """Create SDA SS5 ASAT assessment message"""
        return SDAASATAssessment(
            source=source,
            threatId=threat_id,
            asatType=asat_type,
            asatCapability=asat_capability,
            threatLevel=threat_level,
            targetedAssets=targeted_assets,
            orbitRegimesThreatened=orbit_regimes_threatened,
            interceptCapability=intercept_capability,
            maxReachAltitude=max_reach_altitude,
            effectiveRange=effective_range,
            launchToImpact=launch_to_impact,
            assessmentTime=datetime.now(timezone.utc),
            confidence=confidence,
            intelligence_sources=intelligence_sources
        )
    
    @staticmethod
    def create_ss2_state_vector(
        object_id: str,
        position: List[float],
        velocity: List[float],
        epoch: datetime,
        source: str = "astroshield",
        covariance: Optional[List[List[float]]] = None,
        coordinate_frame: str = "ITRF",
        data_source: Optional[str] = None,
        quality_metric: Optional[float] = None,
        propagated_from: Optional[datetime] = None
    ) -> SDASS2StateVector:
        """Create SDA SS2 State Vector message"""
        return SDASS2StateVector(
            source=source,
            objectId=object_id,
            position=position,
            velocity=velocity,
            epoch=epoch,
            covariance=covariance,
            coordinateFrame=coordinate_frame,
            dataSource=data_source,
            qualityMetric=quality_metric,
            propagatedFrom=propagated_from
        )
    
    @staticmethod
    def create_ss2_elset(
        object_id: str,
        epoch: datetime,
        mean_motion: float,
        eccentricity: float,
        inclination: float,
        arg_of_perigee: float,
        raan: float,
        mean_anomaly: float,
        source: str = "astroshield",
        catalog_number: Optional[str] = None,
        classification: str = "U",
        intl_designator: Optional[str] = None,
        line1: Optional[str] = None,
        line2: Optional[str] = None,
        rcs_size: Optional[str] = None,
        object_type: Optional[str] = None,
        data_source: Optional[str] = None,
        quality_metric: Optional[float] = None
    ) -> SDASS2Elset:
        """Create SDA SS2 Elset message"""
        return SDASS2Elset(
            source=source,
            objectId=object_id,
            catalogNumber=catalog_number,
            classification=classification,
            intlDesignator=intl_designator,
            epoch=epoch,
            meanMotion=mean_motion,
            eccentricity=eccentricity,
            inclination=inclination,
            argOfPerigee=arg_of_perigee,
            raan=raan,
            meanAnomaly=mean_anomaly,
            line1=line1,
            line2=line2,
            rcsSize=rcs_size,
            objectType=object_type,
            dataSource=data_source,
            qualityMetric=quality_metric
        )
    
    @staticmethod
    def create_ss6_response_recommendation(
        threat_id: str,
        threat_type: str,
        threat_level: str,
        threatened_assets: List[str],
        primary_coa: str,
        priority: str,
        confidence: float,
        source: str = "astroshield",
        response_id: Optional[str] = None,
        alternate_coas: Optional[List[str]] = None,
        tactics_and_procedures: Optional[List[str]] = None,
        time_to_implement: Optional[float] = None,
        effective_window: Optional[List[datetime]] = None,
        rationale: Optional[str] = None,
        risk_assessment: Optional[str] = None,
        analyst_id: Optional[str] = None
    ) -> SDASS6ResponseRecommendation:
        """Create SDA SS6 Response Recommendation message"""
        return SDASS6ResponseRecommendation(
            source=source,
            threatId=threat_id,
            responseId=response_id or str(uuid.uuid4()),
            threatType=threat_type,
            threatLevel=threat_level,
            threatenedAssets=threatened_assets,
            primaryCOA=primary_coa,
            alternateCOAs=alternate_coas,
            tacticsAndProcedures=tactics_and_procedures,
            priority=priority,
            timeToImplement=time_to_implement,
            effectiveWindow=effective_window,
            confidence=confidence,
            rationale=rationale,
            riskAssessment=risk_assessment,
            recommendationTime=datetime.now(timezone.utc),
            analystId=analyst_id
        )
    
    @staticmethod
    def create_ss0_weather_data(
        data_type: str,
        timestamp: datetime,
        source: str = "EarthCast",
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        altitude: Optional[float] = None,
        resolution: Optional[float] = None,
        valid_time: Optional[datetime] = None,
        forecast_time: Optional[datetime] = None,
        value: Optional[float] = None,
        values: Optional[List[float]] = None,
        units: Optional[str] = None,
        quality_flag: Optional[str] = None,
        confidence: Optional[float] = None
    ) -> SDASS0WeatherData:
        """Create SDA SS0 Weather Data message"""
        return SDASS0WeatherData(
            source=source,
            dataType=data_type,
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            resolution=resolution,
            timestamp=timestamp,
            validTime=valid_time,
            forecastTime=forecast_time,
            value=value,
            values=values,
            units=units,
            qualityFlag=quality_flag,
            confidence=confidence
        )


def validate_sda_schema(schema_name: str, data: Dict[str, Any]) -> bool:
    """Validate data against SDA schema"""
    try:
        if schema_name == "maneuvers_detected":
            # Check required fields
            if 'source' not in data or 'satNo' not in data:
                return False
            
            # Validate data types
            if not isinstance(data['source'], str) or not isinstance(data['satNo'], str):
                return False
            
            # Validate optional numeric fields
            numeric_fields = [
                'prePosX', 'prePosY', 'prePosZ',
                'preVelX', 'preVelY', 'preVelZ',
                'postPosX', 'postPosY', 'postPosZ',
                'postVelX', 'postVelY', 'postVelZ'
            ]
            
            for field in numeric_fields:
                if field in data and data[field] is not None:
                    if not isinstance(data[field], (int, float)):
                        return False
            
            # Validate covariance matrices
            for cov_field in ['preCov', 'postCov']:
                if cov_field in data and data[cov_field] is not None:
                    if not isinstance(data[cov_field], list):
                        return False
                    for row in data[cov_field]:
                        if not isinstance(row, list):
                            return False
                        for val in row:
                            if not isinstance(val, (int, float)):
                                return False
            
            return True
            
        return False  # Unknown schema
        
    except Exception:
        return False 