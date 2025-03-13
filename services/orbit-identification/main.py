from fastapi import FastAPI, HTTPException
from fastapi.openapi.utils import get_openapi
from typing import List, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field, model_validator
import uvicorn
import math

# Constants
EARTH_RADIUS = 6378.137  # Earth radius in kilometers
EARTH_MU = 398600.4418  # Earth's gravitational parameter in km^3/s^2
SIDEREAL_DAY = 86164.0905  # Sidereal day in seconds

# Define the MPE (Minimum Propagatable Element Set) Class
class MPE(BaseModel):
    ENTITY_ID: Optional[str] = Field(None, description="Unique identifier for the space object")
    EPOCH: Optional[str] = Field(None, description="Epoch time for the orbital elements")
    SEMI_MAJOR_AXIS: Optional[float] = Field(None, description="Semi-major axis in kilometers")
    MEAN_MOTION: Optional[float] = Field(None, description="Mean motion in revolutions per day")
    ECCENTRICITY: float = Field(..., description="Orbital eccentricity (unitless)")
    INCLINATION: float = Field(..., description="Orbital inclination in degrees")
    RA_OF_ASC_NODE: Optional[float] = Field(None, description="Right ascension of ascending node in degrees")
    ARG_OF_PERICENTER: Optional[float] = Field(None, description="Argument of pericenter in degrees")
    MEAN_ANOMALY: Optional[float] = Field(None, description="Mean anomaly in degrees")
    BSTAR: Optional[float] = Field(None, description="Drag term (1/Earth radii)")

    @model_validator(mode='after')
    def validate_orbital_elements(self):
        """Validate that either semi-major axis or mean motion is provided"""
        if self.SEMI_MAJOR_AXIS is None and self.MEAN_MOTION is None:
            raise ValueError("Either SEMI_MAJOR_AXIS or MEAN_MOTION must be provided")
        return self

# Define Reference Frame and Time System literals
ReferenceFrame = Literal["ICRF", "GCRF", "ITRF", "EME2000", "TEME"]
TimeSystem = Literal["UTC", "TAI", "TT", "UT1", "GPS", "TDB"]
MeanElementTheory = Literal["SGP4", "J2", "J4", "TWO_BODY", "HIGH_PRECISION"]

# Define the MPECOLLECTION Class
class MPECOLLECTION(BaseModel):
    REF_FRAME: Optional[ReferenceFrame] = Field("GCRF", description="Reference frame for the orbital elements")
    TIME_SYSTEM: Optional[TimeSystem] = Field("UTC", description="Time system for the epoch")
    MEAN_ELEMENT_THEORY: Optional[MeanElementTheory] = Field("SGP4", description="Mean element theory used")
    RECORDS: List[MPE] = Field(..., description="Collection of orbital element records")

# Define Orbit Tag literals
OrbitTagEnum = Literal[
    # Altitude-based classes
    "LEO", "VLEO", "MEO", "GSO", "NEAR_GSO", "GEO", "NEAR_GEO", "HIGH_EARTH_ORBIT", "GEO_GRAVEYARD",
    # Direction classes
    "PROGRADE", "RETROGRADE",
    # Eccentricity classes
    "CIRCULAR", "NEAR_CIRCULAR", "ELLIPTIC", "PARABOLIC", "HYPERBOLIC", "HIGHLY_ELLIPTICAL_ORBIT",
    # IADC Protected Regions
    "IADC_LEO_PROTECTED_REGION", "IADC_LEO_PROTECTED_REGION_CROSSING", 
    "IADC_GEO_PROTECTED_REGION", "IADC_GEO_PROTECTED_REGION_CROSSING",
    # Inclination classes
    "EQUATORIAL", "NEAR_EQUATORIAL", "POLAR",
    # Orbit planes
    "ECLIPTIC", "NEAR_ECLIPTIC", "SSO"
]

# Define response models
class OrbitTagMessage(BaseModel):
    TAGS: List[OrbitTagEnum] = Field(..., description="List of orbit family tags")

class OrbitTagMessageCollection(BaseModel):
    RECORDS: List[OrbitTagMessage] = Field(..., description="Collection of orbit tag records")

# Define validation error models
class ValidationError(BaseModel):
    loc: List[Union[str, int]] = Field(..., description="Location of the error")
    msg: str = Field(..., description="Error message")
    type: str = Field(..., description="Error type")

class HTTPValidationError(BaseModel):
    detail: List[ValidationError] = Field(..., description="Validation error details")

# Create FastAPI application
app = FastAPI(
    title="Orbit Family Identification API",
    description="API for identifying orbit families based on orbital parameters",
    version="1.0.0",
    docs_url="/services/orbit-identification/docs",
    redoc_url="/services/orbit-identification/redoc",
    openapi_url="/services/orbit-identification/openapi.json",
)

# Helper Functions
def calculate_perigee_apogee(semi_major_axis: float, eccentricity: float) -> tuple:
    """Calculate perigee and apogee altitudes in kilometers"""
    perigee = (semi_major_axis * (1 - eccentricity)) - EARTH_RADIUS
    apogee = (semi_major_axis * (1 + eccentricity)) - EARTH_RADIUS
    return perigee, apogee

def calculate_period(semi_major_axis: float) -> float:
    """Calculate orbital period in seconds"""
    return 2 * math.pi * math.sqrt(semi_major_axis**3 / EARTH_MU)

def mean_motion_to_sma(mean_motion: float) -> float:
    """Convert mean motion (rev/day) to semi-major axis (km)"""
    # Calculate period in seconds from mean motion
    period = 86400 / mean_motion  # 86400 seconds in a day
    # Calculate semi-major axis using Kepler's Third Law
    return (EARTH_MU * (period / (2 * math.pi))**2) ** (1/3)

def identify_orbit_tags(orbital_elements: MPE) -> List[OrbitTagEnum]:
    """Identify orbit family tags based on orbital elements"""
    tags = []
    
    # Get semi-major axis either directly or from mean motion
    semi_major_axis = orbital_elements.SEMI_MAJOR_AXIS
    if semi_major_axis is None and orbital_elements.MEAN_MOTION is not None:
        semi_major_axis = mean_motion_to_sma(orbital_elements.MEAN_MOTION)
    
    eccentricity = orbital_elements.ECCENTRICITY
    inclination = orbital_elements.INCLINATION
    
    # Calculate perigee and apogee
    perigee, apogee = calculate_perigee_apogee(semi_major_axis, eccentricity)
    
    # Calculate orbital period
    period = calculate_period(semi_major_axis)
    
    # Direction-based classifications
    if inclination < 90:
        tags.append("PROGRADE")
    else:
        tags.append("RETROGRADE")
    
    # Eccentricity-based classifications
    if eccentricity <= 0.02:
        tags.append("CIRCULAR")
    if eccentricity <= 0.1:
        tags.append("NEAR_CIRCULAR")
    if eccentricity < 1.0:
        tags.append("ELLIPTIC")
    elif abs(eccentricity - 1.0) < 0.01:
        tags.append("PARABOLIC")
    elif eccentricity > 1.0:
        tags.append("HYPERBOLIC")
    
    # Check for Highly Elliptical Orbit (HEO)
    if perigee < 2000 and apogee > 35000:
        tags.append("HIGHLY_ELLIPTICAL_ORBIT")
    
    # Altitude-based classifications
    if perigee < 2000 and perigee >= 80:
        tags.append("LEO")
        if perigee < 450:
            tags.append("VLEO")
    elif perigee >= 2000 and apogee < 35786:
        tags.append("MEO")
    elif apogee > 35786:
        tags.append("HIGH_EARTH_ORBIT")
    
    # Geosynchronous and Geostationary checks
    sidereal_period = SIDEREAL_DAY  # Sidereal day in seconds
    if abs(period - sidereal_period) < 300:  # Within 5 minutes of sidereal day
        tags.append("GSO")
        if abs(inclination) < 0.1 and eccentricity < 0.01:
            tags.append("GEO")
    
    # Near-GSO
    if abs(period - sidereal_period) < 3600:  # Within 1 hour of sidereal day
        tags.append("NEAR_GSO")
        if abs(inclination) < 0.5 and eccentricity < 0.02:
            tags.append("NEAR_GEO")
    
    # GEO Graveyard
    if semi_major_axis > 42164 and semi_major_axis < 45000:
        perigee_height = perigee + EARTH_RADIUS
        if perigee_height > 42464:  # 300 km above GEO
            tags.append("GEO_GRAVEYARD")
    
    # Inclination-based classifications
    if abs(inclination) < 0.1 or abs(inclination - 180) < 0.1:
        tags.append("EQUATORIAL")
    elif abs(inclination) < 5 or abs(inclination - 180) < 5:
        tags.append("NEAR_EQUATORIAL")
    elif inclination > 60 and inclination < 120:
        tags.append("POLAR")
    
    # IADC Protected Regions
    # LEO Protected Region: Altitudes up to 2000 km
    if perigee < 2000 and perigee >= 80:
        tags.append("IADC_LEO_PROTECTED_REGION")
        tags.append("IADC_LEO_PROTECTED_REGION_CROSSING")
    elif perigee > 2000 and apogee < 2000:
        tags.append("IADC_LEO_PROTECTED_REGION_CROSSING")
    
    # GEO Protected Region: GEO altitude +/- 200 km, inclination +/- 15 degrees
    geo_altitude = 35786
    if abs(apogee - geo_altitude) <= 200 and abs(perigee - geo_altitude) <= 200 and abs(inclination) <= 15:
        tags.append("IADC_GEO_PROTECTED_REGION")
        tags.append("IADC_GEO_PROTECTED_REGION_CROSSING")
    elif (apogee > geo_altitude + 200 and perigee < geo_altitude - 200) and abs(inclination) <= 15:
        tags.append("IADC_GEO_PROTECTED_REGION_CROSSING")
    
    # Special orbits
    # Sun-Synchronous Orbit (SSO)
    # Simplified check (in reality, it depends on the exact orbital precession rate)
    if inclination > 96 and inclination < 100 and semi_major_axis < 8000:
        tags.append("SSO")
    
    # Ecliptic orbit (simplified check)
    if abs(inclination - 23.4) < 0.1:
        tags.append("ECLIPTIC")
    elif abs(inclination - 23.4) < 5:
        tags.append("NEAR_ECLIPTIC")
    
    return tags

@app.post(
    "/identify",
    response_model=OrbitTagMessage,
    responses={422: {"model": HTTPValidationError}},
    summary="Identify orbit families for a single set of orbital elements",
    description="Returns a list of orbit family tags applicable to the provided orbital elements",
)
async def identify_orbit(orbital_elements: MPE) -> OrbitTagMessage:
    """
    Identify orbit families for a single set of orbital elements.
    
    - Required: Either SEMI_MAJOR_AXIS or MEAN_MOTION
    - Required: ECCENTRICITY
    - Required: INCLINATION
    """
    # Validate required fields
    if orbital_elements.SEMI_MAJOR_AXIS is None and orbital_elements.MEAN_MOTION is None:
        raise HTTPException(
            status_code=422,
            detail="Either SEMI_MAJOR_AXIS or MEAN_MOTION must be provided",
        )
    
    # Identify orbit tags
    tags = identify_orbit_tags(orbital_elements)
    
    return OrbitTagMessage(TAGS=tags)

@app.post(
    "/identify-batch",
    response_model=OrbitTagMessageCollection,
    responses={422: {"model": HTTPValidationError}},
    summary="Identify orbit families for multiple sets of orbital elements",
    description="Returns a collection of orbit family tag lists for each set of provided orbital elements",
)
async def identify_orbit_batch(orbital_elements_collection: MPECOLLECTION) -> OrbitTagMessageCollection:
    """
    Identify orbit families for multiple sets of orbital elements.
    
    - Required: RECORDS (collection of MPE objects)
    - For each record:
      - Required: Either SEMI_MAJOR_AXIS or MEAN_MOTION
      - Required: ECCENTRICITY
      - Required: INCLINATION
    """
    results = []
    
    for orbital_elements in orbital_elements_collection.RECORDS:
        # Validate required fields
        if orbital_elements.SEMI_MAJOR_AXIS is None and orbital_elements.MEAN_MOTION is None:
            raise HTTPException(
                status_code=422,
                detail="Either SEMI_MAJOR_AXIS or MEAN_MOTION must be provided for each record",
            )
        
        # Identify orbit tags
        tags = identify_orbit_tags(orbital_elements)
        results.append(OrbitTagMessage(TAGS=tags))
    
    return OrbitTagMessageCollection(RECORDS=results)

# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Add OpenAPI 3.1 compliance
    openapi_schema["openapi"] = "3.1.0"
    
    # Add additional information
    openapi_schema["info"]["contact"] = {
        "name": "Space Domain Awareness Team",
        "email": "sda@example.com",
        "url": "https://example.com",
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 