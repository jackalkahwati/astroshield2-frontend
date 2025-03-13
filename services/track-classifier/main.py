from fastapi import FastAPI, HTTPException, Path
from fastapi.openapi.utils import get_openapi
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime
import uvicorn
import random

# Define the models based on the Swagger documentation

class EOO(BaseModel):
    """Electro-Optical Observation (EOO)."""
    OB_TIME: datetime = Field(..., description="Observation time")
    RA: float = Field(..., description="Right Ascension")
    DEC: float = Field(..., description="Declination")
    MAG: float = Field(..., description="Magnitude")
    SOLAR_PHASE_ANGLE: float = Field(..., description="Solar Phase Angle")
    SOURCE: str = Field(..., description="Source of the observation")

class EOOCOLLECTION(BaseModel):
    """Electro-Optical Observation Collection."""
    RECORDS: List[EOO] = Field(..., description="Collection of EOO records")

class ObjectTypeClassificationMessage(BaseModel):
    """Classification message for broad object type."""
    PAYLOAD: float = Field(..., description="Probability of being a payload")
    ROCKET_BODY: float = Field(..., description="Probability of being a rocket body")
    DEBRIS: float = Field(..., description="Probability of being debris")

class ObjectTypeClassificationMessageCollection(BaseModel):
    """Collection of classification messages for broad object type."""
    RECORDS: List[ObjectTypeClassificationMessage] = Field(..., description="Collection of classification records")

class ValidationError(BaseModel):
    """Validation error message."""
    loc: List[Union[str, int]] = Field(..., description="Location of the error")
    msg: str = Field(..., description="Error message")
    type: str = Field(..., description="Error type")

class HTTPValidationError(BaseModel):
    """HTTP validation error response."""
    detail: List[ValidationError] = Field(None, description="Error details")

# Create the FastAPI application
app = FastAPI(
    title="Track Classification API",
    description="This API provides a mechanism for classifying tracks to determine the type of object.",
    version="1.0.0",
    docs_url="/services/track-classifier/docs",
    redoc_url="/services/track-classifier/redoc",
    openapi_url="/services/track-classifier/openapi.json"
)

# Helper function to simulate classification
def classify_eo_observation(obs: EOO) -> ObjectTypeClassificationMessage:
    """
    Simulates classification of an Electro-Optical observation.
    In a real implementation, this would use a machine learning model.
    """
    # Simple rule-based classification for demonstration
    # Magnitude-based classification
    if obs.MAG < 4.0:
        # Brighter objects are more likely to be payloads
        payload_prob = 0.7
        rocket_prob = 0.2
        debris_prob = 0.1
    elif 4.0 <= obs.MAG < 7.0:
        # Medium brightness could be rocket bodies
        payload_prob = 0.3
        rocket_prob = 0.6
        debris_prob = 0.1
    else:
        # Dimmer objects more likely to be debris
        payload_prob = 0.1
        rocket_prob = 0.3
        debris_prob = 0.6
        
    # Add some randomness
    rand_factor = 0.1
    payload_prob += random.uniform(-rand_factor, rand_factor)
    rocket_prob += random.uniform(-rand_factor, rand_factor)
    debris_prob += random.uniform(-rand_factor, rand_factor)
    
    # Normalize to ensure probabilities sum to 1
    total = payload_prob + rocket_prob + debris_prob
    payload_prob /= total
    rocket_prob /= total
    debris_prob /= total
    
    return ObjectTypeClassificationMessage(
        PAYLOAD=payload_prob,
        ROCKET_BODY=rocket_prob,
        DEBRIS=debris_prob
    )

# Classify single EO track
@app.post("/classify-eo-track/object-type", 
         tags=["Electro-Optical: Object Type"],
         response_model=ObjectTypeClassificationMessage,
         responses={
             200: {"description": "Successful Response"},
             422: {"model": HTTPValidationError, "description": "Validation Error"}
         })
async def classify_eo_track_object_type(collection: EOOCOLLECTION):
    """
    Returns the object type classification.
    """
    if not collection.RECORDS:
        raise HTTPException(status_code=422, detail="No records provided for classification")
    
    # Use the first record for classification in the single track case
    result = classify_eo_observation(collection.RECORDS[0])
    return result

# Batch classify EO tracks
@app.post("/classify-eo-track/object-type/batch", 
         tags=["Electro-Optical: Object Type"],
         response_model=ObjectTypeClassificationMessageCollection,
         responses={
             200: {"description": "Successful Response"},
             422: {"model": HTTPValidationError, "description": "Validation Error"}
         })
async def classify_eo_track_object_type_batch(collections: List[EOOCOLLECTION]):
    """
    Returns the object type classification for multiple tracks.
    """
    if not collections:
        raise HTTPException(status_code=422, detail="No collections provided for classification")
    
    results = []
    
    for collection in collections:
        if not collection.RECORDS:
            # Skip empty collections
            continue
            
        # Use the first record of each collection for classification
        result = classify_eo_observation(collection.RECORDS[0])
        results.append(result)
    
    if not results:
        raise HTTPException(status_code=422, detail="No valid records found for classification")
    
    return ObjectTypeClassificationMessageCollection(RECORDS=results)

# Custom OpenAPI schema to ensure it matches the required specification
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Ensure it's labeled as OpenAPI 3.1.0
    openapi_schema["openapi"] = "3.1.0"
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 