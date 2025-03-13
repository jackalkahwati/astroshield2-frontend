from fastapi import FastAPI, HTTPException, Path
from fastapi.openapi.utils import get_openapi
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import uvicorn

# Define the OMM model based on the schema
class OMM(BaseModel):
    SEMI_MAJOR_AXIS: float = Field(..., description="Semi-major axis value")
    ECCENTRICITY: float = Field(..., description="Eccentricity value")
    INCLINATION: float = Field(..., description="Inclination value")

# Create the FastAPI application
app = FastAPI(
    title="AstroShield FastAPI Microservice",
    description="Example microservice implemented with FastAPI",
    version="1.0.0",
    docs_url="/services/fastapi-example/docs",
    redoc_url="/services/fastapi-example/redoc",
    openapi_url="/services/fastapi-example/openapi.json"
)

# In-memory storage for OMMs
omm_storage: List[OMM] = []

# Hello World endpoint
@app.get("/hello", tags=["Hello"])
async def hello_world():
    """
    Hello World endpoint - returns a simple greeting.
    """
    return {"message": "Hello World"}

# Post a new OMM
@app.post("/omm/new", tags=["Data Posting"], response_model=OMM, status_code=201)
async def post_omm(omm: OMM):
    """
    Post a new Orbit Mean-Elements Message (OMM).
    """
    omm_storage.append(omm)
    return omm

# Get all OMMs
@app.get("/omm/", tags=["Data Retrieval"], response_model=List[OMM])
async def get_all_omms():
    """
    Get all stored Orbit Mean-Elements Messages (OMMs).
    """
    return omm_storage

# Get the latest OMM
@app.get("/omm/latest", tags=["Data Retrieval"], response_model=OMM)
async def get_latest_omm():
    """
    Get the latest Orbit Mean-Elements Message (OMM).
    """
    if not omm_storage:
        raise HTTPException(status_code=404, detail="No OMMs available")
    return omm_storage[-1]

# Get an OMM by index
@app.get("/omm/{index}", tags=["Data Retrieval"], response_model=OMM)
async def get_omm_by_index(
    index: int = Path(..., description="Index of the OMM to retrieve", ge=0)
):
    """
    Get an Orbit Mean-Elements Message (OMM) by its index.
    """
    if index >= len(omm_storage):
        raise HTTPException(status_code=404, detail=f"OMM at index {index} not found")
    return omm_storage[index]

# Delete all OMMs
@app.delete("/omm/", tags=["Data Deletion"], status_code=204)
async def delete_all_omms():
    """
    Delete all stored Orbit Mean-Elements Messages (OMMs).
    """
    omm_storage.clear()
    return None

# Delete the latest OMM
@app.delete("/omm/latest", tags=["Data Deletion"], status_code=204)
async def delete_latest_omm():
    """
    Delete the latest Orbit Mean-Elements Message (OMM).
    """
    if not omm_storage:
        raise HTTPException(status_code=404, detail="No OMMs available to delete")
    omm_storage.pop()
    return None

# Delete an OMM by index
@app.delete("/omm/{index}", tags=["Data Deletion"], status_code=204)
async def delete_omm_by_index(
    index: int = Path(..., description="Index of the OMM to delete", ge=0)
):
    """
    Delete an Orbit Mean-Elements Message (OMM) by its index.
    """
    if index >= len(omm_storage):
        raise HTTPException(status_code=404, detail=f"OMM at index {index} not found")
    omm_storage.pop(index)
    return None

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