from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mangum import Mangum
import logging
import os
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AstroShield API",
    description="Backend API for AstroShield satellite management system",
    version="1.0.1",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Configure CORS
origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "https://astroshield.vercel.app",
    "https://astroshield-staging.vercel.app",
    "https://asttroshield-v0-fxlubawwu-jackalkahwatis-projects.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import routes
from . import endpoints

# Add routes to app
app.include_router(endpoints.router, prefix="/api")

# Create handler for AWS Lambda
handler = Mangum(app)

# Startup Event
@app.on_event("startup")
async def startup_event():
    logger.info("Starting up AstroShield API...")
    # Add any startup tasks here (e.g., database connections)

# Shutdown Event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down AstroShield API...")
    # Add any cleanup tasks here

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to AstroShield API",
        "version": "1.0.1",
        "status": "operational",
        "docs": "/api/docs"
    }

# Global exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )

# Generic exception handler
@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500
        }
    ) 