"""
Main application entry point for AstroShield
"""
import os
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from app.common.logging import logger
from app.trajectory.router import router as trajectory_router
from app.maneuvers.router import router as maneuvers_router
from app.ccdm.router import router as ccdm_router

# Create FastAPI app
app = FastAPI(title="AstroShield API", version="0.1.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(trajectory_router)
app.include_router(maneuvers_router)
app.include_router(ccdm_router)

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "0.1.0"}

@app.get("/api/v1/health")
def api_health_check():
    """API health check endpoint"""
    return {
        "status": "healthy",
        "version": "0.1.0",
        "services": {
            "trajectory": "ready",
            "maneuvers": "ready",
            "ccdm": "ready"
        }
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to the AstroShield API",
        "documentation": "/docs",
        "health": "/health"
    }

def main():
    """Run the application"""
    port = int(os.environ.get("PORT", 5002))
    logger.info(f"Starting AstroShield backend on port {port}")
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)

if __name__ == "__main__":
    main() 