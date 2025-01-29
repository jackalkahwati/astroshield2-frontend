from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import os
from app.middleware.cors import dynamic_cors_middleware

app = FastAPI(
    title="AstroShield API",
    description="Backend API for the AstroShield satellite protection system",
    version="1.0.0"
)

# Add our dynamic CORS middleware
app.middleware("http")(dynamic_cors_middleware)

# Import and include routers
from app.routers import health, analytics, maneuvers, satellites, advanced

# Include routers with prefixes
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(analytics.router, prefix="/api/v1", tags=["analytics"])
app.include_router(maneuvers.router, prefix="/api/v1", tags=["maneuvers"])
app.include_router(satellites.router, prefix="/api/v1", tags=["satellites"])
app.include_router(advanced.router, prefix="/api/v1/advanced", tags=["advanced"])

# Root endpoint
@app.get("/")
async def root():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3001) 