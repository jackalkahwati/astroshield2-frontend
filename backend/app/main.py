from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from app.endpoints import router
from app.core.tasks import setup_periodic_tasks
from app.core.security import key_store
from app.api.api_v1.api import api_router
from datetime import datetime

# Initialize tracer
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Set up Jaeger exporter
jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=6831,
)
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)

app = FastAPI(
    title="AstroShield API",
    description="Backend API for the AstroShield satellite protection system",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://asttroshieldv0-dodjmxe8f-jackalkahwatis-projects.vercel.app",
        "http://localhost:3000",
        "https://astroshield2-api-production.up.railway.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenTelemetry instrumentation
FastAPIInstrumentor.instrument_app(app)

# Import and include routers
from app.routers import ccdm, analytics, maneuvers, health, comprehensive

app.include_router(ccdm.router, prefix="/api/v1/ccdm", tags=["CCDM"])
app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["Analytics"])
app.include_router(maneuvers.router, prefix="/api/v1/maneuvers", tags=["Maneuvers"])
app.include_router(health.router, prefix="/api/v1/health", tags=["Health"])
app.include_router(comprehensive.router, prefix="/api/v1/comprehensive", tags=["Comprehensive"])

# Include router
app.include_router(router, prefix="/api")

# Include API router
app.include_router(api_router, prefix="/api/v1")

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    # Ensure we have valid encryption keys
    if not key_store.get_current_key():
        key_store.rotate_keys()
    
    # Setup periodic tasks
    setup_periodic_tasks(app)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if hasattr(app.state, "scheduler"):
        app.state.scheduler.shutdown()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "current_key_version": key_store.current_version
    }

@app.get("/")
async def root():
    return {"message": "AstroShield API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3001) 