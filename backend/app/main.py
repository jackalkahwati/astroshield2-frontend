from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import os

app = FastAPI(
    title="AstroShield API",
    description="Backend API for the AstroShield satellite protection system",
    version="1.0.0"
)

# Configure tracing only if JAEGER_ENABLED is set
if os.getenv("JAEGER_ENABLED", "false").lower() == "true":
    from opentelemetry import trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    # Initialize tracer
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)

    # Set up Jaeger exporter
    jaeger_exporter = JaegerExporter(
        agent_host_name=os.getenv("JAEGER_HOST", "localhost"),
        agent_port=int(os.getenv("JAEGER_PORT", "6831")),
    )
    trace.get_tracer_provider().add_span_processor(
        BatchSpanProcessor(jaeger_exporter)
    )
    
    # Initialize OpenTelemetry instrumentation
    FastAPIInstrumentor.instrument_app(app)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://astroshield2.vercel.app",
        "http://localhost:3000",
        "https://astroshield2-api-production.up.railway.app",
        "https://nosy-boy-production.up.railway.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include routers
from app.routers import ccdm, analytics, maneuvers, health, comprehensive

# Include routers with prefixes
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(maneuvers.router, prefix="/api/v1", tags=["maneuvers"])
app.include_router(analytics.router, prefix="/api/v1", tags=["analytics"])
app.include_router(comprehensive.router, prefix="/api/v1", tags=["comprehensive"])
app.include_router(ccdm.router, prefix="/api/v1", tags=["ccdm"])

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