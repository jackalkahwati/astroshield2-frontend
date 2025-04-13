from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from app.models.ccdm import *
from app.services.ccdm import CCDMService

# Manually mock the ccdm_bp if we can't import it
try:
    from api.ccdm_endpoints import ccdm_bp
except ImportError:
    # Create a minimal mock of what we need
    from fastapi import APIRouter
    ccdm_bp = APIRouter(prefix="/api/v1/ccdm")
    
    @ccdm_bp.get("/status")
    async def status():
        return {"status": "ok", "version": "1.0.0"}

app = FastAPI(title="AstroShield Backend API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Register API blueprints
app.include_router(ccdm_bp)

@app.get("/")
async def root():
    return {"message": "Welcome to AstroShield API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"} 