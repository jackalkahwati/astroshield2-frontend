from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from models.database import Base
from api.endpoints import router
import os
import logging
from api.ccdm_endpoints import ccdm_bp
from api.rpo_shape_endpoints import rpo_shape_bp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./astrashield.db")

engine = create_async_engine(DATABASE_URL)
async_session = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

def create_app():
    app = FastAPI(
        title="AstroShield API",
        description="API for AstroShield space protection system",
        version="1.0.0"
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Dependency to get DB session
    async def get_db():
        async with async_session() as session:
            yield session
    
    # Include API routes
    app.include_router(router, prefix="/api")
    
    # Include CCDM routes
    from flask import Flask
    flask_app = Flask(__name__)
    flask_app.register_blueprint(ccdm_bp, url_prefix='/api/ccdm')
    flask_app.register_blueprint(rpo_shape_bp, url_prefix='/api/rpo-shape')
    
    # Create database tables
    @app.on_event("startup")
    async def startup():
        logger.info("Starting AstroShield API")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        # Initialize services
        logger.info("Initializing services")
        try:
            # Initialize UDL service
            from services.udl_service import UDLService
            udl_base_url = os.getenv("UDL_BASE_URL", "https://udl.example.com/api")
            udl_api_key = os.getenv("UDL_API_KEY", "dummy_key")
            app.state.udl_service = UDLService(udl_base_url, udl_api_key)
            await app.state.udl_service.initialize()
            logger.info("UDL service initialized")
            
            # Initialize Cross-Tag Correlation service
            from services.cross_tag_correlation_service import CrossTagCorrelationService
            app.state.cross_tag_service = CrossTagCorrelationService()
            logger.info("Cross-Tag Correlation service initialized")
            
            # Initialize TMDB Comparison service
            from services.tmdb_comparison_service import TMDBComparisonService
            tmdb_base_url = os.getenv("TMDB_BASE_URL", "https://tmdb.example.com/api")
            tmdb_api_key = os.getenv("TMDB_API_KEY", "dummy_key")
            app.state.tmdb_service = TMDBComparisonService(tmdb_base_url, tmdb_api_key)
            logger.info("TMDB Comparison service initialized")
            
            # RPO Shape Analysis service is already initialized in the endpoints
            logger.info("All services initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing services: {str(e)}")
    
    @app.on_event("shutdown")
    async def shutdown():
        logger.info("Shutting down AstroShield API")
        # Close UDL service connection
        if hasattr(app.state, "udl_service"):
            await app.state.udl_service.close()
    
    return app

app = create_app()

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5003)
