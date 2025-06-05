from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
from fastapi.staticfiles import StaticFiles
from datetime import datetime, timedelta
import logging
import os
from pydantic import BaseModel
from app.middleware.cors import dynamic_cors_middleware
from app.core.security import (
    Token, authenticate_user, create_access_token, 
    ACCESS_TOKEN_EXPIRE_MINUTES, check_roles
)
from app.core.roles import Roles
from app.schemas.user import User as UserSchema
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.db.session import get_db, engine
from app.routers import ccdm, trajectory, maneuvers
from app.routers import login as login_router
from app.routers import local_ml_inference
from app.core.config import settings
from app.middlewares.rate_limiter import add_rate_limit_middleware
from app.middlewares.request_id import add_request_id_middleware
from app.core.errors import register_exception_handlers
from app.models.user import User as UserModel
from app.api.api_v1.api import api_router
from app.sda_integration.welders_arc_integration import router as sda_router

# Import error handling
from fastapi.exceptions import RequestValidationError
from fastapi import status, Request

# Add imports for graceful shutdown
import signal
import sys
import threading
import time

# Configure logging
logger = logging.getLogger(__name__)

# Configure CORS origins
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")
logger.info(f"Configuring CORS for origins: {CORS_ORIGINS}")

app = FastAPI(
    title="AstroShield API",
    description="AstroShield Space Situational Awareness & Satellite Protection System API",
    version="0.1.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
)

# Configure standard CORS middleware for simpler development
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add dynamic CORS middleware if available
try:
    app.middleware("http")(dynamic_cors_middleware)
except Exception as e:
    logger.warning(f"Could not add dynamic CORS middleware: {str(e)}")

# Register error handlers
register_exception_handlers(app)

# Add request ID middleware (must be first to track all requests)
add_request_id_middleware(app)

# Add rate limiting middleware
add_rate_limit_middleware(app)

# Include routers
app.include_router(ccdm.router)
app.include_router(trajectory.router, prefix="/api/v1", tags=["trajectory"])
app.include_router(maneuvers.router, prefix="/api/v1", tags=["maneuvers"])
app.include_router(login_router.router, prefix="/api/v1", tags=["auth"])
app.include_router(local_ml_inference.router, prefix="/api/v1/ml", tags=["ml", "local-inference"])
app.include_router(api_router, prefix=settings.API_V1_STR)
app.include_router(sda_router)

# Track active requests for graceful shutdown
active_requests = 0
active_requests_lock = threading.Lock()
app_shutdown_event = threading.Event()

@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Middleware to track active requests for graceful shutdown"""
    global active_requests
    
    # Increment active request counter
    with active_requests_lock:
        active_requests += 1
    
    try:
        # Check if server is shutting down
        if app_shutdown_event.is_set():
            return {
                "status": "error",
                "message": "Server is shutting down, please try again later"
            }
        
        # Process the request normally
        response = await call_next(request)
        return response
    finally:
        # Decrement active request counter
        with active_requests_lock:
            active_requests -= 1

# Database connection pool
db_pool = {}
db_pool_lock = threading.Lock()

def close_db_connections():
    """Close all database connections in the pool"""
    with db_pool_lock:
        for session_id, session in db_pool.items():
            try:
                logger.info(f"Closing database session {session_id}")
                session.close()
            except Exception as e:
                logger.error(f"Error closing database session {session_id}: {str(e)}")
        
        # Clear the pool
        db_pool.clear()

def graceful_shutdown(signum, frame):
    """
    Handle graceful shutdown of the application
    
    This ensures:
    1. No new requests are accepted
    2. Existing requests are allowed to complete
    3. Database connections are properly closed
    4. Any cleanup operations are performed
    """
    logger.info(f"Received signal {signum}, initiating graceful shutdown")
    
    # Set shutdown event to prevent new requests
    app_shutdown_event.set()
    
    # Wait for active requests to complete (max 30 seconds)
    max_wait = 30
    wait_step = 0.5
    total_waited = 0
    
    logger.info(f"Waiting for {active_requests} active requests to complete")
    
    while active_requests > 0 and total_waited < max_wait:
        time.sleep(wait_step)
        total_waited += wait_step
        if total_waited % 5 == 0:  # Log every 5 seconds
            logger.info(f"Still waiting for {active_requests} requests to complete ({total_waited}/{max_wait}s)")
    
    # Close database connections
    logger.info("Closing database connections")
    close_db_connections()
    
    # Log shutdown complete
    logger.info("Graceful shutdown complete")
    
    # Exit the application
    sys.exit(0)

# Register shutdown handlers
signal.signal(signal.SIGTERM, graceful_shutdown)
signal.signal(signal.SIGINT, graceful_shutdown)

@app.on_event("startup")
async def startup_event():
    """Handle application startup"""
    logger.info("Starting AstroShield API...")
    
    # Create database tables if they don't exist (for development/testing)
    # In production, you would typically use Alembic migrations.
    from app.db.base_class import Base
    from app.models import user  # noqa: F401 ensure model import for table creation
    logger.info("Creating database tables if they don't exist...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables checked/created.")
    
    # Check UDL configuration
    udl_username = os.environ.get("UDL_USERNAME")
    udl_password = os.environ.get("UDL_PASSWORD")
    udl_base_url = os.environ.get("UDL_BASE_URL")
    
    if not all([udl_username, udl_password, udl_base_url]):
        logger.warning("UDL credentials not fully configured. Using mock UDL service if available.")
    else:
        logger.info(f"UDL configured with base URL: {udl_base_url}")
        
    # Additional startup logic (unchanged)
    try:
        # Ensure database connections are ready
        # Initialize UDL service
        # Set up any other required services
        pass
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        # We'll continue running, but some services might be unavailable

@app.on_event("shutdown")
async def shutdown_event():
    """Handle application shutdown"""
    logger.info("AstroShield API shutting down")
    
    # Close any remaining database connections
    close_db_connections()

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/api/v1/health")
def api_health_check():
    """API health check endpoint"""
    return {
        "status": "healthy",
        "services": {
            "database": "connected",
            "api": "online"
        }
    }

@app.get("/api/v1/system-info")
def system_info():
    """System information endpoint"""
    return {
        "version": "0.1.0",
        "components": {
            "frontend": "online",
            "backend": "online",
            "database": "connected"
        }
    }

# Custom OpenAPI schema generator
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="AstroShield API",
        version="1.0.0",
        description="""
        # AstroShield API Documentation
        
        This API provides access to AstroShield's satellite protection system, including:
        
        - Satellite management and tracking
        - Maneuver planning and execution
        - Collision avoidance
        - Threat analysis and monitoring
        - CCDM (Concealment, Camouflage, Deception, and Maneuvering) capabilities
        
        ## Authentication
        
        Most endpoints require authentication using a JWT token. To obtain a token, use the `/api/v1/token` endpoint.
        
        ## Rate Limiting
        
        API requests are rate-limited to protect the system from abuse. Please respect the rate limits indicated in response headers:
        - `X-RateLimit-Limit`: Maximum requests per time window
        - `X-RateLimit-Remaining`: Remaining requests in current window
        - `X-RateLimit-Reset`: Seconds until window resets
        
        ## Versioning
        
        This API follows semantic versioning. The current version is v1.0.0.
        """,
        routes=app.routes,
    )
    
    # Add security schemes
    openapi_schema["components"] = openapi_schema.get("components", {})
    openapi_schema["components"]["securitySchemes"] = {
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        }
    }
    
    # Apply global security
    openapi_schema["security"] = [{"bearerAuth": []}]
    
    # Add contact and license information
    openapi_schema["info"] = {
        **openapi_schema.get("info", {}),
        "contact": {
            "name": "AstroShield Support",
            "url": "https://astroshield.com/support",
            "email": "support@astroshield.com"
        },
        "license": {
            "name": "Proprietary",
            "url": "https://astroshield.com/license"
        },
        "termsOfService": "https://astroshield.com/terms",
    }
    
    # Add server information
    openapi_schema["servers"] = [
        {
            "url": "/",
            "description": "Current environment"
        },
        {
            "url": "https://api.astroshield.com",
            "description": "Production environment"
        },
        {
            "url": "https://api-staging.astroshield.com",
            "description": "Staging environment"
        }
    ]
    
    # Attempt to merge with existing YAML file if it exists
    try:
        import yaml
        import os
        
        yaml_path = os.path.join(os.path.dirname(__file__), "openapi.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path, "r") as f:
                yaml_schema = yaml.safe_load(f)
                
            # Merge components section from YAML into auto-generated schema
            if "components" in yaml_schema and "schemas" in yaml_schema["components"]:
                openapi_schema["components"]["schemas"] = {
                    **openapi_schema["components"].get("schemas", {}),
                    **yaml_schema["components"]["schemas"]
                }
                
            # Log successful merge
            logger.info("Successfully merged existing OpenAPI YAML schema with auto-generated schema")
    except Exception as e:
        logger.warning(f"Error merging OpenAPI schemas: {str(e)}")
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Custom Swagger UI endpoint
@app.get("/api/v1/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui.css",
        swagger_favicon_url="https://fastapi.tiangolo.com/img/favicon.png",
        init_oauth={
            "clientId": "astroshield-docs",
            "usePkceWithAuthorizationCodeGrant": True,
        },
        swagger_ui_parameters={
            "docExpansion": "none",  # "list", "full", or "none"
            "deepLinking": True,
            "showExtensions": True,
            "showCommonExtensions": True,
            "defaultModelsExpandDepth": 3,
            "displayRequestDuration": True,
            "filter": True,
            "syntaxHighlight.theme": "monokai",
        }
    )

# Custom ReDoc endpoint
@app.get("/api/v1/redoc", include_in_schema=False)
async def redoc_html():
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - ReDoc",
        redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@2.0.0/bundles/redoc.standalone.js",
    )

# Documentation landing page
@app.get("/api/v1/documentation", include_in_schema=False)
async def documentation_landing_page():
    """
    Documentation landing page with links to Swagger UI and ReDoc
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AstroShield API Documentation</title>
        <meta charset="utf-8"/>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@1/css/pico.min.css">
        <style>
            :root {
                --primary: #3949ab;
                --primary-hover: #303f9f;
                --primary-focus: rgba(57, 73, 171, 0.125);
            }
            body {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                color: #333;
            }
            header {
                margin-bottom: 2rem;
                border-bottom: 1px solid #eee;
                padding-bottom: 1rem;
            }
            .container {
                padding: 0;
            }
            .card {
                margin-bottom: 2rem;
                padding: 1.5rem;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }
            .card:hover {
                transform: translateY(-5px);
                box-shadow: 0 6px 12px rgba(0,0,0,0.15);
            }
            .buttons {
                display: flex;
                gap: 1rem;
                margin-top: 1rem;
            }
            .tag {
                display: inline-block;
                background-color: #e0e0e0;
                color: #424242;
                padding: 0.2rem 0.5rem;
                border-radius: 4px;
                font-size: 0.8rem;
                margin-right: 0.5rem;
                margin-bottom: 0.5rem;
            }
            .tag.get { background-color: #e3f2fd; color: #1565c0; }
            .tag.post { background-color: #e8f5e9; color: #2e7d32; }
            .tag.put { background-color: #fff8e1; color: #f9a825; }
            .tag.delete { background-color: #ffebee; color: #c62828; }
            
            .endpoints {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                gap: 1rem;
                margin-top: 1rem;
            }
            .endpoint {
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                padding: 0.5rem;
                margin-bottom: 0.5rem;
            }
            .api-route {
                font-family: monospace;
                font-weight: bold;
            }
            code {
                background-color: #f5f5f5;
                padding: 0.2rem 0.4rem;
                border-radius: 4px;
                font-size: 0.9rem;
            }
            .version-badge {
                background-color: #3949ab;
                color: white;
                padding: 0.2rem 0.5rem;
                border-radius: 4px;
                font-size: 0.8rem;
                margin-left: 0.5rem;
            }
            .copy-curl {
                cursor: pointer;
                background-color: #f5f5f5;
                border: none;
                padding: 0.2rem 0.4rem;
                border-radius: 4px;
                font-size: 0.8rem;
                margin-left: 0.5rem;
            }
            .footer {
                margin-top: 3rem;
                border-top: 1px solid #eee;
                padding-top: 1rem;
                text-align: center;
                color: #757575;
            }
        </style>
    </head>
    <body>
        <header>
            <hgroup>
                <h1>AstroShield API <span class="version-badge">v1.0.0</span></h1>
                <h2>Interactive documentation for the AstroShield satellite protection system</h2>
            </hgroup>
        </header>
        
        <main class="container">
            <div class="grid">
                <div class="card">
                    <h3>Interactive API Documentation</h3>
                    <p>Choose your preferred API documentation format:</p>
                    <div class="buttons">
                        <a href="/api/v1/docs" role="button">Swagger UI</a>
                        <a href="/api/v1/redoc" role="button" class="secondary">ReDoc</a>
                        <a href="/api/v1/openapi.json" role="button" class="contrast">OpenAPI (JSON)</a>
                    </div>
                </div>
                
                <div class="card">
                    <h3>Getting Started</h3>
                    <p>To use the AstroShield API, you'll need to authenticate first:</p>
                    <pre><code>curl -X POST http://localhost:3001/api/v1/token \
-H "Content-Type: application/json" \
-d '{"username": "your_username", "password": "your_password"}'
</code></pre>
                    <p>Then use the returned token for authenticated requests:</p>
                    <pre><code>curl -X GET http://localhost:3001/api/v1/satellites \
-H "Authorization: Bearer your_token_here"
</code></pre>
                </div>
            </div>
            
            <h3>API Overview</h3>
            <p>The AstroShield API is organized into the following categories:</p>
            
            <div class="grid">
                <article>
                    <header><h4>Satellites</h4></header>
                    <p>Manage and monitor satellite information</p>
                    <div>
                        <span class="tag get">GET</span>
                        <span class="tag post">POST</span>
                        <span class="tag put">PUT</span>
                    </div>
                    <footer>
                        <a href="/api/v1/docs#/satellites" role="button" class="outline">View Endpoints</a>
                    </footer>
                </article>
                
                <article>
                    <header><h4>Maneuvers</h4></header>
                    <p>Plan and execute satellite maneuvers</p>
                    <div>
                        <span class="tag get">GET</span>
                        <span class="tag post">POST</span>
                        <span class="tag put">PUT</span>
                    </div>
                    <footer>
                        <a href="/api/v1/docs#/maneuvers" role="button" class="outline">View Endpoints</a>
                    </footer>
                </article>
                
                <article>
                    <header><h4>Analytics</h4></header>
                    <p>Access satellite analytics and reports</p>
                    <div>
                        <span class="tag get">GET</span>
                    </div>
                    <footer>
                        <a href="/api/v1/docs#/analytics" role="button" class="outline">View Endpoints</a>
                    </footer>
                </article>
                
                <article>
                    <header><h4>CCDM</h4></header>
                    <p>Concealment, Camouflage, Deception, and Maneuvering capabilities</p>
                    <div>
                        <span class="tag get">GET</span>
                        <span class="tag post">POST</span>
                    </div>
                    <footer>
                        <a href="/api/v1/docs#/ccdm" role="button" class="outline">View Endpoints</a>
                    </footer>
                </article>
            </div>
            
            <h3>Sample Requests</h3>
            <div class="grid">
                <div class="card">
                    <h4>Authentication</h4>
                    <p>Get an API token</p>
                    <span class="tag post">POST</span>
                    <code class="api-route">/api/v1/token</code>
                    <pre><code>curl -X POST http://localhost:3001/api/v1/token \
-H "Content-Type: application/json" \
-d '{"username": "your_username", "password": "your_password"}'</code></pre>
                </div>
                
                <div class="card">
                    <h4>List Satellites</h4>
                    <p>Get all satellites</p>
                    <span class="tag get">GET</span>
                    <code class="api-route">/api/v1/satellites</code>
                    <pre><code>curl -X GET http://localhost:3001/api/v1/satellites \
-H "Authorization: Bearer your_token_here"</code></pre>
                </div>
            </div>
            
            <h3>Additional Resources</h3>
            <ul>
                <li><a href="/backend/docs/swagger_guide.md">Swagger Documentation Guide</a></li>
                <li><a href="https://github.com/your-org/astroshield">GitHub Repository</a></li>
                <li><a href="mailto:support@astroshield.com">Contact Support</a></li>
            </ul>
            
            <div class="footer">
                <p>© 2024 AstroShield. All rights reserved.</p>
                <p>API Version 1.0.0</p>
            </div>
        </main>
        
        <script>
            // Add copy functionality for code blocks
            document.querySelectorAll('pre code').forEach((block) => {
                const copyButton = document.createElement('button');
                copyButton.textContent = 'Copy';
                copyButton.className = 'copy-curl';
                copyButton.addEventListener('click', () => {
                    navigator.clipboard.writeText(block.textContent);
                    copyButton.textContent = 'Copied!';
                    setTimeout(() => {
                        copyButton.textContent = 'Copy';
                    }, 2000);
                });
                block.parentNode.insertBefore(copyButton, block);
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint providing basic API information"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "api_name": "AstroShield API",
        "documentation": "/api/v1/documentation"
    }

@app.get("/api/v1/users/me", response_model=UserSchema, tags=["auth"])
async def read_users_me(current_user: UserModel = Depends(check_roles([Roles.viewer]))):
    """
    Get current user information
    """
    return current_user

@app.get("/api/v1/system-info", tags=["system"])
async def system_info(db: Session = Depends(get_db)):
    """
    Get system information including component statuses.
    """
    # Check database status
    db_status = "pending"
    try:
        db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception:
        db_status = "pending"
    
    return {
        "system": "AstroShield Platform",
        "version": "1.0.0",
        "components": {
            "frontend": "online",
            "backend": "online",
            "database": db_status,
            "authentication": "active"
        },
        "environment": os.getenv("ENVIRONMENT", "development"),
        "started_at": datetime.utcnow().isoformat(),
        "features": [
            "Collision avoidance",
            "Threat analysis",
            "Concealment detection",
            "Maneuver monitoring"
        ]
    }

@app.get("/api/v1/maneuvers", tags=["maneuvers"])
async def get_maneuvers_mock():
    """
    Mock endpoint for maneuvers that doesn't require auth
    """
    from datetime import datetime, timedelta
    import uuid
    
    return [
        {
            "id": "mnv-001",
            "satellite_id": "sat-001",
            "status": "completed",
            "type": "collision_avoidance",
            "start_time": datetime.utcnow() - timedelta(hours=2),
            "end_time": datetime.utcnow() - timedelta(hours=1, minutes=45),
            "resources": {
                "fuel_remaining": 85.5,
                "power_available": 90.0,
                "thruster_status": "nominal"
            },
            "parameters": {
                "delta_v": 0.02,
                "burn_duration": 15.0,
                "direction": {"x": 0.1, "y": 0.0, "z": -0.1},
                "target_orbit": {"altitude": 500.2, "inclination": 45.0, "eccentricity": 0.001}
            },
            "created_by": "user@example.com",
            "created_at": datetime.utcnow() - timedelta(days=1),
            "updated_at": datetime.utcnow() - timedelta(hours=1)
        },
        {
            "id": "mnv-002",
            "satellite_id": "sat-001",
            "status": "scheduled",
            "type": "station_keeping",
            "start_time": datetime.utcnow() + timedelta(hours=5),
            "end_time": None,
            "resources": {
                "fuel_remaining": 85.5,
                "power_available": 90.0,
                "thruster_status": "nominal"
            },
            "parameters": {
                "delta_v": 0.01,
                "burn_duration": 10.0,
                "direction": {"x": 0.0, "y": 0.0, "z": 0.1},
                "target_orbit": {"altitude": 500.0, "inclination": 45.0, "eccentricity": 0.001}
            },
            "created_by": "user@example.com",
            "created_at": datetime.utcnow() - timedelta(hours=3),
            "updated_at": None
        }
    ]

# Add a mock UDL endpoint for testing without UDL connections
@app.get("/api/v1/mock-udl-status", tags=["development"])
async def mock_udl_status():
    """
    Get the status of the mock UDL service
    """
    udl_base_url = os.environ.get("UDL_BASE_URL", "https://mock-udl-service.local/api/v1")
    mock_mode = udl_base_url.startswith(("http://localhost", "https://mock"))
    
    return {
        "mock_mode": mock_mode,
        "udl_url": udl_base_url,
        "status": "connected" if mock_mode else "using real UDL",
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "3002"))  # Use port 3002 for backend API
    log_level = os.environ.get("LOG_LEVEL", "info").lower()
    
    logger.info(f"Starting AstroShield API server at {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level=log_level)