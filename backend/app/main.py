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
from app.models.user import User
from fastapi.responses import HTMLResponse, JSONResponse
import time
import psutil

# Import error handling
from app.core.error_handling import register_exception_handlers

# Import security middleware
from app.middleware.security import add_security_middleware

# Import compression middleware
from app.middleware.compression import add_compression_middleware

# Import common logging utilities
try:
    from src.asttroshield.common.logging_utils import configure_logging, get_logger
    # Configure logging using common utility
    configure_logging(level=logging.INFO)
    # Get logger using common utility
    logger = get_logger(__name__)
except ImportError:
    # Fallback if the imports are not available
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.warning("Using fallback logging configuration")

# Configure CORS origins
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")
logger.info(f"Configuring CORS for origins: {CORS_ORIGINS}")

app = FastAPI(
    title="AstroShield API",
    description="Backend API for the AstroShield satellite protection system",
    version="1.0.0",
    docs_url=None,  # Disable default Swagger UI
    redoc_url=None, # Disable default ReDoc UI
    openapi_url="/api/v1/openapi.json"
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

# Add security headers middleware
add_security_middleware(app)

# Add compression middleware for large responses
add_compression_middleware(app)

# Register error handlers
register_exception_handlers(app)

# Import and include routers
try:
    from app.routers import health, analytics, maneuvers, satellites, advanced, dashboard, ccdm, trajectory, comparison, events, diagnostics

    # Include routers with prefixes
    app.include_router(health.router, prefix="/api/v1", tags=["health"])
    app.include_router(analytics.router, prefix="/api/v1", tags=["analytics"])
    app.include_router(maneuvers.router, prefix="/api/v1", tags=["maneuvers"])
    app.include_router(satellites.router, prefix="/api/v1", tags=["satellites"])
    app.include_router(advanced.router, prefix="/api/v1/advanced", tags=["advanced"])
    app.include_router(dashboard.router, prefix="/api/v1", tags=["dashboard"])
    app.include_router(ccdm.router, prefix="/api/v1/ccdm", tags=["ccdm"])
    app.include_router(trajectory.router, prefix="/api", tags=["trajectory"])
    app.include_router(comparison.router, prefix="/api", tags=["comparison"])
    app.include_router(events.router, prefix="/api/v1", tags=["events"])
    app.include_router(diagnostics.router, prefix="/api/v1", tags=["diagnostics"])
except ImportError as e:
    logger.warning(f"Could not import all routers: {str(e)}")
    logger.info("Some endpoints may not be available")

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

# Authentication endpoints
class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/api/v1/token", response_model=Token, tags=["auth"])
async def login_for_access_token(login_data: LoginRequest):
    """
    Get an access token for future requests
    """
    user = await authenticate_user(login_data.username, login_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    expires_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    expires_at = datetime.utcnow() + expires_delta
    
    access_token = create_access_token(
        data={"sub": user.email, "is_superuser": user.is_superuser},
        expires_delta=expires_delta
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_at=expires_at
    )

@app.get("/api/v1/users/me", response_model=User, tags=["auth"])
async def read_users_me(current_user: User = Depends(check_roles(["active"]))):
    """
    Get current user information
    """
    return current_user

@app.get("/api/v1/system-info", tags=["system"])
async def system_info():
    """
    Get system information
    """
    return {
        "version": "1.0.0",
        "environment": os.environ.get("ENVIRONMENT", "development"),
        "python_version": os.environ.get("PYTHON_VERSION", "3.9+"),
        "api_uptime": "Unknown",  # In a real implementation, you would track this
        "build_timestamp": datetime.utcnow().isoformat(),
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

# Add this health check endpoint
@app.get("/health", tags=["Health"], summary="Check service health")
async def health_check():
    """
    Health check endpoint that returns the status of the API and its dependencies.
    Used by monitoring services and container orchestration.
    """
    start_time = time.time()
    
    # Basic system metrics
    health_info = {
        "status": "healthy",
        "timestamp": time.time(),
        "uptime": time.time() - app.state.start_time if hasattr(app.state, "start_time") else 0,
        "version": os.environ.get("APP_VERSION", "1.0.0"),
        "environment": os.environ.get("ENVIRONMENT", "development"),
        "system": {
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent
        }
    }
    
    # Add database check if needed
    try:
        # Simple database check - replace with actual DB health check
        # db_result = await check_database_connection()
        health_info["database"] = {"status": "connected"}
    except Exception as e:
        health_info["database"] = {"status": "error", "message": str(e)}
        health_info["status"] = "degraded"
    
    # Add response time
    health_info["response_time_ms"] = (time.time() - start_time) * 1000
    
    return health_info

# Add this to the startup event to track application uptime
@app.on_event("startup")
async def startup_event():
    app.state.start_time = time.time()
    logger.info("Starting AstroShield API...")
    
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

# ------------------------------------------------------------
# Trajectory comparison endpoint (toy implementation)
# ------------------------------------------------------------

from typing import List, Dict
import math, json, os, pathlib, datetime as _dt

# Load TIP window data once at startup
TIP_PATHS = [
    pathlib.Path("data/tip_window1.json"),
    pathlib.Path("tip_window1.json"),
]
_TIP_DATA: List[Dict] = []
for p in TIP_PATHS:
    if p.exists():
        try:
            _TIP_DATA = json.loads(p.read_text())
            break
        except Exception:
            pass

def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2*R*math.asin(math.sqrt(a))


@app.get("/api/v1/trajectory/{norad_id}", tags=["trajectory"])
async def trajectory_compare(norad_id: str):
    """Return mock trajectories & error metrics for a NORAD object."""
    rec = next((r for r in _TIP_DATA if r["NORAD_CAT_ID"] == norad_id), None)
    if not rec:
        raise HTTPException(status_code=404, detail="NORAD ID not found in TIP file")

    impact_lat = float(rec["LAT"])
    impact_lon = float(rec["LON"])
    impact_t = _dt.datetime.strptime(rec["DECAY_EPOCH"], "%Y-%m-%d %H:%M:%S")

    def make_path(dlon, dlat, scale=1.0):
        traj = []
        for i in range(180, -1, -1):  # 180 → 0 minutes before impact
            frac = i/180
            t = impact_t - _dt.timedelta(minutes=i)
            lon = impact_lon + dlon * (1-frac) * 20 * scale
            lat = impact_lat + dlat * (1-frac) * 20 * scale
            alt = 120000*frac  # 120 km down to 0
            traj.append({
                "time": t.replace(tzinfo=_dt.timezone.utc).isoformat(),
                "position": [lon, lat, alt],
                "velocity": [0, 0, -200 + 150*frac],
            })
        return traj

    truth_traj = make_path(0.1, 0.08, 1.0)
    phys_traj  = make_path(0.11, 0.07, 1.02)
    ml_traj    = make_path(0.09, 0.09, 0.98)

    def metrics(pred, true):
        dR = [_haversine_km(p["position"][1], p["position"][0], t["position"][1], t["position"][0]) for p, t in zip(pred, true)]
        dT = [( _dt.datetime.fromisoformat(p["time"]) - _dt.datetime.fromisoformat(t["time"]) ).total_seconds() for p,t in zip(pred,true)]
        return {"dR": dR, "dT": dT}

    return {
        "models": [
            {"name": "Ground Truth", "color": "#4CAF50", "trajectory": truth_traj},
            {"name": "Physics Propagation", "color": "#1E90FF", "trajectory": phys_traj},
            {"name": "ML Forecast", "color": "#FF5722", "trajectory": ml_traj},
        ],
        "metrics": {
            "physics": metrics(phys_traj, truth_traj),
            "ml": metrics(ml_traj, truth_traj),
        }
    }

if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "3002"))  # Use port 3002 for backend API
    log_level = os.environ.get("LOG_LEVEL", "info").lower()
    
    logger.info(f"Starting AstroShield API server at {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level=log_level)