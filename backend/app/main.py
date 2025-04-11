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
from fastapi.responses import HTMLResponse

# Import common logging utilities
from src.asttroshield.common.logging_utils import configure_logging, get_logger

# Configure logging using common utility
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
# )
configure_logging(level=logging.INFO)

# Get logger using common utility
# logger = logging.getLogger(__name__)
logger = get_logger(__name__)

app = FastAPI(
    title="AstroShield API",
    description="Backend API for the AstroShield satellite protection system",
    version="1.0.0",
    docs_url=None,  # Disable default Swagger UI
    redoc_url=None, # Disable default ReDoc UI
    openapi_url="/api/v1/openapi.json"
)

# Add our dynamic CORS middleware
app.middleware("http")(dynamic_cors_middleware)

# Import and include routers
from app.routers import health, analytics, maneuvers, satellites, advanced, dashboard, ccdm, trajectory, comparison, events

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

@app.on_event("startup")
async def startup_event():
    """Run database initialization on startup."""
    from app.db.init_db import init_db
    from app.db.session import SessionLocal

    # Initialize the database
    db = SessionLocal()
    try:
        init_db(db)
    finally:
        db.close()
    
    logger.info("Startup event completed")

if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "3002"))  # Use port 3002 for backend API
    log_level = os.environ.get("LOG_LEVEL", "info").lower()
    
    logger.info(f"Starting AstroShield API server at {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level=log_level)