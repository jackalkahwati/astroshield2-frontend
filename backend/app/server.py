from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum

app = FastAPI(title="AstroShield API v1.0.1")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

@app.get("/")
async def root():
    return {"message": "Welcome to AstroShield API v1.0.1"} 