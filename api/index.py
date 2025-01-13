from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from endpoints import router

app = FastAPI(title="AstroShield API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the router without prefix since Vercel will handle the /api prefix
app.include_router(router)

# Create handler for AWS Lambda with base path
handler = Mangum(app, api_gateway_base_path="/api")

@app.get("/")
async def root():
    return {"message": "Welcome to AstroShield API"} 