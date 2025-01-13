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

# Include the router with /api prefix
app.include_router(router, prefix="/api")

# Create handler for AWS Lambda
handler = Mangum(app)

@app.get("/")
async def root():
    return {"message": "Welcome to AstroShield API"} 