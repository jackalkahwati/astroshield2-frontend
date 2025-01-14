from fastapi import FastAPI
from mangum import Mangum
from endpoints import router

app = FastAPI()

# Include the router with /api prefix
app.include_router(router, prefix="/api")

# Create handler for AWS Lambda
handler = Mangum(app, api_gateway_base_path="/api")

@app.get("/api")
async def root():
    return {"message": "Welcome to AstroShield API v1.0"} 