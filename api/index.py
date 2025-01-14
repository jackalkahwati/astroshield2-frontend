from fastapi import FastAPI
from mangum import Mangum

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to AstroShield API v1.0"}

# Create handler for AWS Lambda with base path
handler = Mangum(app, api_gateway_base_path="/api") 