from fastapi import FastAPI, Header, HTTPException
from mangum import Mangum
from typing import Optional
import os

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to AstroShield API v1.0"}

@app.get("/cron")
async def cron_job(authorization: Optional[str] = Header(None)):
    if not authorization or authorization != f"Bearer {os.getenv('CRON_SECRET')}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    return {"message": "Hello Cron!"}

# Create handler for AWS Lambda with base path
handler = Mangum(app, api_gateway_base_path="/api") 