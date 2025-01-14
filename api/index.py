from fastapi import FastAPI, Header, HTTPException
from mangum import Mangum
from typing import Optional
import os

app = FastAPI(root_path="/api")

@app.get("/")
async def root():
    return {"message": "Welcome to AstroShield API v1.0"}

@app.get("/cron")
async def cron_job(authorization: Optional[str] = Header(None)):
    expected_auth = "Bearer 1234"
    if not authorization or authorization != expected_auth:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return {"message": "Hello Cron!"}

# Create handler for AWS Lambda
handler = Mangum(app) 