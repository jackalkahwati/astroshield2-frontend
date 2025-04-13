#!/usr/bin/env python3
"""
Mock UDL service for local development.
"""
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
from datetime import datetime
import random

app = FastAPI(title="Mock UDL Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "online", "service": "Mock UDL"}

@app.post("/auth/token")
def get_token(username: str = "", password: str = ""):
    """Mock authentication endpoint"""
    # Allow test_user or the actual username found in .env
    valid_user = os.environ.get("UDL_USERNAME", "test_user")
    valid_pass = os.environ.get("UDL_PASSWORD", "test_password")
    if username == valid_user and password == valid_pass:
        return {"token": "mock-udl-token-for-testing"}
    if username == "test_user" and password == "test_password": # Also allow default if .env isn't loaded
        return {"token": "mock-udl-token-for-testing"}

    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/statevector")
def get_state_vectors(epoch: str = "now"):
    """Mock state vector endpoint"""
    return {
        "stateVectors": [
            {
                "id": f"sat-{i}",
                "name": f"Test Satellite {i}",
                "epoch": datetime.utcnow().isoformat(),
                "position": {
                    "x": random.uniform(-7000, 7000),
                    "y": random.uniform(-7000, 7000),
                    "z": random.uniform(-7000, 7000)
                },
                "velocity": {
                    "x": random.uniform(-7, 7),
                    "y": random.uniform(-7, 7),
                    "z": random.uniform(-7, 7)
                }
            }
            for i in range(1, 11)
        ]
    }

if __name__ == "__main__":
    print("Starting Mock UDL service at http://localhost:8888")
    uvicorn.run(app, host="0.0.0.0", port=8888)
