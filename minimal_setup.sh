#!/bin/bash
# Minimal setup script for AstroShield that avoids Nix and complex dependencies
set -e

echo "=== AstroShield Minimal Setup ==="

# Setup Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install minimal dependencies
pip install --upgrade pip
pip install fastapi uvicorn python-dotenv requests pydantic

# Create mock .env file if it doesn't exist
if [ ! -f .env ]; then
  echo "Creating mock .env file with placeholder values..."
  cat > .env << EOT
# UDL Configuration
UDL_USERNAME=test_user
UDL_PASSWORD=test_password
UDL_BASE_URL=https://mock-udl-service.local/api/v1

# API Configuration
API_PORT=5000
FRONTEND_PORT=3000

# Database Configuration
DATABASE_URL=sqlite:///./astroshield.db
EOT
fi

# Create mock UDL service
mkdir -p mock_services
cat > mock_services/mock_udl.py << EOT
#!/usr/bin/env python3
"""
Mock UDL service for local development.
Run with: python mock_services/mock_udl.py
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
    if username and password:
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
EOT

chmod +x mock_services/mock_udl.py

echo "=== Setup Complete ==="
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To start the mock UDL service:"
echo "  python mock_services/mock_udl.py"
echo ""
echo "To start the backend API server:"
echo "  cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 5000"
echo ""
echo "To start the frontend development server:"
echo "  cd frontend && npm install && npm run dev" 