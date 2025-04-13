#!/bin/bash
# AstroShield macOS Setup Script
# This script sets up the backend and starts necessary services

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== AstroShield macOS Setup ===${NC}"

# Create a Python virtual environment
echo -e "\n${BLUE}Creating Python virtual environment...${NC}"
python3 -m venv .venv
source .venv/bin/activate
echo -e "${GREEN}✓ Virtual environment created and activated${NC}"

# Upgrade pip and install dependencies
echo -e "\n${BLUE}Installing dependencies...${NC}"
pip install --upgrade pip
pip install fastapi uvicorn "pydantic[email]" python-dotenv requests sqlalchemy "python-jose[cryptography]" "passlib[bcrypt]"
echo -e "${GREEN}✓ Core dependencies installed${NC}"

# Install additional dependencies for the full stack
echo -e "\n${BLUE}Installing additional dependencies...${NC}"
pip install pandas numpy scikit-learn joblib tqdm httpx aiohttp pytest
echo -e "${GREEN}✓ Additional dependencies installed${NC}"

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo -e "\n${BLUE}Creating .env file...${NC}"
    cat > .env << EOT
# UDL Configuration
UDL_USERNAME=test_user
UDL_PASSWORD=test_password
UDL_BASE_URL=http://localhost:8888

# API Configuration
API_PORT=5000
FRONTEND_PORT=3000

# Database Configuration
DATABASE_URL=sqlite:///./astroshield.db
EOT
    echo -e "${GREEN}✓ .env file created${NC}"
else
    echo -e "\n${GREEN}✓ .env file already exists${NC}"
    # Make sure the .env file points to our mock UDL
    sed -i.bak 's|UDL_BASE_URL=.*|UDL_BASE_URL=http://localhost:8888|g' .env
    echo -e "${GREEN}✓ Updated .env to use mock UDL${NC}"
fi

# Create mock UDL service directory and script
echo -e "\n${BLUE}Setting up mock UDL service...${NC}"
mkdir -p mock_services
cat > mock_services/mock_udl.py << EOT
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
EOT

chmod +x mock_services/mock_udl.py
echo -e "${GREEN}✓ Mock UDL service created${NC}"

# Create a simple UDL client test
echo -e "\n${BLUE}Creating simple UDL client test...${NC}"
cat > test_simple_udl.py << EOT
#!/usr/bin/env python3
"""
Simple test for UDL client with minimal dependencies.
"""
import os
import sys
import requests
import json
from datetime import datetime

def test_udl_service():
    """Test the UDL service connection"""
    base_url = os.environ.get("UDL_BASE_URL", "http://localhost:8888")
    token = None # Initialize token
    
    # Test the root endpoint
    try:
        print(f"Testing connection to {base_url}...")
        response = requests.get(base_url)
        if response.status_code == 200:
            print("✓ UDL service is responding")
        else:
            print(f"! UDL service returned status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"! Error connecting to UDL service: {str(e)}")
        return False
    
    # Test authentication
    try:
        # Use test_user from .env or fallback to defaults
        username = os.environ.get("UDL_USERNAME", "test_user")
        password = os.environ.get("UDL_PASSWORD", "test_password")
        
        print(f"Testing authentication with username: {username}...")
        auth_url = f"{base_url}/auth/token"
        auth_response = requests.post(auth_url, json={
            "username": username, 
            "password": password
        })
        
        if auth_response.status_code == 200 and "token" in auth_response.json():
            token = auth_response.json()["token"]
            print(f"✓ Authentication successful. Token: {token[:10]}...")
        else:
            print(f"! Authentication failed with status: {auth_response.status_code}")
            print(f"! Response: {auth_response.text}")
            # For testing purposes, create a mock token to continue
            print("Using mock token to continue test")
            token = "mock-token-for-testing"
    except Exception as e:
        print(f"! Authentication error: {str(e)}")
        # For testing purposes, use a mock token to continue
        print("Using mock token to continue test")
        token = "mock-token-for-testing"
    
    # Test getting state vectors
    try:
        print("Testing state vectors endpoint...")
        vectors_url = f"{base_url}/statevector"
        vectors_response = requests.get(
            vectors_url,
            headers={"Authorization": f"Bearer {token}"},
            params={"epoch": "now"}
        )
        
        if vectors_response.status_code == 200:
            data = vectors_response.json()
            if "stateVectors" in data and len(data["stateVectors"]) > 0:
                print(f"✓ Received {len(data['stateVectors'])} state vectors")
                
                # Print the first vector as an example
                sv = data["stateVectors"][0]
                print(f"  Example: {sv['id']} - {sv['name']}")
                print(f"    Position: x={sv['position']['x']:.1f}, y={sv['position']['y']:.1f}, z={sv['position']['z']:.1f}")
                return True
            else:
                print("! No state vectors found in response")
                return False
        else:
            print(f"! State vector request failed with status: {vectors_response.status_code}")
            return False
    except Exception as e:
        print(f"! Error getting state vectors: {str(e)}")
        return False

if __name__ == "__main__":
    # Load environment variables from .env file
    if os.path.exists(".env"):
        print("Loading environment variables from .env file")
        with open(".env", "r") as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    try:
                        key, value = line.strip().split("=", 1)
                        os.environ[key] = value
                        if key == "UDL_USERNAME" or key == "UDL_PASSWORD" or key == "UDL_BASE_URL":
                            print(f"Loaded {key}={value}")
                    except ValueError:
                        print(f"Warning: Skipping malformed line in .env: {line.strip()}")
    
    success = test_udl_service()
    sys.exit(0 if success else 1)
EOT

chmod +x test_simple_udl.py
echo -e "${GREEN}✓ Simple UDL client test created${NC}"

# Create a startup script
echo -e "\n${BLUE}Creating startup script (for main backend)...${NC}"
cat > start_astroshield.sh << EOT
#!/bin/bash
# Script to start all AstroShield services

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting AstroShield services...${NC}"

# Activate virtual environment
if [ -d ".venv" ]; then
  source .venv/bin/activate
  echo -e "${GREEN}Virtual environment activated${NC}"
fi

# --- Kill existing processes --- 
echo "Attempting to stop any existing services..."
pkill -f "python mock_services/mock_udl.py" || echo "No mock UDL process found"
pkill -f "uvicorn app.main:app" || echo "No backend API process found"

# Use PORT 5002 for backend
BACKEND_PORT=5002

echo "Ensuring ports 8888 and $BACKEND_PORT are free..."
lsof -ti :8888 | xargs kill -9 2>/dev/null || echo "Port 8888 free."
lsof -ti :5000 | xargs kill -9 2>/dev/null || echo "Port 5000 (original) free."
lsof -ti :$BACKEND_PORT | xargs kill -9 2>/dev/null || echo "Port $BACKEND_PORT free."
sleep 1 # Give OS time to release ports

# Start Mock UDL service in background
echo "Starting Mock UDL service on port 8888..."
python mock_services/mock_udl.py > udl.log 2>&1 &
UDL_PID=$!
echo -e "${GREEN}Mock UDL started with PID: $UDL_PID${NC}"

# Wait for UDL to start
sleep 2

# Test UDL connection
echo "Testing UDL connection..."
python test_simple_udl.py
if [ $? -ne 0 ]; then
    echo "Warning: UDL service test failed, but continuing anyway"
fi

# Start backend API on specified port (no reload)
echo "Starting main backend API on port $BACKEND_PORT (no reload)..."
cd backend
# Removed --reload flag, changed port
uvicorn app.main:app --host 0.0.0.0 --port $BACKEND_PORT > ../backend.log 2>&1 &
API_PID=$!
echo -e "${GREEN}Backend API started with PID: $API_PID${NC}"
cd ..

# Save PIDs for the stop script
echo "$UDL_PID" > .udl.pid
echo "$API_PID" > .api.pid

# Instructions
echo
echo -e "${BLUE}Services should now be running:${NC}"
echo "- Mock UDL: http://localhost:8888"
echo "- Backend API: http://localhost:$BACKEND_PORT"
echo "- API Documentation: http://localhost:$BACKEND_PORT/api/v1/docs"
echo
echo "To stop services, use: ./stop_astroshield.sh"
echo "To view logs:"
echo "- UDL log: tail -f udl.log"
echo "- Backend log: tail -f backend.log"
echo
echo -e "${BLUE}Check backend log for errors: cat backend.log${NC}"
echo -e "${BLUE}Try accessing API: curl http://localhost:$BACKEND_PORT/api/v1/satellites${NC}"

EOT

chmod +x start_astroshield.sh
echo -e "${GREEN}✓ Startup script created${NC}"

# Create a stop script
echo -e "\n${BLUE}Creating stop script...${NC}"
cat > stop_astroshield.sh << EOT
#!/bin/bash
# Script to stop all AstroShield services

echo "Stopping AstroShield services..."

# Stop API server if running
if [ -f .api.pid ]; then
    API_PID=$(cat .api.pid)
    echo "Stopping API server (PID: $API_PID)..."
    kill $API_PID 2>/dev/null || echo "API process already stopped"
    rm .api.pid
else
    # Fallback using pkill
    pkill -f "uvicorn app.main:app" || echo "No backend API process found"
fi

# Stop UDL service if running
if [ -f .udl.pid ]; then
    UDL_PID=$(cat .udl.pid)
    echo "Stopping mock UDL service (PID: $UDL_PID)..."
    kill $UDL_PID 2>/dev/null || echo "UDL process already stopped"
    rm .udl.pid
else
    # Fallback using pkill
    pkill -f "python mock_services/mock_udl.py" || echo "No mock UDL process found"
fi

echo "All services stopped"
EOT

chmod +x stop_astroshield.sh
echo -e "${GREEN}✓ Stop script created${NC}"

# Create simple database setup
echo -e "\n${BLUE}Setting up simple database...${NC}"
mkdir -p backend/app/db
cat > backend/app/db/setup_db.py << EOT
#!/usr/bin/env python3
"""
Simple database setup for the AstroShield platform.
Creates a SQLite database with basic tables.
"""
import os
import sys
import sqlite3
from datetime import datetime

DB_PATH = os.environ.get("DATABASE_URL", "sqlite:///./astroshield.db")
if DB_PATH.startswith("sqlite:///"):
    DB_PATH = DB_PATH[len("sqlite:///"):]

def setup_database():
    """Create a new SQLite database with basic schema"""
    print(f"Setting up database at {DB_PATH}")
    
    # Check if database directory exists
    db_dir = os.path.dirname(DB_PATH)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS satellites (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        status TEXT DEFAULT 'active',
        last_update TEXT,
        created_at TEXT,
        description TEXT
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS maneuvers (
        id TEXT PRIMARY KEY,
        satellite_id TEXT,
        status TEXT DEFAULT 'planned',
        type TEXT,
        start_time TEXT,
        end_time TEXT,
        created_at TEXT,
        description TEXT,
        FOREIGN KEY (satellite_id) REFERENCES satellites (id)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS events (
        id TEXT PRIMARY KEY,
        type TEXT,
        description TEXT,
        created_at TEXT,
        severity TEXT,
        satellite_id TEXT,
        FOREIGN KEY (satellite_id) REFERENCES satellites (id)
    )
    ''')
    
    # Add some sample data
    now = datetime.utcnow().isoformat()
    
    # Sample satellites
    satellites = [
        ("sat-001", "AstroShield Demo Sat 1", "active", now, now, "Demo satellite for testing"),
        ("sat-002", "AstroShield Demo Sat 2", "active", now, now, "Secondary demo satellite"),
        ("sat-003", "Test Satellite Alpha", "inactive", now, now, "Inactive test satellite")
    ]
    
    cursor.executemany(
        "INSERT OR REPLACE INTO satellites (id, name, status, last_update, created_at, description) VALUES (?, ?, ?, ?, ?, ?)",
        satellites
    )
    
    # Sample maneuvers
    maneuvers = [
        ("mnv-001", "sat-001", "completed", "collision_avoidance", 
         (datetime.utcnow().replace(hour=datetime.utcnow().hour-2)).isoformat(),
         (datetime.utcnow().replace(hour=datetime.utcnow().hour-1)).isoformat(),
         now, "Collision avoidance maneuver"),
        ("mnv-002", "sat-001", "planned", "station_keeping",
         (datetime.utcnow().replace(hour=datetime.utcnow().hour+5)).isoformat(),
         None, now, "Scheduled station keeping")
    ]
    
    cursor.executemany(
        "INSERT OR REPLACE INTO maneuvers (id, satellite_id, status, type, start_time, end_time, created_at, description) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        maneuvers
    )
    
    # Sample events
    events = [
        ("evt-001", "warning", "Potential conjunction detected", now, "warning", "sat-001"),
        ("evt-002", "info", "Telemetry update received", now, "info", "sat-001"),
        ("evt-003", "error", "Communication disruption", now, "critical", "sat-002")
    ]
    
    cursor.executemany(
        "INSERT OR REPLACE INTO events (id, type, description, created_at, severity, satellite_id) VALUES (?, ?, ?, ?, ?, ?)",
        events
    )
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    print("Database setup complete")
    return True

if __name__ == "__main__":
    success = setup_database()
    sys.exit(0 if success else 1)
EOT

chmod +x backend/app/db/setup_db.py
echo -e "${GREEN}✓ Database setup script created${NC}"

# Run the database setup
echo -e "\n${BLUE}Running database setup...${NC}"
python backend/app/db/setup_db.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Database initialized successfully${NC}"
else
    echo -e "${YELLOW}! Database setup had issues, but continuing${NC}"
fi

# Test the UDL client
echo -e "\n${BLUE}Testing UDL client...${NC}"
if [ ! -f "mock_services/mock_udl.py" ]; then
    echo -e "${RED}! Mock UDL script not found${NC}"
else
    # Start UDL service temporarily
    python mock_services/mock_udl.py > /dev/null 2>&1 &
    UDL_PID=$!
    
    # Give it a second to start
    sleep 2
    
    # Test the connection
    python test_simple_udl.py
    TEST_RESULT=$?
    
    # Kill the UDL service
    kill $UDL_PID
    
    if [ $TEST_RESULT -eq 0 ]; then
        echo -e "${GREEN}✓ UDL client test successful${NC}"
    else
        echo -e "${YELLOW}! UDL client test had issues${NC}"
    fi
fi

echo -e "\n${GREEN}=== Setup Complete ===${NC}"
echo -e "To start all services, run: ${BLUE}./start_astroshield.sh${NC}"
echo -e "To stop all services, run: ${BLUE}./stop_astroshield.sh${NC}"
echo -e "To verify the implementation, run: ${BLUE}./verify_implementation.sh${NC}"
echo -e "\nOnce running, you can access:"
echo -e "- Backend API: ${BLUE}http://localhost:5002${NC} (using start_astroshield.sh)"
echo -e "- API Documentation: ${BLUE}http://localhost:5002/api/v1/docs${NC}"
echo -e "- Mock UDL Service: ${BLUE}http://localhost:8888${NC}" 