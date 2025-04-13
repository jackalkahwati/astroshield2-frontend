#!/bin/bash
# Script to verify the implementation of our enhancements

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== AstroShield Implementation Verification ===${NC}"
echo "This script tests if our implementation is working properly"
echo

# Step 1: Check if Python environment is active
echo -e "${BLUE}Step 1: Checking Python environment${NC}"
if python -c "import fastapi" 2>/dev/null; then
    echo -e "${GREEN}✓ Python environment with FastAPI is active${NC}"
else
    echo -e "${YELLOW}! FastAPI not found in Python environment${NC}"
    echo "Activating environment from minimal_setup.sh..."
    
    if [ -d "venv" ]; then
        source venv/bin/activate
        echo -e "${GREEN}✓ Virtual environment activated${NC}"
    else
        echo -e "${YELLOW}! Running minimal setup first${NC}"
        ./minimal_setup.sh
        source venv/bin/activate
    fi
fi
echo

# Step 2: Start the Mock UDL service if needed
echo -e "${BLUE}Step 2: Setting up Mock UDL Service${NC}"
if lsof -i:8888 -sTCP:LISTEN >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Service already running on port 8888${NC}"
else
    echo "Starting Mock UDL service on port 8888..."
    if [ -f "mock_services/mock_udl.py" ]; then
        python mock_services/mock_udl.py &
        MOCK_UDL_PID=$!
        echo -e "${GREEN}✓ Mock UDL service started with PID: $MOCK_UDL_PID${NC}"
        # Wait for service to start
        sleep 2
    else
        echo -e "${RED}✗ mock_udl.py not found, please run minimal_setup.sh first${NC}"
        exit 1
    fi
fi
echo

# Step 3: Check if .env file exists with UDL configuration
echo -e "${BLUE}Step 3: Checking .env configuration${NC}"
if [ -f ".env" ]; then
    echo -e "${GREEN}✓ .env file exists${NC}"
    
    # Update .env to use our mock UDL service
    if grep -q "UDL_BASE_URL=http://localhost:8888" .env; then
        echo -e "${GREEN}✓ .env already configured for mock UDL${NC}"
    else
        echo "Updating .env to use mock UDL..."
        sed -i.bak 's|UDL_BASE_URL=.*|UDL_BASE_URL=http://localhost:8888|g' .env
        echo -e "${GREEN}✓ .env updated to use mock UDL${NC}"
    fi
else
    echo -e "${YELLOW}! .env file not found, creating it...${NC}"
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
fi
echo

# Step 4: Test the UDL client directly
echo -e "${BLUE}Step 4: Testing UDL Client${NC}"
cat > test_udl_client.py << EOT
#!/usr/bin/env python3
"""
Test script for the UDL client.
"""
import asyncio
import os
import sys
from pathlib import Path

# Add backend directory to Python path
backend_path = Path("backend")
sys.path.append(str(backend_path.absolute()))

async def main():
    try:
        # Import the UDL client
        from app.services.udl_client import get_udl_client
        
        # Get UDL client
        print("Getting UDL client...")
        udl_client = await get_udl_client()
        
        # Test authentication
        print("Testing authentication...")
        authenticated = await udl_client.authenticate()
        print(f"Authentication result: {authenticated}")
        
        # Get state vectors
        print("Getting state vectors...")
        state_vectors = await udl_client.get_state_vectors(limit=3)
        
        print(f"Received {len(state_vectors)} state vectors:")
        for i, sv in enumerate(state_vectors):
            print(f"  {i+1}. {sv.id}: {sv.name}")
            print(f"     Position: x={sv.position['x']:.1f}, y={sv.position['y']:.1f}, z={sv.position['z']:.1f}")
            print(f"     Velocity: x={sv.velocity['x']:.1f}, y={sv.velocity['y']:.1f}, z={sv.velocity['z']:.1f}")
        
        print("\nUDL client test successful!")
        return True
    except Exception as e:
        print(f"Error testing UDL client: {str(e)}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
EOT

chmod +x test_udl_client.py
echo "Running UDL client test..."
python test_udl_client.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ UDL client test successful${NC}"
else
    echo -e "${RED}✗ UDL client test failed${NC}"
    exit 1
fi
echo

# Step 5: Test the API endpoints using curl
echo -e "${BLUE}Step 5: Testing API endpoints${NC}"

# Start the FastAPI server if not already running
if lsof -i:5000 -sTCP:LISTEN >/dev/null 2>&1; then
    echo -e "${GREEN}✓ API already running on port 5000${NC}"
else
    echo "Starting API server on port 5000..."
    cd backend
    uvicorn app.main:app --port 5000 --host 0.0.0.0 &
    API_PID=$!
    cd ..
    echo -e "${GREEN}✓ API server started with PID: $API_PID${NC}"
    # Wait for server to start
    sleep 5
fi

# Test the /api/v1/satellites endpoint
echo "Testing /api/v1/satellites endpoint..."
SATELLITES_RESPONSE=$(curl -s http://localhost:5000/api/v1/satellites)
if [[ $SATELLITES_RESPONSE == *"id"* && $SATELLITES_RESPONSE == *"name"* ]]; then
    echo -e "${GREEN}✓ Satellites endpoint successful${NC}"
    # Pretty print first satellite entry
    echo "Sample satellite data:"
    echo "$SATELLITES_RESPONSE" | python -m json.tool | head -20
else
    echo -e "${RED}✗ Satellites endpoint failed${NC}"
    echo "Response: $SATELLITES_RESPONSE"
    exit 1
fi
echo

# Test the mock UDL status endpoint
echo "Testing /api/v1/mock-udl-status endpoint..."
UDL_STATUS_RESPONSE=$(curl -s http://localhost:5000/api/v1/mock-udl-status)
if [[ $UDL_STATUS_RESPONSE == *"mock_mode"* ]]; then
    echo -e "${GREEN}✓ Mock UDL status endpoint successful${NC}"
    echo "Response:"
    echo "$UDL_STATUS_RESPONSE" | python -m json.tool
else
    echo -e "${RED}✗ Mock UDL status endpoint failed${NC}"
    echo "Response: $UDL_STATUS_RESPONSE"
    exit 1
fi
echo

echo -e "${BLUE}=== Verification Summary ===${NC}"
echo -e "${GREEN}✓ Python environment is set up correctly${NC}"
echo -e "${GREEN}✓ Mock UDL service is working${NC}"
echo -e "${GREEN}✓ .env file is configured correctly${NC}"
echo -e "${GREEN}✓ UDL client works properly${NC}"
echo -e "${GREEN}✓ API endpoints are functioning${NC}"
echo
echo -e "${GREEN}All tests passed! The implementation is working as expected.${NC}"
echo
echo "Next steps:"
echo "1. Use the robust_setup.sh script for a full environment setup"
echo "2. Start the frontend with: cd frontend && npm install && npm run dev"
echo "3. Access the API documentation at: http://localhost:5000/api/v1/docs"
echo

# Clean up
if [ -n "$MOCK_UDL_PID" ]; then
    echo "Stopping Mock UDL service (PID: $MOCK_UDL_PID)..."
    kill $MOCK_UDL_PID 2>/dev/null || true
fi

if [ -n "$API_PID" ]; then
    echo "API server is still running (PID: $API_PID)"
    echo "To stop it, run: kill $API_PID"
fi 