#!/bin/bash
# Script to start all AstroShield services

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting AstroShield services...${NC}"

# --- Define Ports --- 
export BACKEND_PORT=5002
# UDL_PORT is no longer needed for mock service
# export UDL_PORT=8888 

# Activate virtual environment
if [ -d ".venv" ]; then
  source .venv/bin/activate
  echo -e "${GREEN}Virtual environment activated${NC}"
fi

# --- Load Environment Variables --- 
if [ -f .env ]; then
  set -o allexport
  source .env
  set +o allexport
  echo -e "${GREEN}Loaded environment variables from .env file${NC}"
fi

# --- Kill existing processes --- 
echo "Attempting to stop any existing services..."
# Removed command to kill mock UDL
# pkill -f "python mock_services/mock_udl.py" || echo "No mock UDL process found"
pkill -f "uvicorn app.main:app --port $BACKEND_PORT" || echo "No backend API process found on port $BACKEND_PORT"

# Remove check for UDL_PORT as it's not needed for mock
# echo "Ensuring ports $UDL_PORT and $BACKEND_PORT are free..."
# lsof -ti :$UDL_PORT | xargs kill -9 2>/dev/null || echo "Port $UDL_PORT free."
echo "Ensuring port $BACKEND_PORT is free..."
lsof -ti :$BACKEND_PORT | xargs kill -9 2>/dev/null || echo "Port $BACKEND_PORT free."
sleep 1 # Give OS time to release ports

# --- Remove Mock UDL Startup --- 
# The section for starting and testing the mock UDL service has been removed.
# echo "Starting Mock UDL service on port $UDL_PORT..."
# python mock_services/mock_udl.py > udl.log 2>&1 &
# UDL_PID=$!
# ... (rest of mock UDL logic removed) ...

# --- Test Real UDL Connection (Optional but recommended) --- 
if [ -n "$UDL_BASE_URL" ]; then
    echo "Testing connection to REAL UDL service at $UDL_BASE_URL..."
    # You might need a specific script or command to test the real UDL
    # Using curl as a basic example - adjust as needed
    curl -s --head $UDL_BASE_URL | head -n 1 | grep "HTTP/.* 200" > /dev/null
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Real UDL service seems responsive.${NC}"
    else
        echo -e "${YELLOW}⚠ Warning: Could not confirm connection to real UDL service at $UDL_BASE_URL. Check URL and network.${NC}"
    fi
    # Add authentication test if needed
    # python test_real_udl_auth.py # Example script
else
    echo -e "${YELLOW}⚠ UDL_BASE_URL not set in .env - cannot test real UDL connection.${NC}"
fi

# --- Start backend API --- 
echo "Starting main backend API on port $BACKEND_PORT (no reload)..."
cd backend_fixed # Ensure correct directory for backend
# Using exported BACKEND_PORT
uvicorn app.main:app --host 0.0.0.0 --port $BACKEND_PORT > ../backend.log 2>&1 &
API_PID=$!
sleep 2 # Give it a bit more time to start, especially if connecting to external services
cd ..

# Verify API PID
if ! ps -p $API_PID > /dev/null; then
    echo -e "${RED}Failed to start Backend API service. Check backend.log.${NC}"
    exit 1
fi
echo -e "${GREEN}Backend API started with PID: $API_PID${NC}"

# Save PIDs for the stop script
# Remove UDL_PID saving
# echo "$UDL_PID" > .udl.pid
echo "$API_PID" > .api.pid

# --- Instructions --- 
echo
echo -e "${BLUE}Services should now be running:${NC}"
echo "- Backend API (configured for REAL UDL): http://localhost:$BACKEND_PORT"
echo "- API Documentation: http://localhost:$BACKEND_PORT/api/v1/docs"
echo
echo "To stop services, use: ./stop_astroshield.sh"
echo "To view logs:"
# Remove UDL log reference
# echo "- UDL log: tail -f udl.log"
echo "- Backend log: tail -f backend.log"
echo
echo -e "${BLUE}Check backend log for errors: cat backend.log${NC}"
echo -e "${BLUE}Try accessing API: curl http://localhost:$BACKEND_PORT/health${NC}"

