#!/bin/bash
# Script to start all AstroShield services

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting AstroShield services...${NC}"

# Allow overriding backend port via environment variable when invoking the script.
# Primary variable: BACKEND_PORT
# Legacy/alternate variable: BACKEND_BACKEND_PORT (some users mistakenly set this)
#   Example: BACKEND_PORT=3001 ./start_astroshield.sh
#   Example: BACKEND_BACKEND_PORT=3001 ./start_astroshield.sh
# Precedence:
#   1. If BACKEND_PORT is set, use it.
#   2. Else if BACKEND_BACKEND_PORT is set, use it.
#   3. Fallback to default 5002.

if [ -z "$BACKEND_PORT" ]; then
  # shellcheck disable=SC2154 # Variable can be set externally
  if [ -n "$BACKEND_BACKEND_PORT" ]; then
    BACKEND_PORT=$BACKEND_BACKEND_PORT
  else
    BACKEND_PORT=5002
  fi
fi

export BACKEND_PORT

# --- Define Ports --- 
export UDL_PORT=8888

# Activate virtual environment
if [ -d ".venv" ]; then
  source .venv/bin/activate
  echo -e "${GREEN}Virtual environment activated${NC}"
fi

# --- Kill existing processes --- 
echo "Attempting to stop any existing services..."
pkill -f "python mock_services/mock_udl.py" || echo "No mock UDL process found"
pkill -f "uvicorn app.main:app --port $BACKEND_PORT" || echo "No backend API process found on port $BACKEND_PORT"

echo "Ensuring ports $UDL_PORT and $BACKEND_PORT are free..."
lsof -ti :$UDL_PORT | xargs kill -9 2>/dev/null || echo "Port $UDL_PORT free."
lsof -ti :$BACKEND_PORT | xargs kill -9 2>/dev/null || echo "Port $BACKEND_PORT free."
sleep 1 # Give OS time to release ports

# Start Mock UDL service in background
echo "Starting Mock UDL service on port $UDL_PORT..."
# Ensure UDL port is used if defined in mock_udl.py (though it's hardcoded there now)
python3 mock_services/mock_udl.py > udl.log 2>&1 &
UDL_PID=$!
sleep 1 # Give it a moment to start

# Verify UDL PID
if ! ps -p $UDL_PID > /dev/null; then
    echo -e "${RED}Failed to start Mock UDL service. Check udl.log.${NC}"
    exit 1
fi
echo -e "${GREEN}Mock UDL started with PID: $UDL_PID${NC}"

# Wait for UDL to start fully
sleep 2

# Test UDL connection
echo "Testing UDL connection..."
python3 test_simple_udl.py
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Warning: UDL service test failed. Check connection to http://localhost:$UDL_PORT. Continuing anyway...${NC}"
fi

# Start backend API on specified port (no reload)
echo "Starting main backend API on port $BACKEND_PORT (no reload)..."
cd backend
# Using exported BACKEND_PORT
python3 -m uvicorn app.main:app --host 0.0.0.0 --port $BACKEND_PORT > ../backend.log 2>&1 &
API_PID=$!
sleep 1 # Give it a moment to start
cd ..

# Verify API PID
if ! ps -p $API_PID > /dev/null; then
    echo -e "${RED}Failed to start Backend API service. Check backend.log.${NC}"
    exit 1
fi
echo -e "${GREEN}Backend API started with PID: $API_PID${NC}"

# Save PIDs for the stop script
echo "$UDL_PID" > .udl.pid
echo "$API_PID" > .api.pid

# Instructions
echo
echo -e "${BLUE}Services should now be running:${NC}"
echo "- Mock UDL: http://localhost:$UDL_PORT"
echo "- Backend API: http://localhost:$BACKEND_PORT"
echo "- API Documentation: http://localhost:$BACKEND_PORT/api/v1/docs"
echo
echo "To stop services, use: ./stop_astroshield.sh"
echo "To view logs:"
echo "- UDL log: tail -f udl.log"
echo "- Backend log: tail -f backend.log"
echo
echo -e "${BLUE}Check backend log for errors: cat backend.log${NC}"
echo -e "${BLUE}Try accessing API: curl http://localhost:$BACKEND_PORT/health${NC}"

