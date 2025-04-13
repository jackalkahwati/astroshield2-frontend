#!/bin/bash
# Script to start AstroShield services using the simple backend

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting AstroShield services...${NC}"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
  source .venv/bin/activate
  echo -e "${GREEN}Virtual environment activated${NC}"
fi

# Make sure nothing is running on our ports
pkill -f "python mock_services/mock_udl.py" || echo "No mock UDL process found"
pkill -f "python simple_backend.py" || echo "No simple backend process found"

# Start Mock UDL service in background
echo "Starting Mock UDL service..."
python mock_services/mock_udl.py > udl.log 2>&1 &
UDL_PID=$!
echo -e "${GREEN}Mock UDL started with PID: $UDL_PID${NC}"

# Wait for UDL to start
sleep 2

# Start simple backend API on port 5001
echo "Starting simple backend API on port 5001..."
API_PORT=5001 python simple_backend.py > backend.log 2>&1 &
API_PID=$!
echo -e "${GREEN}Backend API started with PID: $API_PID${NC}"

# Save PIDs for the stop script
echo "$UDL_PID" > .udl.pid
echo "$API_PID" > .api.pid

# Instructions
echo
echo -e "${BLUE}Services are now running:${NC}"
echo "- Mock UDL: http://localhost:8888"
echo "- Backend API: http://localhost:5001"
echo "- API Documentation: http://localhost:5001/docs"
echo
echo "To stop services, use: ./stop_astroshield.sh"
echo "To view logs:"
echo "- UDL log: tail -f udl.log"
echo "- Backend log: tail -f backend.log"
echo
echo -e "${BLUE}Try the following endpoints:${NC}"
echo "- API info: curl http://localhost:5001"
echo "- Satellites list: curl http://localhost:5001/api/v1/satellites"
echo "- Health check: curl http://localhost:5001/health" 