#!/bin/bash
# start_demo.sh - Start AstroShield demo with simplified backend

set -e  # Exit on error

# ANSI color codes for better readability
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
BLUE="\033[0;34m"
RESET="\033[0m"

echo -e "${GREEN}Starting AstroShield Demo${RESET}"
echo "This script launches the frontend and simplified backend for AstroShield."

# Stop any existing services
echo -e "${YELLOW}Stopping any existing services...${RESET}"
./stop_demo.sh || true  # Don't exit if this fails

# Make sure ports are free
echo -e "${YELLOW}Ensuring ports are free...${RESET}"
kill $(lsof -ti:5002) 2>/dev/null || echo "Port 5002 is free"
kill $(lsof -ti:3003) 2>/dev/null || echo "Port 3003 is free"

# Start simplified backend
echo -e "${BLUE}Starting simplified backend...${RESET}"
python simple_backend.py &
BACKEND_PID=$!
echo "Backend process started with PID: $BACKEND_PID"

# Wait for backend to start
echo "Waiting for backend to start..."
sleep 3

# Test if backend is up
if curl -s http://localhost:5002/health > /dev/null; then
    echo -e "${GREEN}Backend is up and running!${RESET}"
else
    echo -e "${YELLOW}Backend may not be fully started yet, but proceeding...${RESET}"
fi

# Start frontend
echo -e "${BLUE}Starting frontend...${RESET}"
cd frontend
npm run dev &
FRONTEND_PID=$!
echo "Frontend process started with PID: $FRONTEND_PID"
cd ..

# Save PIDs to file for cleanup
echo "$BACKEND_PID $FRONTEND_PID" > demo_pids.txt

echo -e "${GREEN}AstroShield Demo Started${RESET}"
echo -e "Backend URL: ${BLUE}http://localhost:5002${RESET}"
echo -e "Frontend URL: ${BLUE}http://localhost:3003${RESET}"
echo -e "Trajectory Analysis: ${BLUE}http://localhost:3003/trajectory${RESET}"
echo -e "Maneuvers Page: ${BLUE}http://localhost:3003/maneuvers${RESET}"
echo ""
echo -e "To stop the demo, run: ${YELLOW}./stop_demo.sh${RESET}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to exit this terminal, the services will continue running in the background.${RESET}"

wait 