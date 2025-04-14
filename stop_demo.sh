#!/bin/bash
# stop_demo.sh - Stop AstroShield demo services

# ANSI color codes for better readability
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
RESET="\033[0m"

echo -e "${RED}Stopping AstroShield Demo${RESET}"

# Check if PID file exists
if [ -f "demo_pids.txt" ]; then
    # Read PIDs from file
    read BACKEND_PID FRONTEND_PID < demo_pids.txt
    
    # Kill backend process
    if [ -n "$BACKEND_PID" ]; then
        echo -e "${YELLOW}Stopping backend process (PID: $BACKEND_PID)...${RESET}"
        kill $BACKEND_PID 2>/dev/null || true
    fi
    
    # Kill frontend process
    if [ -n "$FRONTEND_PID" ]; then
        echo -e "${YELLOW}Stopping frontend process (PID: $FRONTEND_PID)...${RESET}"
        kill $FRONTEND_PID 2>/dev/null || true
    fi
    
    # Remove PID file
    rm demo_pids.txt
else
    echo "No PID file found, trying to kill processes by port..."
    
    # Try to kill processes by finding their port
    BACKEND_PID=$(lsof -ti:5002)
    if [ -n "$BACKEND_PID" ]; then
        echo -e "${YELLOW}Stopping backend process on port 5002 (PID: $BACKEND_PID)...${RESET}"
        kill $BACKEND_PID 2>/dev/null || true
    fi
    
    FRONTEND_PID=$(lsof -ti:3003)
    if [ -n "$FRONTEND_PID" ]; then
        echo -e "${YELLOW}Stopping frontend process on port 3003 (PID: $FRONTEND_PID)...${RESET}"
        kill $FRONTEND_PID 2>/dev/null || true
    fi
fi

# Double-check that ports are free
echo "Checking if ports are free..."
BACKEND_STILL_RUNNING=$(lsof -ti:5002)
FRONTEND_STILL_RUNNING=$(lsof -ti:3003)

if [ -n "$BACKEND_STILL_RUNNING" ] || [ -n "$FRONTEND_STILL_RUNNING" ]; then
    echo -e "${RED}Some processes are still running:${RESET}"
    [ -n "$BACKEND_STILL_RUNNING" ] && echo -e "Backend (PID: $BACKEND_STILL_RUNNING) still on port 5002"
    [ -n "$FRONTEND_STILL_RUNNING" ] && echo -e "Frontend (PID: $FRONTEND_STILL_RUNNING) still on port 3003"
    echo -e "${YELLOW}You may need to kill these processes manually.${RESET}"
else
    echo -e "${GREEN}All AstroShield demo services stopped successfully.${RESET}"
fi 