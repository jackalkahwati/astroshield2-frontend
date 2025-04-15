#!/bin/bash
# Script to stop all AstroShield services

# Colors for output
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Stopping AstroShield services...${NC}"

# --- Stop API server --- 
API_PID_FILE=".api.pid"
API_PID=0

if [ -f "$API_PID_FILE" ]; then
    API_PID=$(cat "$API_PID_FILE")
    echo "Stopping API server (PID: $API_PID)..."
    if ps -p $API_PID > /dev/null; then
        kill $API_PID 2>/dev/null
        sleep 1 # Give it a moment to terminate gracefully
        if ps -p $API_PID > /dev/null; then
            echo "API process $API_PID did not stop gracefully, forcing kill..."
            kill -9 $API_PID 2>/dev/null
        fi
        echo "API process stopped."
    else
        echo "API process $API_PID already stopped."
    fi
    rm -f "$API_PID_FILE"
else
    echo "No API PID file found. Attempting fallback using pkill..."
    # Using a more specific pattern if possible, assuming main backend runs from backend_fixed
    pkill -f "uvicorn.*backend_fixed/app/main:app" || echo "No matching backend API process found with pkill."
fi

# --- Remove Mock UDL Stopping Logic --- 
# The section for stopping the mock UDL service has been removed.
# UDL_PID_FILE=".udl.pid"
# UDL_PID=0
# ... (rest of mock UDL stopping logic removed) ...

echo -e "${BLUE}All managed services stopped.${NC}"
