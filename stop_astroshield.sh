#!/bin/bash
# Script to stop all AstroShield services

echo "Stopping AstroShield services..."

# Stop API server if running
if [ -f .api.pid ]; then
    API_PID=26710
    echo "Stopping API server (PID: )..."
    kill  2>/dev/null || echo "API process already stopped"
    rm .api.pid
else
    # Fallback using pkill
    pkill -f "uvicorn app.main:app" || echo "No backend API process found"
fi

# Stop UDL service if running
if [ -f .udl.pid ]; then
    UDL_PID=26701
    echo "Stopping mock UDL service (PID: )..."
    kill  2>/dev/null || echo "UDL process already stopped"
    rm .udl.pid
else
    # Fallback using pkill
    pkill -f "python mock_services/mock_udl.py" || echo "No mock UDL process found"
fi

echo "All services stopped"
