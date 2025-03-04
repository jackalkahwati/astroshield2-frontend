#!/bin/bash

# Kill any processes using our ports
kill_port() {
    local port=$1
    local pid=$(lsof -t -i:$port)
    if [ ! -z "$pid" ]; then
        echo "Killing process using port $port..."
        kill -9 $pid
    fi
}

# Kill processes on ports 3000 and 8000
kill_port 3000
kill_port 8000

# Start backend
echo "Starting backend server..."
cd backend
source venv/bin/activate
python -m uvicorn app.main:app --reload --port 8000 &
cd ..

# Wait for backend to start
sleep 2

# Start frontend
echo "Starting frontend server..."
cd v0
npm run dev &
cd ..

# Wait for user input
echo "Press Ctrl+C to stop both servers"
wait 