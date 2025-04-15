#!/bin/bash
set -e

echo "=== AstroShield Connection Helper ==="
echo "This script will establish a persistent SSH tunnel"

# Kill any existing processes using these ports
echo "Cleaning up any existing processes..."
for port in 7777 8080 8443 8888 9000 9001 7443 3001 3005 5432; do
  pid=$(lsof -ti:$port 2>/dev/null || true)
  if [ -n "$pid" ]; then
    echo "Killing process on port $port (PID: $pid)"
    kill -9 $pid 2>/dev/null || true
  fi
done

# Wait for ports to be released
sleep 2

# Use a much higher port to avoid conflicts
echo "Establishing SSH tunnel on port 10080..."
ssh -v -N -f -L 10080:localhost:80 astroshield

# Check if tunnel was created successfully
if [ $? -eq 0 ]; then
  echo "Tunnel established successfully!"
  echo "✅ Access the dashboard at: http://127.0.0.1:10080"
  echo ""
  echo "To stop the tunnel: pkill -f \"ssh.*10080\""
else
  echo "Failed to establish tunnel. Trying another approach..."
  
  # Alternative with direct jump server
  echo "Trying to connect directly with verbose output..."
  ssh -v -N -L 10080:localhost:80 astroshield
fi 