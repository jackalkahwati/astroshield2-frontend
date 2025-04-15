#!/bin/bash
set -e

echo "=== AstroShield Firefox-Compatible Connection Helper ==="
echo "This script will establish an SSH tunnel using Firefox-friendly ports"

# Kill any existing processes using these ports
echo "Cleaning up any existing processes..."
for port in 3000 8000 8080 8888 9000 10080; do
  pid=$(lsof -ti:$port 2>/dev/null || true)
  if [ -n "$pid" ]; then
    echo "Killing process on port $port (PID: $pid)"
    kill -9 $pid 2>/dev/null || true
  fi
done

# Wait for ports to be released
sleep 2

# Connect with a Firefox-friendly port (8000)
echo "Establishing SSH tunnel on standard web port 8000..."

# Critical: Use the -N flag to keep connection open without executing a command
# Don't use the -f flag as this might cause the connection to close prematurely
ssh -v -N -L 8000:localhost:80 astroshield

# Note: This script will not return until the SSH connection is closed
# Run this script in a separate terminal window and keep it open 