#!/bin/bash
set -e

echo "=== Fixing port conflicts on EC2 ==="

# Find and kill process using port 3001
echo "Finding process using port 3001..."
PID=$(sudo lsof -ti:3001)
if [ -n "$PID" ]; then
  echo "Killing process $PID that's using port 3001..."
  sudo kill -9 $PID
  echo "Process terminated."
else
  echo "No process found using port 3001."
fi

# Also check for port 3000 (frontend) just to be safe
echo "Finding process using port 3000..."
PID=$(sudo lsof -ti:3000)
if [ -n "$PID" ]; then
  echo "Killing process $PID that's using port 3000..."
  sudo kill -9 $PID
  echo "Process terminated."
else
  echo "No process found using port 3000."
fi

# Restart Docker containers
echo "Restarting Docker containers..."
cd /home/stardrive/astroshield/deployment
sudo docker-compose down
sudo docker-compose up -d

# Verify services are running
echo "Verifying services..."
sudo docker ps

echo "=== Port conflicts resolved ==="
