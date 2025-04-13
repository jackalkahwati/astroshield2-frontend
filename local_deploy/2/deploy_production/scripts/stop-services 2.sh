#!/bin/bash
# Stop AstroShield services

set -e

echo "Stopping AstroShield services..."

# Stop frontend service
echo "Stopping frontend service..."
sudo systemctl stop astroshield-frontend

# Stop backend service
echo "Stopping backend service..."
sudo systemctl stop astroshield-backend

echo "All services stopped successfully."