#!/bin/bash
# Start AstroShield services

set -e

echo "Starting AstroShield services..."

# Start backend service
echo "Starting backend service..."
sudo systemctl start astroshield-backend
sudo systemctl status astroshield-backend --no-pager

# Start frontend service
echo "Starting frontend service..."
sudo systemctl start astroshield-frontend
sudo systemctl status astroshield-frontend --no-pager

# Ensure Nginx is running
echo "Ensuring Nginx is running..."
sudo systemctl start nginx
sudo systemctl status nginx --no-pager

echo "All services started successfully."
echo "AstroShield is now available at https://astroshield.sdataplab.com"