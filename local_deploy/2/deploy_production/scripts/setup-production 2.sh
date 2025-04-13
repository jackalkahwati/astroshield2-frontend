#!/bin/bash
# Full Production Setup Script for AstroShield
# This script sets up the production environment with Nginx and systemd services

set -e

echo "Setting up AstroShield production environment..."

# Stop any existing services
echo "Stopping any existing services..."
sudo systemctl stop astroshield-backend.service 2>/dev/null || true
sudo systemctl stop astroshield-frontend.service 2>/dev/null || true

# Install Nginx configuration
echo "Setting up Nginx configuration..."
sudo mkdir -p /etc/nginx/conf.d
sudo cp /home/stardrive/astroshield/nginx/conf.d/astroshield.conf /etc/nginx/conf.d/

# Set up systemd services
echo "Setting up systemd services..."
sudo cp /home/stardrive/astroshield/scripts/astroshield-backend.service /etc/systemd/system/
sudo cp /home/stardrive/astroshield/scripts/astroshield-frontend.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable astroshield-backend.service
sudo systemctl enable astroshield-frontend.service

# Start services
echo "Starting services..."
sudo systemctl start astroshield-backend.service
sudo systemctl start astroshield-frontend.service

# Restart Nginx
echo "Restarting Nginx..."
sudo systemctl restart nginx

# Check if services are running
echo "Checking service status..."
echo "Backend status:"
sudo systemctl status astroshield-backend.service --no-pager
echo "Frontend status:"
sudo systemctl status astroshield-frontend.service --no-pager
echo "Nginx status:"
sudo systemctl status nginx --no-pager

echo "Setup complete! AstroShield should now be accessible at http://astroshield.sdataplab.com"
echo "Backend API available at http://astroshield.sdataplab.com/api/v1"
echo "Status endpoint at http://astroshield.sdataplab.com/status"