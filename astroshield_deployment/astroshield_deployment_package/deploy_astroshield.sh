#!/bin/bash

# Exit on error
set -e

echo "Starting AstroShield deployment..."

# Create necessary directories
mkdir -p /var/www/astroshield
mkdir -p /var/log/astroshield

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y python3-venv nginx certbot python3-certbot-nginx

# Set up Python virtual environment
echo "Setting up Python virtual environment..."
python3 -m venv /var/www/astroshield/venv
source /var/www/astroshield/venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r /var/www/astroshield/requirements.txt

# Configure Nginx
echo "Configuring Nginx..."
sudo cp /var/www/astroshield/nginx/astroshield.conf /etc/nginx/sites-available/
sudo ln -sf /etc/nginx/sites-available/astroshield.conf /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Test Nginx configuration
echo "Testing Nginx configuration..."
sudo nginx -t

# Set up SSL certificate
echo "Setting up SSL certificate..."
sudo certbot --nginx -d astroshield.sdataplab.com --non-interactive --agree-tos --email admin@sdataplab.com

# Set up systemd service
echo "Setting up systemd service..."
sudo cp /var/www/astroshield/systemd/astroshield.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable astroshield
sudo systemctl start astroshield

# Set up monitoring script
echo "Setting up monitoring script..."
chmod +x /var/www/astroshield/scripts/monitor_astroshield.sh

# Create backup directory
mkdir -p /var/backups/astroshield

echo "Deployment completed successfully!"
echo "Please check the application at https://astroshield.sdataplab.com" 