#!/bin/bash

# Create directory for SSL certificates
sudo mkdir -p /etc/nginx/ssl

# Generate a self-signed certificate for testing
sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout /etc/nginx/ssl/astroshield.key \
  -out /etc/nginx/ssl/astroshield.crt \
  -subj "/C=US/ST=VA/L=Arlington/O=AstroShield/CN=astroshield.sdataplab.com"

# Set proper permissions
sudo chmod 600 /etc/nginx/ssl/astroshield.key
sudo chmod 644 /etc/nginx/ssl/astroshield.crt

# Update Nginx configuration
sudo cp /tmp/nginx-update/ssl-config.conf /etc/nginx/conf.d/astroshield.conf

# Check Nginx configuration
echo "Checking Nginx configuration..."
sudo nginx -t

# If configuration test is successful, reload Nginx
if [ $? -eq 0 ]; then
    echo "Nginx configuration is valid. Reloading Nginx..."
    sudo systemctl reload nginx
    echo "Nginx reloaded successfully!"
else
    echo "Error in Nginx configuration. Please check the syntax."
    exit 1
fi

echo "SSL setup complete!" 