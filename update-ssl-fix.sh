#!/bin/bash

# Ensure SSL directory exists
sudo mkdir -p /etc/nginx/ssl

# Generate a self-signed certificate for testing if it doesn't exist
if [ ! -f /etc/nginx/ssl/astroshield.crt ]; then
    echo "Generating self-signed SSL certificate..."
    sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
      -keyout /etc/nginx/ssl/astroshield.key \
      -out /etc/nginx/ssl/astroshield.crt \
      -subj "/C=US/ST=VA/L=Arlington/O=AstroShield/CN=astroshield.sdataplab.com"
    
    # Set proper permissions
    sudo chmod 600 /etc/nginx/ssl/astroshield.key
    sudo chmod 644 /etc/nginx/ssl/astroshield.crt
fi

# Copy configuration file
sudo cp /tmp/nginx-update/ssl-fix-config.conf /etc/nginx/conf.d/astroshield.conf

# Verify configuration
echo "Checking Nginx configuration..."
sudo nginx -t

# If configuration test is successful, restart Nginx
if [ $? -eq 0 ]; then
    echo "Nginx configuration is valid. Restarting Nginx..."
    sudo systemctl restart nginx
    echo "Nginx restarted successfully!"
else
    echo "Error in Nginx configuration. Please check the syntax."
    exit 1
fi

# Check Nginx status after restart
sudo systemctl status nginx

echo "Update complete! Testing connectivity..."
curl -s -k -I https://localhost/settings 