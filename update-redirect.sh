#!/bin/bash

# Copy configuration file
sudo cp /tmp/nginx-update/redirect-config.conf /etc/nginx/conf.d/astroshield.conf

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

echo "Update complete!" 