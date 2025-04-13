#!/bin/bash

# Copy configuration file to its destination
sudo cp /tmp/nginx-update/simple-config.conf /etc/nginx/conf.d/astroshield.conf

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

echo "Update complete!" 