#!/bin/bash

# Create log files with correct permissions
sudo touch /var/log/nginx/astroshield-access.log /var/log/nginx/astroshield-error.log
sudo chmod 644 /var/log/nginx/astroshield-access.log /var/log/nginx/astroshield-error.log
sudo chown nginx:nginx /var/log/nginx/astroshield-access.log /var/log/nginx/astroshield-error.log

# Copy configuration file
sudo cp /tmp/nginx-update/optimal-config.conf /etc/nginx/conf.d/astroshield.conf

# Verify configuration
echo "Checking Nginx configuration..."
sudo nginx -t

# If configuration test is successful, restart Nginx (not just reload)
if [ $? -eq 0 ]; then
    echo "Nginx configuration is valid. Restarting Nginx..."
    sudo systemctl restart nginx
    echo "Nginx restarted successfully!"
else
    echo "Error in Nginx configuration. Please check the syntax."
    exit 1
fi

# Ensure Docker containers are running
echo "Checking Docker containers..."
sudo docker ps

echo "Update complete!" 