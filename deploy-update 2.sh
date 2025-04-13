#!/bin/bash

# Copy configuration files to their destinations
sudo cp /tmp/nginx-update/astroshield.conf /etc/nginx/conf.d/astroshield.conf
sudo mkdir -p /var/www/html
sudo cp /tmp/nginx-update/index-static.html /var/www/html/

# Set correct permissions
sudo chown -R www-data:www-data /var/www/html
sudo chmod -R 755 /var/www/html

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

# Clean up
rm -rf /tmp/nginx-update
echo "Deployment complete!" 