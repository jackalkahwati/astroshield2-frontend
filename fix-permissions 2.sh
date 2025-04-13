#!/bin/bash

# Fix permissions for frontend public directory
sudo chmod -R 755 /home/stardrive/astroshield/frontend/public
sudo chown -R nginx:nginx /home/stardrive/astroshield/frontend/public

# Fix permissions for /var/www/html
sudo chmod -R 755 /var/www/html
sudo chown -R nginx:nginx /var/www/html

echo "Permissions fixed!" 