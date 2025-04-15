#!/bin/bash

# This script creates a simpler Nginx configuration to fix the AstroShield SSL issues
# Must be run with sudo privileges

echo "Backing up current configuration..."
sudo cp /etc/nginx/conf.d/astroshield-ssl.conf /etc/nginx/conf.d/astroshield-ssl.conf.bak.$(date +%Y%m%d%H%M%S)

echo "Creating new simplified configuration..."
sudo bash -c 'cat > /etc/nginx/conf.d/astroshield-ssl.conf << EOF
server {
    listen 443 ssl;
    server_name astroshield.sdataplab.com;
    
    # SSL Configuration
    ssl_certificate /etc/nginx/ssl/astroshield.crt;
    ssl_certificate_key /etc/nginx/ssl/astroshield.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    
    # API endpoints
    location /api/ {
        proxy_pass http://127.0.0.1:8080/api/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
    
    # Frontend
    location / {
        proxy_pass http://127.0.0.1:3000/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
    
    # Health endpoint
    location /health {
        proxy_pass http://127.0.0.1:8080/health;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name astroshield.sdataplab.com;
    return 301 https://\$host\$request_uri;
}
EOF'

echo "Testing Nginx configuration..."
sudo nginx -t

if [ $? -eq 0 ]; then
    echo "Configuration test successful. Restarting Nginx..."
    sudo systemctl restart nginx
    echo "Nginx restarted. Please check https://astroshield.sdataplab.com"
else
    echo "Configuration test failed. Please check the errors above."
    echo "Reverting to the previous configuration..."
    sudo cp "$(ls -t /etc/nginx/conf.d/astroshield-ssl.conf.bak.* | head -1)" /etc/nginx/conf.d/astroshield-ssl.conf
    sudo systemctl restart nginx
fi 