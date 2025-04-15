#!/bin/bash

# This script configures Nginx to make client certificates optional
# Must be run with sudo privileges

echo "======= AstroShield Client Certificate Fix Tool ======="

# Backup current configuration
echo "Backing up current configuration..."
sudo cp /etc/nginx/conf.d/astroshield-ssl.conf /etc/nginx/conf.d/astroshield-ssl.conf.bak.$(date +%Y%m%d%H%M%S)

# Create a new configuration
echo "Creating new configuration without client certificate requirement..."
sudo bash -c 'cat > /etc/nginx/conf.d/astroshield-ssl.conf << EOF
# Main HTTPS server configuration
server {
    listen 443 ssl;
    server_name astroshield.sdataplab.com;
    
    # SSL Configuration
    ssl_certificate /etc/nginx/ssl/astroshield.crt;
    ssl_certificate_key /etc/nginx/ssl/astroshield.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    
    # Disable client certificate verification entirely
    ssl_verify_client off;
    
    # API endpoints
    location /api/ {
        proxy_pass http://127.0.0.1:8080/api/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
    
    # Frontend
    location / {
        proxy_pass http://127.0.0.1:3000/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
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

# Test and restart Nginx
echo "Testing Nginx configuration..."
sudo nginx -t

if [ $? -eq 0 ]; then
    echo "Configuration test successful. Restarting Nginx..."
    sudo systemctl restart nginx
    echo "Nginx restarted with client certificate verification disabled."
    echo "You should now be able to access the site without a client certificate."
else
    echo "Configuration test failed. Please check the errors above."
    echo "Reverting to the previous configuration..."
    sudo cp "$(ls -t /etc/nginx/conf.d/astroshield-ssl.conf.bak.* | head -1)" /etc/nginx/conf.d/astroshield-ssl.conf
    sudo systemctl restart nginx
fi 