#!/bin/bash

# This script fixes Nginx proxy settings to resolve 502 Bad Gateway errors
# Must be run with sudo privileges

echo "======= AstroShield Nginx Proxy Fix Tool ======="

# Backup current configuration
echo "Backing up current configuration..."
sudo cp /etc/nginx/conf.d/astroshield-ssl.conf /etc/nginx/conf.d/astroshield-ssl.conf.bak.$(date +%Y%m%d%H%M%S)

# Create a new configuration
echo "Creating new configuration with improved proxy settings..."
sudo bash -c 'cat > /etc/nginx/conf.d/astroshield-ssl.conf << EOF
# Main HTTPS server configuration
server {
    listen 443 ssl;
    server_name astroshield.sdataplab.com;
    
    # SSL Configuration
    ssl_certificate /etc/nginx/ssl/astroshield.crt;
    ssl_certificate_key /etc/nginx/ssl/astroshield.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    
    # Disable client certificate verification
    ssl_verify_client off;
    
    # Proxy buffer settings
    proxy_buffers 16 16k;
    proxy_buffer_size 16k;
    
    # Timeouts
    proxy_connect_timeout 300s;
    proxy_send_timeout 300s;
    proxy_read_timeout 300s;
    
    # API endpoints
    location /api/ {
        proxy_pass http://127.0.0.1:8080/api/;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # Disable buffering for API
        proxy_buffering off;
    }
    
    # Frontend
    location / {
        proxy_pass http://127.0.0.1:3000/;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Special Next.js settings
        proxy_cache_bypass \$http_upgrade;
    }
    
    # Health endpoint
    location /health {
        proxy_pass http://127.0.0.1:8080/health;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        
        # Disable buffering for health checks
        proxy_buffering off;
    }
    
    # Extended timeouts for long-running operations
    fastcgi_read_timeout 300s;
    client_max_body_size 50M;
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
    echo "Nginx restarted with improved proxy settings."
    echo "This should fix the 502 Bad Gateway errors."
else
    echo "Configuration test failed. Please check the errors above."
    echo "Reverting to the previous configuration..."
    sudo cp "$(ls -t /etc/nginx/conf.d/astroshield-ssl.conf.bak.* | head -1)" /etc/nginx/conf.d/astroshield-ssl.conf
    sudo systemctl restart nginx
fi 