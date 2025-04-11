#!/bin/bash

# This script diagnoses and fixes SSL issues with Nginx configuration
# Must be run with sudo privileges

echo "======= AstroShield SSL Diagnostic and Fix Tool ======="
echo "Running diagnostics..."

# Check for any SSL client certificate requirements in Nginx files
echo "Checking for SSL client certificate requirements..."
CLIENT_CERT_FILES=$(sudo find /etc/nginx -type f -exec grep -l "ssl_client_certificate\|ssl_verify_client" {} \; 2>/dev/null)

if [ -n "$CLIENT_CERT_FILES" ]; then
    echo "Found SSL client certificate requirements in these files:"
    echo "$CLIENT_CERT_FILES"
    echo "Disabling client certificate verification..."
    
    for FILE in $CLIENT_CERT_FILES; do
        echo "Backing up $FILE..."
        sudo cp "$FILE" "$FILE.bak.$(date +%Y%m%d%H%M%S)"
        
        echo "Modifying $FILE to disable client certificate verification..."
        sudo sed -i 's/ssl_verify_client on;/ssl_verify_client off;/g' "$FILE"
        sudo sed -i 's/ssl_verify_client optional;/ssl_verify_client off;/g' "$FILE"
    done
else
    echo "No SSL client certificate requirements found in configuration files."
fi

# Check SSL certificate
echo "Checking SSL certificate..."
sudo openssl x509 -in /etc/nginx/ssl/astroshield.crt -text -noout | grep "Subject:"

# Create a new clean configuration
echo "Creating a new clean configuration..."
sudo bash -c 'cat > /etc/nginx/conf.d/astroshield-clean.conf << EOF
# Main HTTPS server configuration
server {
    listen 443 ssl;
    server_name astroshield.sdataplab.com;
    
    # SSL Configuration
    ssl_certificate /etc/nginx/ssl/astroshield.crt;
    ssl_certificate_key /etc/nginx/ssl/astroshield.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    
    # Explicitly disable client certificate verification
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

# Move the existing file aside
echo "Moving existing configuration aside..."
sudo mv /etc/nginx/conf.d/astroshield-ssl.conf /etc/nginx/conf.d/astroshield-ssl.conf.bak.$(date +%Y%m%d%H%M%S)
sudo mv /etc/nginx/conf.d/astroshield-clean.conf /etc/nginx/conf.d/astroshield-ssl.conf

# Check for any additional modules that might be loading SSL configuration
echo "Checking for additional modules or global SSL configuration..."
NGINX_MODULES=$(sudo find /usr/share/nginx/modules -type f -name "*.conf" 2>/dev/null)
if [ -n "$NGINX_MODULES" ]; then
    echo "Found Nginx modules:"
    echo "$NGINX_MODULES"
    for MODULE in $NGINX_MODULES; do
        echo "Contents of $MODULE:"
        sudo cat "$MODULE"
    done
fi

# Restart Nginx with diagnostic output
echo "Testing and restarting Nginx..."
sudo nginx -t

if [ $? -eq 0 ]; then
    echo "Configuration test successful. Restarting Nginx..."
    sudo systemctl restart nginx
    
    echo "Checking Nginx status after restart..."
    sudo systemctl status nginx
    
    echo "Verifying Nginx listening ports..."
    sudo ss -tulpn | grep nginx
    
    echo "Nginx restarted. Please check https://astroshield.sdataplab.com"
else
    echo "Configuration test failed. Please check the errors above."
    echo "Reverting to the previous configuration..."
    sudo cp "$(ls -t /etc/nginx/conf.d/astroshield-ssl.conf.bak.* | head -1)" /etc/nginx/conf.d/astroshield-ssl.conf
    sudo systemctl restart nginx
fi 