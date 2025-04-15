#!/bin/bash

echo "=== Checking Nginx error logs ==="
sudo tail -n 20 /var/log/nginx/error.log

echo -e "\n=== Testing API connections ==="
echo "Testing API health endpoint directly:"
curl -v http://localhost:8080/health

echo -e "\n=== Testing Frontend connections ==="
echo "Testing Frontend connection directly:"
curl -v http://localhost:3000/ -o /dev/null

echo -e "\n=== Modifying Nginx configuration ==="
sudo cp /etc/nginx/conf.d/astroshield-ssl.conf /etc/nginx/conf.d/astroshield-ssl.conf.bak.$(date +%Y%m%d%H%M%S)

# Create a more specific configuration that focuses on the actual endpoints
sudo bash -c 'cat > /etc/nginx/conf.d/astroshield-ssl.conf << "EOC"
# Main HTTPS server configuration
server {
    listen 443 ssl;
    server_name astroshield.sdataplab.com;
    
    # SSL Configuration
    ssl_certificate /etc/nginx/ssl/astroshield.crt;
    ssl_certificate_key /etc/nginx/ssl/astroshield.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    
    # Remove client certificate verification entirely
    # ssl_verify_client off;  # Comment out to avoid any issues
    
    # Debug logging for troubleshooting
    error_log /var/log/nginx/astroshield-error.log debug;
    
    # API endpoints
    location /api/ {
        proxy_pass http://127.0.0.1:8080/api/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Increase timeouts to avoid 502s from slow API responses
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
        
        # Enable debug logs
        proxy_intercept_errors on;
        error_page 502 = @api_fallback;
    }
    
    # API fallback
    location @api_fallback {
        return 503 '{"error": "API service temporarily unavailable", "status": "error"}';
        add_header Content-Type application/json;
    }
    
    # Health endpoint
    location /health {
        proxy_pass http://127.0.0.1:8080/health;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Increase timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # Frontend - Static and dynamic content
    location / {
        proxy_pass http://127.0.0.1:3000/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Increase timeouts
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
        
        # Debug
        proxy_intercept_errors on;
        error_page 502 = @frontend_fallback;
    }
    
    # Frontend fallback for maintenance mode
    location @frontend_fallback {
        return 503 '<!DOCTYPE html><html><head><title>Maintenance</title><style>body{font-family:Arial,sans-serif;margin:40px;line-height:1.6;}</style></head><body><h1>Maintenance in Progress</h1><p>The AstroShield service is currently undergoing maintenance. Please check back shortly.</p></body></html>';
        add_header Content-Type text/html;
    }
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name astroshield.sdataplab.com;
    return 301 https://$host$request_uri;
}
EOC'

echo -e "\n=== Testing new configuration ==="
sudo nginx -t

if [ $? -eq 0 ]; then
    echo "Configuration valid, restarting Nginx..."
    sudo systemctl restart nginx
    sleep 2
    sudo systemctl status nginx | head -n 5

    # Check if we need to restart Docker containers as well
    echo -e "\n=== Would you like to restart Docker containers? (y/n) ==="
    read -p "Restart containers? " restart_containers

    if [[ "$restart_containers" == "y" ]]; then
        echo "Restarting Docker containers..."
        sudo docker restart astroshield-frontend-1 astroshield-astroshield-api-1
        sleep 5
        sudo docker ps | grep -E "frontend|api"
        echo "Containers restarted."
    fi
else
    echo "Configuration not valid, please check errors."
    exit 1
fi

echo -e "\n=== Checking for network issues ==="
echo "Testing connection from Nginx to API container:"
sudo docker exec -it astroshield-astroshield-api-1 curl -I http://localhost:8080/health || echo "Failed to connect to API"

echo -e "\n=== Creating a simple static HTML fallback ==="
sudo mkdir -p /var/www/html
sudo bash -c 'cat > /var/www/html/index.html << EOF
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AstroShield</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 40px; line-height: 1.6; }
        .container { max-width: 800px; margin: 0 auto; }
        h1 { color: #333; }
    </style>
</head>
<body>
    <div class="container">
        <h1>AstroShield Platform</h1>
        <p>Welcome to the AstroShield platform. The application is currently starting up.</p>
        <p>If you continue to see this page, please contact system administration.</p>
    </div>
</body>
</html>
EOF'

echo -e "\n=== Testing from another location ==="
echo "Access the site from your browser at https://astroshield.sdataplab.com"

echo -e "\n=== Add static fallback to Nginx config ==="
sudo bash -c "cat >> /etc/nginx/conf.d/astroshield-ssl.conf << 'EOF'

# Static fallback if all else fails
location @error_fallback {
    root /var/www/html;
    try_files /index.html =502;
}
EOF"

sudo nginx -t && sudo systemctl restart nginx

echo "Fix completed. Check the website now." 