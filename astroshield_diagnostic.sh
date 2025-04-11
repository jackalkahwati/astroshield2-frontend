#!/bin/bash

echo "======================== ASTROSHIELD DIAGNOSTIC ========================"

echo -e "\n=== Checking processes and ports ==="
sudo ss -tulpn | grep -E ':3000|:8080|:443|:80'

echo -e "\n=== Checking Docker containers ==="
echo "Docker containers:"
sudo docker ps -a

echo -e "\n=== Checking Docker container health ==="
echo "Frontend container logs (last 10 lines):"
sudo docker logs astroshield-frontend-1 --tail 10 2>/dev/null || echo "Frontend container logs not available"

echo "API container logs (last 10 lines):"
sudo docker logs astroshield-astroshield-api-1 --tail 10 2>/dev/null || echo "API container logs not available"

echo -e "\n=== Checking logs ==="
echo "Nginx error logs:"
sudo tail -n 15 /var/log/nginx/error.log 2>/dev/null || echo "Nginx error log not found"

echo -e "\n=== Nginx configuration ==="
NGINX_CONF=$(sudo find /etc/nginx -name "*.conf" -type f | grep -v "mime.types" | head -n 1)
if [ -n "$NGINX_CONF" ]; then
    echo "Found Nginx config: $NGINX_CONF"
    echo "Checking backend port configuration:"
    sudo grep -A 2 "proxy_pass" /etc/nginx/conf.d/astroshield-ssl.conf | grep -E "localhost:[0-9]+"
else
    echo "No Nginx config found"
fi

echo -e "\n=== SSL Certificates ==="
echo "Checking SSL certificates:"
sudo ls -la /etc/nginx/ssl/ 2>/dev/null || echo "SSL directory not found"

# Check for client certificate requirement
echo -e "\n=== Checking for client certificate requirements ==="
CLIENT_CERT_FILES=$(sudo find /etc/nginx -type f -exec grep -l "ssl_client_certificate\|ssl_verify_client" {} \; 2>/dev/null)
if [ -n "$CLIENT_CERT_FILES" ]; then
    echo "Found SSL client certificate requirements in these files:"
    echo "$CLIENT_CERT_FILES"
    for FILE in $CLIENT_CERT_FILES; do
        echo "Content of $FILE related to client certificates:"
        sudo grep -A 3 -B 3 "ssl_client\|ssl_verify" $FILE
    done
else
    echo "No SSL client certificate requirements found in configuration files."
fi

echo -e "\n=== Testing direct API access ==="
echo "Testing health endpoint:"
curl -s http://localhost:8080/health || echo "Health endpoint not accessible"

echo -e "\nTesting API status endpoint:"
curl -s http://localhost:8080/api/status || echo "API status endpoint not accessible"

echo -e "\n=== Testing direct frontend access ==="
curl -s http://localhost:3000 -o /dev/null -w "Status code: %{http_code}\n" || echo "Frontend not accessible"

echo -e "\n=== FIX OPTIONS ==="
echo "1. Restart Nginx"
echo "2. Restart Docker containers"
echo "3. Fix SSL configuration (remove client certificate requirement)"
echo "4. Exit without changes"
read -p "Choose an option (1-4): " choice

case $choice in
    1)
        echo "Restarting Nginx..."
        sudo systemctl restart nginx
        echo "Status after restart:"
        sudo systemctl status nginx | head -n 4
        ;;
    2)
        echo "Restarting Docker containers..."
        sudo docker restart astroshield-frontend-1 astroshield-astroshield-api-1
        echo "Containers after restart:"
        sleep 5
        sudo docker ps | grep -E "frontend|api"
        ;;
    3)
        echo "Fixing SSL configuration..."
        sudo cp /etc/nginx/conf.d/astroshield-ssl.conf /etc/nginx/conf.d/astroshield-ssl.conf.bak.$(date +%Y%m%d%H%M%S)
        sudo bash -c 'cat > /etc/nginx/conf.d/astroshield-ssl.conf << EOF
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
        echo "Testing new configuration..."
        sudo nginx -t
        if [ $? -eq 0 ]; then
            echo "Configuration valid, restarting Nginx..."
            sudo systemctl restart nginx
        else
            echo "Configuration not valid, restoring backup..."
            sudo cp "$(ls -t /etc/nginx/conf.d/astroshield-ssl.conf.bak.* | head -1)" /etc/nginx/conf.d/astroshield-ssl.conf
        fi
        ;;
    4)
        echo "Exiting without changes"
        ;;
    *)
        echo "Invalid option"
        ;;
esac

echo -e "\n=== Final status check ==="
echo "Services on relevant ports:"
sudo ss -tulpn | grep -E ':3000|:8080|:443|:80'
echo "Nginx status:"
sudo systemctl status nginx | head -n 3

echo "======================== DIAGNOSTIC COMPLETE ========================" 