#!/bin/bash
set -e

echo "=== Creating simple static frontend ==="

cd /home/stardrive/astroshield/deployment

# Stop all containers
sudo docker-compose down

# Create a simple frontend container with a static HTML page
mkdir -p simple-frontend
cat > simple-frontend/Dockerfile << 'EOT'
FROM nginx:alpine
COPY index.html /usr/share/nginx/html/index.html
EOT

cat > simple-frontend/index.html << 'EOT'
<!DOCTYPE html>
<html>
<head>
    <title>AstroShield Test Page</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            background-color: #f5f5f5;
        }
        .container {
            text-align: center;
            padding: 40px;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            max-width: 800px;
        }
        h1 {
            color: #2c3e50;
        }
        p {
            color: #34495e;
            line-height: 1.6;
        }
        .status {
            margin-top: 20px;
            padding: 15px;
            background-color: #e8f5e9;
            border-radius: 4px;
        }
        .logo {
            margin-bottom: 20px;
            max-width: 150px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>AstroShield Deployment Test</h1>
        <div class="status">
            <h2>Deployment Status</h2>
            <p>✅ Nginx Server: Running</p>
            <p>❌ Backend API: Configuration in progress</p>
            <p>✅ Frontend Static Files: Serving</p>
        </div>
        <p>This test page confirms that the Nginx server is working properly.</p>
        <p>The full application deployment is in progress. The backend services are being configured to resolve dependency issues.</p>
    </div>
</body>
</html>
EOT

# Create a simplified docker-compose file
cat > docker-compose.yml << 'EOT'
version: '3'

services:
  frontend:
    build:
      context: ./simple-frontend
      dockerfile: Dockerfile
    restart: always
    ports:
      - "3010:80"
    networks:
      - astroshield-net

  nginx:
    image: nginx:latest
    restart: always
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
    networks:
      - astroshield-net
    depends_on:
      - frontend

networks:
  astroshield-net:
    driver: bridge
EOT

# Update nginx.conf to point to the simple frontend
cat > nginx/nginx.conf << 'EOT'
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log;
pid /run/nginx.pid;

events {
    worker_connections 1024;
}

http {
    log_format  main  '$remote_addr - $remote_user [$time_local] "$request" '
                      '$status $body_bytes_sent "$http_referer" '
                      '"$http_user_agent" "$http_x_forwarded_for"';

    access_log  /var/log/nginx/access.log  main;

    sendfile            on;
    tcp_nopush          on;
    tcp_nodelay         on;
    keepalive_timeout   65;
    types_hash_max_size 2048;

    include             /etc/nginx/mime.types;
    default_type        application/octet-stream;

    server {
        listen 80;
        server_name astroshield.sdataplab.com;
        
        # Redirect to HTTPS
        location / {
            return 301 https://$host$request_uri;
        }
    }

    server {
        listen 443 ssl;
        server_name astroshield.sdataplab.com;

        ssl_certificate /etc/nginx/ssl/server.crt;
        ssl_certificate_key /etc/nginx/ssl/server.key;
        
        # Static frontend
        location / {
            proxy_pass http://frontend:80;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection 'upgrade';
            proxy_set_header Host $host;
            proxy_cache_bypass $http_upgrade;
            proxy_read_timeout 300s;
            proxy_connect_timeout 75s;
        }
    }
}
EOT

# Start the simplified deployment
echo "Starting simplified deployment..."
sudo docker-compose up -d

# Check running containers
echo "Running containers:"
sudo docker ps

echo "=== Simple static deployment complete ==="
echo "You should now be able to access https://astroshield.sdataplab.com/"
