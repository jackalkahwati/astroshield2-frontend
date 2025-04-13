#!/bin/bash
set -e

echo "=== Creating Final Landing Page ==="
echo "This script will create a landing page using absolute paths"

# Create the script to run on the EC2 instance
cat > ec2_final_landing.sh << 'EOF'
#!/bin/bash
set -e

echo "=== Creating final landing page ==="

# Create directories for the landing page
cd /home/stardrive/astroshield/deployment
mkdir -p simple-static

# Create the landing page directly in the deployment directory
cat > /home/stardrive/astroshield/deployment/simple-static/index.html << 'EOT'
<!DOCTYPE html>
<html>
<head>
    <title>AstroShield Platform</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        :root {
            --primary-color: #0a2342;
            --secondary-color: #2c5282;
            --accent-color: #4fd1c5;
            --text-color: #2d3748;
            --light-bg: #f7fafc;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 0;
            color: var(--text-color);
            background: linear-gradient(135deg, #f6f9fc 0%, #edf2f7 100%);
            min-height: 100vh;
        }
        
        header {
            background-color: var(--primary-color);
            color: white;
            padding: 1rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .container {
            width: 90%;
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 1rem;
        }
        
        nav {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .logo {
            font-size: 1.5rem;
            font-weight: bold;
            display: flex;
            align-items: center;
        }
        
        .logo-icon {
            margin-right: 10px;
            color: var(--accent-color);
        }
        
        .hero {
            text-align: center;
            padding: 4rem 0;
            background: rgba(255, 255, 255, 0.8);
            margin: 2rem 0;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        }
        
        h1 {
            font-size: 2.5rem;
            margin-bottom: 1rem;
            color: var(--primary-color);
        }
        
        .subtitle {
            font-size: 1.25rem;
            color: var(--secondary-color);
            margin-bottom: 2rem;
        }
        
        .status-card {
            background: white;
            padding: 2rem;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
            max-width: 800px;
            margin: 0 auto;
        }
        
        .status-title {
            font-size: 1.5rem;
            color: var(--primary-color);
            margin-bottom: 1.5rem;
            text-align: center;
        }
        
        .status-item {
            display: flex;
            align-items: center;
            padding: 0.75rem 0;
            border-bottom: 1px solid #edf2f7;
        }
        
        .status-item:last-child {
            border-bottom: none;
        }
        
        .status-label {
            flex: 1;
            font-weight: 500;
        }
        
        .status-value {
            padding: 0.25rem 0.75rem;
            border-radius: 16px;
            font-size: 0.875rem;
            font-weight: 600;
        }
        
        .status-success {
            background-color: #c6f6d5;
            color: #2f855a;
        }
        
        .status-warning {
            background-color: #feebc8;
            color: #c05621;
        }
        
        .status-error {
            background-color: #fed7d7;
            color: #c53030;
        }
        
        .message {
            background-color: #ebf8ff;
            color: #2b6cb0;
            padding: 1rem;
            border-radius: 4px;
            margin: 2rem 0;
            text-align: center;
        }
        
        footer {
            text-align: center;
            padding: 2rem 0;
            margin-top: 2rem;
            color: #718096;
            font-size: 0.875rem;
        }
    </style>
</head>
<body>
    <header>
        <div class="container">
            <nav>
                <div class="logo">
                    <span class="logo-icon">🛰️</span>
                    AstroShield
                </div>
            </nav>
        </div>
    </header>
    
    <div class="container">
        <div class="hero">
            <h1>AstroShield Platform</h1>
            <p class="subtitle">Space Situational Awareness & Satellite Protection System</p>
            
            <div class="status-card">
                <h2 class="status-title">System Status</h2>
                
                <div class="status-item">
                    <span class="status-label">Frontend Server</span>
                    <span class="status-value status-success">ONLINE</span>
                </div>
                
                <div class="status-item">
                    <span class="status-label">HTTPS Encryption</span>
                    <span class="status-value status-success">ACTIVE</span>
                </div>
                
                <div class="status-item">
                    <span class="status-label">API Backend</span>
                    <span class="status-value status-warning">IN PROGRESS</span>
                </div>
                
                <div class="status-item">
                    <span class="status-label">Database Connection</span>
                    <span class="status-value status-warning">PENDING</span>
                </div>
            </div>
            
            <div class="message">
                The deployment is in progress. The frontend is accessible, and we're currently configuring the backend services.
            </div>
        </div>
    </div>
    
    <footer>
        <div class="container">
            <p>&copy; 2025 AstroShield - Space Situational Awareness Platform</p>
        </div>
    </footer>
</body>
</html>
EOT

# Update the docker-compose.yml file
cat > /home/stardrive/astroshield/deployment/docker-compose.yml << 'EOT'
version: '3'

services:
  frontend:
    image: nginx:alpine
    restart: always
    volumes:
      - ./simple-static:/usr/share/nginx/html
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

# Update nginx.conf to point to the frontend using port 80
cat > /home/stardrive/astroshield/deployment/nginx/nginx.conf << 'EOT'
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
        
        # Static frontend - using the correct port (80) for the container
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

# Start the containers
echo "Starting containers..."
cd /home/stardrive/astroshield/deployment
sudo docker-compose down
sudo docker-compose up -d

# Verify they're running
echo "Verifying containers:"
sudo docker ps

echo "=== Landing page deployed ==="
echo "The site should now be accessible at https://astroshield.sdataplab.com/"
EOF

# Transfer the script to EC2
echo "Transferring script to EC2..."
chmod +x ec2_final_landing.sh
scp ec2_final_landing.sh astroshield:~/

# Run the script on EC2
echo "Running script on EC2..."
ssh astroshield "chmod +x ~/ec2_final_landing.sh && ~/ec2_final_landing.sh"

echo "Final landing page has been created and deployed."
echo "The AstroShield site is now available at https://astroshield.sdataplab.com/" 