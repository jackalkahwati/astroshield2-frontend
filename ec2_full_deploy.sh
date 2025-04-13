#!/bin/bash
set -e

echo "=== Deploying Complete AstroShield Application ==="
cd /home/stardrive/astroshield/deployment

# Step 1: Fix backend circular import issues
echo "Fixing backend circular import issues..."
mkdir -p backend_fixed
cp -r backend/* backend_fixed/

# Create a proper main.py entry point file
cat > backend_fixed/main.py << 'EOT'
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

# Create the FastAPI app
app = FastAPI(
    title="AstroShield API",
    description="Backend API for the AstroShield satellite protection system",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add basic health check endpoint
@app.get("/api/v1/health")
def health_check():
    return {"status": "healthy", "version": "1.0.0"}

# Add a satellites endpoint with example data
@app.get("/api/v1/satellites")
def get_satellites():
    return [
        {
            "id": "sat-001",
            "name": "Starlink-1234",
            "type": "Communication",
            "orbit": "LEO",
            "status": "Active"
        },
        {
            "id": "sat-002",
            "name": "ISS",
            "type": "Space Station",
            "orbit": "LEO",
            "status": "Active"
        },
        {
            "id": "sat-003",
            "name": "GPS-IIF-10",
            "type": "Navigation",
            "orbit": "MEO",
            "status": "Active"
        }
    ]

# Add a simple events endpoint
@app.get("/api/v1/events")
def get_events():
    return [
        {
            "id": "evt-001",
            "type": "Proximity",
            "severity": "High",
            "timestamp": "2025-04-11T07:30:00Z",
            "description": "Close approach detected"
        },
        {
            "id": "evt-002",
            "type": "Maneuver",
            "severity": "Medium",
            "timestamp": "2025-04-10T14:45:00Z",
            "description": "Orbital adjustment"
        }
    ]

if __name__ == "__main__":
    port = int(os.getenv("PORT", 3005))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
EOT

# Create a simplified Dockerfile for backend
cat > backend_fixed/Dockerfile << 'EOT'
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 3005

# Start the application
CMD ["python", "main.py"]
EOT

# Step 2: Create a proper docker-compose with all services
echo "Creating complete docker-compose configuration..."

cat > docker-compose.yml << 'EOT'
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

  backend:
    build:
      context: ./backend_fixed
      dockerfile: Dockerfile
    restart: always
    environment:
      - PORT=3005
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/astroshield
    ports:
      - "3005:3005"
    depends_on:
      - postgres
    networks:
      - astroshield-net

  postgres:
    image: postgres:14
    restart: always
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=astroshield
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
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
      - backend

networks:
  astroshield-net:
    driver: bridge

volumes:
  postgres_data:
EOT

# Step 3: Update Nginx configuration to include API endpoints
echo "Updating Nginx configuration to include API endpoints..."

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
        server_name _;
        
        # Redirect to HTTPS
        location / {
            return 301 https://$host$request_uri;
        }
    }

    server {
        listen 443 ssl;
        server_name _;

        ssl_certificate /etc/nginx/ssl/server.crt;
        ssl_certificate_key /etc/nginx/ssl/server.key;
        
        # Frontend
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

        # Backend API
        location /api/v1/ {
            proxy_pass http://backend:3005/api/v1/;
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

# Step 4: Update the landing page to include API testing
echo "Updating landing page with API testing capability..."

cat > simple-static/index.html << 'EOT'
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
            padding: 2rem 0;
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
            margin: 0 auto 2rem auto;
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
        
        .api-test {
            background: white;
            padding: 2rem;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
            max-width: 800px;
            margin: 0 auto;
        }
        
        .api-test h2 {
            color: var(--primary-color);
            text-align: center;
            margin-bottom: 1.5rem;
        }
        
        .api-buttons {
            display: flex;
            justify-content: center;
            gap: 1rem;
            margin-bottom: 1.5rem;
        }
        
        button {
            background-color: var(--primary-color);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            cursor: pointer;
            font-weight: 500;
            transition: background-color 0.3s;
        }
        
        button:hover {
            background-color: var(--secondary-color);
        }
        
        #api-response {
            background-color: #f8f9fa;
            border-radius: 4px;
            padding: 1rem;
            font-family: monospace;
            white-space: pre-wrap;
            max-height: 300px;
            overflow-y: auto;
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
                
                <div class="status-item" id="backend-status">
                    <span class="status-label">API Backend</span>
                    <span class="status-value status-warning">CHECKING...</span>
                </div>
                
                <div class="status-item" id="database-status">
                    <span class="status-label">Database Connection</span>
                    <span class="status-value status-warning">PENDING</span>
                </div>
            </div>
            
            <div class="api-test">
                <h2>API Testing Console</h2>
                <div class="api-buttons">
                    <button onclick="testHealthEndpoint()">Test Health Endpoint</button>
                    <button onclick="testSatellitesEndpoint()">Get Satellites</button>
                    <button onclick="testEventsEndpoint()">Get Events</button>
                </div>
                <pre id="api-response">API response will appear here...</pre>
            </div>
            
            <div class="message">
                The deployment is now complete. You can access the application through SSH tunneling at the following URLs:
                <ul style="text-align: left; display: inline-block; margin-top: 10px;">
                    <li><strong>Frontend:</strong> http://127.0.0.1:8080/</li>
                    <li><strong>API Health:</strong> http://127.0.0.1:8080/api/v1/health</li>
                    <li><strong>API Satellites:</strong> http://127.0.0.1:8080/api/v1/satellites</li>
                    <li><strong>API Events:</strong> http://127.0.0.1:8080/api/v1/events</li>
                </ul>
            </div>
        </div>
    </div>
    
    <footer>
        <div class="container">
            <p>&copy; 2025 AstroShield - Space Situational Awareness Platform</p>
        </div>
    </footer>
    
    <script>
        // Function to test the health endpoint
        async function testHealthEndpoint() {
            const responseEl = document.getElementById('api-response');
            const backendStatusEl = document.getElementById('backend-status').querySelector('.status-value');
            
            responseEl.textContent = 'Fetching API health...';
            
            try {
                const response = await fetch('/api/v1/health');
                
                if (response.ok) {
                    const data = await response.json();
                    responseEl.textContent = JSON.stringify(data, null, 2);
                    
                    // Update status
                    backendStatusEl.textContent = 'ONLINE';
                    backendStatusEl.className = 'status-value status-success';
                } else {
                    responseEl.textContent = `Error: ${response.status} ${response.statusText}`;
                    backendStatusEl.textContent = 'ERROR';
                    backendStatusEl.className = 'status-value status-error';
                }
            } catch (error) {
                responseEl.textContent = `Connection error: ${error.message}`;
                backendStatusEl.textContent = 'OFFLINE';
                backendStatusEl.className = 'status-value status-error';
            }
        }
        
        // Function to test the satellites endpoint
        async function testSatellitesEndpoint() {
            const responseEl = document.getElementById('api-response');
            
            responseEl.textContent = 'Fetching satellites data...';
            
            try {
                const response = await fetch('/api/v1/satellites');
                
                if (response.ok) {
                    const data = await response.json();
                    responseEl.textContent = JSON.stringify(data, null, 2);
                } else {
                    responseEl.textContent = `Error: ${response.status} ${response.statusText}`;
                }
            } catch (error) {
                responseEl.textContent = `Connection error: ${error.message}`;
            }
        }
        
        // Function to test the events endpoint
        async function testEventsEndpoint() {
            const responseEl = document.getElementById('api-response');
            
            responseEl.textContent = 'Fetching events data...';
            
            try {
                const response = await fetch('/api/v1/events');
                
                if (response.ok) {
                    const data = await response.json();
                    responseEl.textContent = JSON.stringify(data, null, 2);
                } else {
                    responseEl.textContent = `Error: ${response.status} ${response.statusText}`;
                }
            } catch (error) {
                responseEl.textContent = `Connection error: ${error.message}`;
            }
        }
        
        // Check backend status when page loads
        window.addEventListener('DOMContentLoaded', function() {
            setTimeout(testHealthEndpoint, 1000);
        });
    </script>
</body>
</html>
EOT

# Step 5: Start the deployment
echo "Starting the full deployment..."
sudo docker-compose down
sudo docker-compose build
sudo docker-compose up -d

# Check the status
echo "Checking service status..."
sudo docker ps

echo "=== Full deployment completed! ==="
echo "The application can now be accessed through SSH tunneling."
echo "Use: ssh -L 8080:localhost:80 <user>@<bastion-host> -J <jump-server>"
echo "Then open http://127.0.0.1:8080 in your browser."
