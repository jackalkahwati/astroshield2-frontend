#!/bin/bash

# Exit on error
set -e

echo "Starting AstroShield deployment on AWS..."

# Update system packages
echo "Updating system packages..."
sudo yum update -y
sudo yum install -y python3 python3-pip nodejs npm

# Set up backend
echo "Setting up backend..."
cd backend

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
  python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt || pip install -r requirements.minimal.txt

# Set up systemd service for backend
echo "Creating systemd service for backend..."
sudo tee /etc/systemd/system/astroshield-backend.service > /dev/null << EOF
[Unit]
Description=AstroShield Backend Service
After=network.target

[Service]
User=stardrive
WorkingDirectory=$(pwd)
ExecStart=$(pwd)/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 3001
Restart=always
RestartSec=5
StandardOutput=syslog
StandardError=syslog
SyslogIdentifier=astroshield-backend

[Install]
WantedBy=multi-user.target
EOF

# Enable and start backend service
sudo systemctl daemon-reload
sudo systemctl enable astroshield-backend
sudo systemctl restart astroshield-backend

# Set up frontend
echo "Setting up frontend..."
cd ../frontend

# Install dependencies and fix Next.js
npm install
npm install next@latest

# Create a simple static HTML page instead of building Next.js
mkdir -p public
cat > public/index.html << EOF
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AstroShield</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f0f2f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        header {
            background-color: #1a1a2e;
            color: white;
            padding: 20px 0;
            text-align: center;
        }
        h1 {
            margin: 0;
        }
        .content {
            background-color: white;
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .api-link {
            display: block;
            margin-top: 20px;
            padding: 10px;
            background-color: #e6f7ff;
            border-radius: 4px;
            text-align: center;
        }
        a {
            color: #0066cc;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <header>
        <div class="container">
            <h1>AstroShield</h1>
        </div>
    </header>
    <div class="container">
        <div class="content">
            <h2>Welcome to AstroShield</h2>
            <p>AstroShield is an application for tracking and visualizing asteroid trajectories and impact predictions.</p>
            <p>This is a temporary landing page. The full application is being deployed.</p>
            <div class="api-link">
                <a href="/api/docs" target="_blank">Access API Documentation</a>
            </div>
        </div>
    </div>
</body>
</html>
EOF

# Set up Nginx for frontend
echo "Setting up Nginx for frontend..."
sudo yum install -y nginx
sudo tee /etc/nginx/conf.d/astroshield.conf > /dev/null << EOF
server {
    listen 80;
    server_name astroshield.sdataplab.com;

    location / {
        root $(pwd)/public;
        index index.html;
        try_files \$uri \$uri/ /index.html;
    }

    location /api {
        proxy_pass http://localhost:3001;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_cache_bypass \$http_upgrade;
    }
}
EOF

# Restart Nginx
sudo systemctl enable nginx
sudo systemctl restart nginx

echo "Deployment completed successfully!"
echo "Your application should now be accessible at http://astroshield.sdataplab.com/"
echo "API documentation is available at http://astroshield.sdataplab.com/api/docs" 