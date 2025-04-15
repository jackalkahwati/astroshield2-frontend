#!/bin/bash

# Exit on error
set -e

echo "========================================="
echo "AstroShield EC2 Deployment Script"
echo "========================================="

# Configuration
APP_DIR="/home/ec2-user/astroshield"
BACKEND_PORT=3001
FRONTEND_PORT=3000
DOMAIN="astroshield.sdataplab.com"

# Create application directory
mkdir -p $APP_DIR
cd $APP_DIR

# Extract the deployment package if it exists as a tarball
if [ -f astroshield-deploy.tar.gz ]; then
  tar -xzf astroshield-deploy.tar.gz
fi

# Install system dependencies
echo "Installing system dependencies..."

# Update package list
sudo yum update -y

# Install Node.js
if ! command -v node &> /dev/null; then
  echo "Installing Node.js..."
  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | bash
  source ~/.nvm/nvm.sh
  nvm install 16
  nvm use 16
  nvm alias default 16
fi

# Install Python
if ! command -v python3 &> /dev/null; then
  echo "Installing Python..."
  sudo yum install -y python3 python3-pip
fi

# Install Nginx
if ! command -v nginx &> /dev/null; then
  echo "Installing Nginx..."
  sudo amazon-linux-extras install -y nginx1
  sudo systemctl enable nginx
fi

# Install Certbot for SSL certificates
if ! command -v certbot &> /dev/null; then
  echo "Installing Certbot..."
  sudo amazon-linux-extras install -y epel
  sudo yum install -y certbot python-certbot-nginx
fi

# Setup frontend
echo "Setting up frontend..."
cd frontend
npm install
cd ..

# Setup backend
echo "Setting up backend..."
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd ..

# Setup Nginx configuration
echo "Configuring Nginx..."
sudo bash -c "cat > /etc/nginx/conf.d/astroshield.conf << EOF
server {
    listen 80;
    server_name $DOMAIN;
    
    location / {
        proxy_pass http://localhost:$FRONTEND_PORT;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_cache_bypass \$http_upgrade;
    }
    
    location /api/v1 {
        proxy_pass http://localhost:$BACKEND_PORT;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_cache_bypass \$http_upgrade;
    }
    
    location /maneuvers {
        proxy_pass http://localhost:$BACKEND_PORT/maneuvers;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_cache_bypass \$http_upgrade;
    }
    
    location /satellites {
        proxy_pass http://localhost:$BACKEND_PORT/satellites;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_cache_bypass \$http_upgrade;
    }
}
EOF"

# Test Nginx configuration
sudo nginx -t

# Reload Nginx to apply changes
sudo systemctl restart nginx

# Setup SSL certificates with Let's Encrypt
echo "Setting up SSL certificates..."
sudo certbot --nginx -d $DOMAIN --non-interactive --agree-tos --email admin@sdataplab.com --redirect

# Create start script
echo "Creating start script..."
cat > start.sh << EOF
#!/bin/bash
cd \$(dirname "\$0")

# Start backend
cd backend
source venv/bin/activate
nohup python3 minimal_server.py > ../logs/backend.log 2>&1 &
echo \$! > ../logs/backend.pid
cd ..

# Start frontend
cd frontend
PORT=$FRONTEND_PORT nohup npm start > ../logs/frontend.log 2>&1 &
echo \$! > ../logs/frontend.pid
cd ..

echo "AstroShield started"
EOF

# Create stop script
echo "Creating stop script..."
cat > stop.sh << EOF
#!/bin/bash
cd \$(dirname "\$0")

# Stop backend
if [ -f logs/backend.pid ]; then
  kill -9 \$(cat logs/backend.pid) 2>/dev/null || true
  rm logs/backend.pid
fi

# Stop frontend
if [ -f logs/frontend.pid ]; then
  kill -9 \$(cat logs/frontend.pid) 2>/dev/null || true
  rm logs/frontend.pid
fi

echo "AstroShield stopped"
EOF

# Create systemd service file
echo "Creating systemd service file..."
sudo bash -c "cat > /etc/systemd/system/astroshield.service << EOF
[Unit]
Description=AstroShield Application
After=network.target

[Service]
Type=forking
User=ec2-user
WorkingDirectory=$APP_DIR
ExecStart=$APP_DIR/start.sh
ExecStop=$APP_DIR/stop.sh
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF"

# Create logs directory
mkdir -p logs

# Make scripts executable
chmod +x start.sh stop.sh

# Setup systemd service
echo "Setting up systemd service..."
sudo systemctl daemon-reload
sudo systemctl enable astroshield.service
sudo systemctl start astroshield.service

echo "========================================="
echo "AstroShield deployment completed!"
echo "========================================="
echo "Frontend is accessible at: https://$DOMAIN"
echo "Backend API is accessible at: https://$DOMAIN/api/v1"
echo "=========================================" 