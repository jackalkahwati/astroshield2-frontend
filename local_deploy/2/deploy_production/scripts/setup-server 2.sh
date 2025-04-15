#!/bin/bash
# Server setup script for AstroShield application in AWS Gov environment
# Installs all required dependencies and sets up the environment

set -e

echo "Setting up server environment for AstroShield..."

# Update the system - adapted for AWS Gov environment
echo "Updating system packages..."
if command -v apt-get &> /dev/null; then
    # Debian/Ubuntu
    sudo apt-get update
    sudo apt-get upgrade -y
    sudo apt-get install -y python3 python3-venv python3-pip nodejs npm nginx certbot
elif command -v yum &> /dev/null; then
    # Amazon Linux/CentOS/RHEL
    sudo yum update -y
    sudo yum install -y python3 python3-pip nodejs npm nginx
    
    # Install certbot from EPEL
    sudo yum install -y epel-release
    sudo yum install -y certbot python3-certbot-nginx
fi

# Set up the application directory and permissions
APP_DIR="/opt/astroshield"
sudo mkdir -p "$APP_DIR"
sudo chown -R $USER:$USER "$APP_DIR"

# Set up backend
echo "Setting up backend environment..."
cd "$APP_DIR/backend"
python3 -m venv venv || python3 -m virtualenv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
deactivate

# Set up systemd service for backend
echo "Creating systemd service for backend..."
cat > /tmp/astroshield-backend.service << EOL
[Unit]
Description=AstroShield Backend Service
After=network.target

[Service]
User=$USER
WorkingDirectory=$APP_DIR/backend
ExecStart=$APP_DIR/backend/venv/bin/python3 minimal_server.py
Restart=always
StandardOutput=journal
StandardError=journal
Environment="PYTHONPATH=$APP_DIR"
Environment="PRODUCTION=true"
Environment="HOST=0.0.0.0"
Environment="PORT=3001"

[Install]
WantedBy=multi-user.target
EOL

sudo mv /tmp/astroshield-backend.service /etc/systemd/system/

# Set up frontend
echo "Setting up frontend environment..."
cd "$APP_DIR/frontend"
# Check if npm is available, if not try to use a pre-installed version or nvm
if ! command -v npm &> /dev/null; then
    echo "npm not found in path, checking for nvm or pre-installed Node.js..."
    if [ -d "$HOME/.nvm" ]; then
        export NVM_DIR="$HOME/.nvm"
        [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
        nvm use stable || nvm install stable
    elif [ -d "/usr/local/lib/nodejs" ]; then
        export PATH="/usr/local/lib/nodejs/bin:$PATH"
    fi
fi

npm install || echo "Warning: npm install failed. May need to install Node.js manually."

# Set up systemd service for frontend
echo "Creating systemd service for frontend..."
cat > /tmp/astroshield-frontend.service << EOL
[Unit]
Description=AstroShield Frontend Service
After=network.target

[Service]
User=$USER
WorkingDirectory=$APP_DIR/frontend
ExecStart=$(which npm) start
Restart=always
StandardOutput=journal
StandardError=journal
Environment="NODE_ENV=production"
Environment="PORT=3000"
Environment="NEXT_PUBLIC_API_URL=https://astroshield.sdataplab.com/api/v1"
Environment="NEXT_PUBLIC_WS_URL=wss://astroshield.sdataplab.com/ws"

[Install]
WantedBy=multi-user.target
EOL

sudo mv /tmp/astroshield-frontend.service /etc/systemd/system/

# Enable services but don't start them yet
sudo systemctl daemon-reload
sudo systemctl enable astroshield-backend
sudo systemctl enable astroshield-frontend

echo "Server setup completed successfully."