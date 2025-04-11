#!/bin/bash

# Exit on error
set -e

# Load configuration
source deploy-config.sh

echo "========================================="
echo "AstroShield Deployment Script"
echo "========================================="
echo "Deploying to: $EC2_HOST"
echo "Using SSH key: $SSH_KEY_PATH"
echo "========================================="

# Verify SSH key exists
if [ ! -f "${SSH_KEY_PATH/#\~/$HOME}" ]; then
  echo "Error: SSH key not found at $SSH_KEY_PATH"
  echo "Please update SSH_KEY_PATH in deploy-config.sh to point to a valid SSH key."
  exit 1
fi

# Copy the systemd service file
mkdir -p deploy
cp astroshield.service deploy/

# Prepare backend
echo "Preparing backend..."
# Get version dependencies and create requirements.txt
pip freeze > requirements.txt

# Create a deploy directory for backend
mkdir -p deploy/backend
cp -r minimal_server.py requirements.txt deploy/backend/

# Build frontend
echo "Building frontend..."
cd $FRONTEND_DIR
npm run build
cd ..

# Create a deploy directory for frontend
mkdir -p deploy/frontend
cp -r $FRONTEND_DIR/.next $FRONTEND_DIR/public $FRONTEND_DIR/package.json $FRONTEND_DIR/next.config.js deploy/frontend/

# Create necessary scripts
echo "Creating deployment scripts..."

# Create start script with configuration
cat > deploy/start.sh << EOF
#!/bin/bash
cd \$(dirname "\$0")

# Start backend
cd backend
source venv/bin/activate
nohup python3 minimal_server.py > backend.log 2>&1 &
echo \$! > backend.pid
cd ..

# Start frontend
cd frontend
PORT=${FRONTEND_PORT} nohup npm start > frontend.log 2>&1 &
echo \$! > frontend.pid
EOF

# Create stop script
cat > deploy/stop.sh << 'EOF'
#!/bin/bash
cd $(dirname "$0")

# Stop backend
if [ -f backend/backend.pid ]; then
  kill -9 $(cat backend/backend.pid) 2>/dev/null || true
  rm backend/backend.pid
fi

# Stop frontend
if [ -f frontend/frontend.pid ]; then
  kill -9 $(cat frontend/frontend.pid) 2>/dev/null || true
  rm frontend/frontend.pid
fi
EOF

# Make scripts executable
chmod +x deploy/start.sh deploy/stop.sh

# Create a setup script to run on the server
cat > deploy/setup.sh << EOF
#!/bin/bash
cd \$(dirname "\$0")

# Install Node.js if not already installed
if ! command -v node &> /dev/null; then
  echo "Installing Node.js..."
  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | bash
  source ~/.nvm/nvm.sh
  nvm install ${NODE_VERSION}
  nvm use ${NODE_VERSION}
fi

# Install Python dependencies if not already installed
if ! command -v python${PYTHON_VERSION} &> /dev/null; then
  echo "Installing Python..."
  sudo yum update -y
  sudo yum install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-pip
fi

# Install Nginx if not already installed
if ! command -v nginx &> /dev/null; then
  echo "Installing Nginx..."
  sudo amazon-linux-extras install -y nginx1
  sudo systemctl enable nginx
fi

# Install Certbot for SSL certificates if not already installed
if ! command -v certbot &> /dev/null; then
  echo "Installing Certbot..."
  sudo amazon-linux-extras install -y epel
  sudo yum install -y certbot python-certbot-nginx
fi

# Setup frontend
cd frontend
npm install
cd ..

# Setup backend
cd backend
python${PYTHON_VERSION} -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd ..

# Setup Nginx
sudo tee /etc/nginx/conf.d/astroshield.conf > /dev/null << 'NGINX_CONFIG'
server {
    listen 80;
    server_name ${EC2_HOST};
    
    location / {
        proxy_pass http://localhost:${FRONTEND_PORT};
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_cache_bypass \$http_upgrade;
    }
    
    location /api/v1 {
        proxy_pass http://localhost:${BACKEND_PORT};
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_cache_bypass \$http_upgrade;
    }
    
    location /ws {
        proxy_pass http://localhost:${BACKEND_PORT};
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_cache_bypass \$http_upgrade;
    }
}
NGINX_CONFIG

# Test Nginx configuration
sudo nginx -t

# Reload Nginx to apply changes
sudo systemctl reload nginx

# Setup SSL certificates with Let's Encrypt
echo "Setting up SSL certificates..."
sudo certbot --nginx -d ${EC2_HOST} --non-interactive --agree-tos --email admin@${EC2_HOST} --redirect

# Setup systemd service if running as root
if [ \$(id -u) -eq 0 ]; then
  echo "Setting up systemd service..."
  cp astroshield.service /etc/systemd/system/
  systemctl daemon-reload
  systemctl enable astroshield.service
else
  echo "Not running as root, skipping systemd setup"
  echo "To setup the systemd service manually, run:"
  echo "sudo cp astroshield.service /etc/systemd/system/"
  echo "sudo systemctl daemon-reload"
  echo "sudo systemctl enable astroshield.service"
fi
EOF

chmod +x deploy/setup.sh

# Also include the config file
cp deploy-config.sh deploy/

# Update the frontend API config to use the correct backend URL
echo "Updating frontend API configuration..."
mkdir -p deploy/frontend/lib
cat > deploy/frontend/lib/api-config.js << EOF
// API Configuration
export const API_CONFIG = {
  baseURL: process.env.NEXT_PUBLIC_API_URL || 'https://${EC2_HOST}:${BACKEND_PORT}/api/v1',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json'
  },
  withCredentials: true
}

// Rate limiting configuration
export const RATE_LIMIT_CONFIG = {
  maxRequests: 50,
  windowMs: 60000, // 1 minute
  retryAfter: 5000 // 5 seconds
}

// Security configuration
export const SECURITY_CONFIG = {
  headers: {
    'Content-Security-Policy': [
      "default-src 'self'",
      "script-src 'self'",
      "style-src 'self' 'unsafe-inline'",
      "img-src 'self' data: https:",
      "font-src 'self'",
      "connect-src 'self' https://${EC2_HOST}:${BACKEND_PORT} wss://${EC2_HOST}:${BACKEND_PORT}",
      "report-uri /api/csp-report"
    ].join('; '),
    'Strict-Transport-Security': 'max-age=63072000; includeSubDomains; preload',
    'X-Content-Type-Options': 'nosniff',
    'X-Frame-Options': 'DENY',
    'X-XSS-Protection': '1; mode=block',
    'Referrer-Policy': 'strict-origin-when-cross-origin',
    'Permissions-Policy': 'geolocation=(), microphone=(), camera=()'
  }
}

// Monitoring configuration
export const MONITORING_CONFIG = {
  maxMetricsHistory: 1000,
  metricsWindowMs: 300000, // 5 minutes
  errorThreshold: 0.1, // 10% error rate threshold
  circuitBreakerTimeout: 60000 // 1 minute timeout for circuit breaker
}

export const WS_CONFIG = {
  url: process.env.NEXT_PUBLIC_WS_URL || 'wss://${EC2_HOST}:${BACKEND_PORT}/ws',
  reconnectInterval: 1000,
  maxReconnectAttempts: 5,
}
EOF

# Compress the deploy directory
echo "Compressing application..."
tar -czf astroshield-deploy.tar.gz -C deploy .

# Upload to EC2
echo "Uploading to EC2..."
ssh -i "${SSH_KEY_PATH/#\~/$HOME}" -o StrictHostKeyChecking=no $EC2_USER@$EC2_HOST "mkdir -p $REMOTE_DIR"
scp -i "${SSH_KEY_PATH/#\~/$HOME}" -o StrictHostKeyChecking=no astroshield-deploy.tar.gz $EC2_USER@$EC2_HOST:$REMOTE_DIR/

# Deploy on EC2
echo "Deploying on EC2..."
ssh -i "${SSH_KEY_PATH/#\~/$HOME}" -o StrictHostKeyChecking=no $EC2_USER@$EC2_HOST << EOF
cd $REMOTE_DIR
tar -xzf astroshield-deploy.tar.gz
rm astroshield-deploy.tar.gz
./setup.sh
./stop.sh
./start.sh

# Setup systemd service if not already done
sudo cp astroshield.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable astroshield.service
sudo systemctl start astroshield.service

echo "AstroShield deployment completed!"
EOF

# Clean up
rm -rf deploy astroshield-deploy.tar.gz

echo "==========================================================="
echo "Deployment to AstroShield Production completed successfully!"
echo "==========================================================="
echo "Frontend is accessible at: https://${EC2_HOST}"
echo "Backend API is accessible at: https://${EC2_HOST}/api/v1"
echo "==========================================================="
echo "Remember to check the security groups to ensure ports 80 and 443 are open for web traffic"