#!/bin/bash

# Exit on error
set -e

echo "==========================================="
echo "AstroShield Web Deployment Package Creator"
echo "==========================================="
echo "Domain: astroshield.sdataplab.com"
echo "==========================================="

# Configuration
DOMAIN="astroshield.sdataplab.com"
BACKEND_PORT=3001
FRONTEND_PORT=3000
BACKEND_DIR="."
FRONTEND_DIR="./frontend"

# Create deploy directories
mkdir -p deploy/backend
mkdir -p deploy/frontend

# Prepare backend
echo "Preparing backend..."
# Get version dependencies and create requirements.txt
pip freeze > requirements.txt

# Copy backend files
cp -r minimal_server.py requirements.txt deploy/backend/

# Build frontend
echo "Building frontend..."
cd $FRONTEND_DIR
npm run build
cd ..

# Copy frontend files
cp -r $FRONTEND_DIR/.next $FRONTEND_DIR/public $FRONTEND_DIR/package.json $FRONTEND_DIR/next.config.js deploy/frontend/

# Create start scripts
echo "Creating deployment scripts..."

# Create start script
cat > deploy/start.sh << EOF
#!/bin/bash
cd \$(dirname "\$0")

# Start backend
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
nohup python3 minimal_server.py > backend.log 2>&1 &
echo \$! > backend.pid
cd ..

# Start frontend
cd frontend
npm install
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

# Create Nginx configuration
cat > deploy/nginx.conf << EOF
server {
    listen 80;
    server_name ${DOMAIN};
    
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
EOF

# Create setup instructions
cat > deploy/README.md << EOF
# AstroShield Web Deployment

## Setup Instructions

1. Upload this package to your server
2. Extract the package: \`tar -xzf astroshield-web-deploy.tar.gz\`
3. Install dependencies:
   \`\`\`bash
   # Install Node.js
   curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | bash
   source ~/.nvm/nvm.sh
   nvm install 16
   nvm use 16
   
   # Install Python 3
   sudo yum update -y
   sudo yum install -y python3 python3-pip
   
   # Install Nginx
   sudo amazon-linux-extras install -y nginx1
   sudo systemctl enable nginx
   \`\`\`

4. Configure Nginx:
   \`\`\`bash
   sudo cp nginx.conf /etc/nginx/conf.d/astroshield.conf
   sudo nginx -t
   sudo systemctl reload nginx
   \`\`\`

5. Setup SSL with Let's Encrypt:
   \`\`\`bash
   sudo amazon-linux-extras install -y epel
   sudo yum install -y certbot python-certbot-nginx
   sudo certbot --nginx -d ${DOMAIN} --non-interactive --agree-tos --email admin@${DOMAIN} --redirect
   \`\`\`

6. Start the application:
   \`\`\`bash
   ./start.sh
   \`\`\`

7. To stop the application:
   \`\`\`bash
   ./stop.sh
   \`\`\`

## Additional Configuration

- The backend runs on port ${BACKEND_PORT}
- The frontend runs on port ${FRONTEND_PORT}
- Logs are available in \`backend/backend.log\` and \`frontend/frontend.log\`
EOF

# Update the frontend API config to use the correct production domain
echo "Updating frontend API configuration..."
cat > deploy/frontend/lib/api-config.js << EOF
// API Configuration
export const API_CONFIG = {
  baseURL: process.env.NEXT_PUBLIC_API_URL || 'https://${DOMAIN}/api/v1',
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
      "connect-src 'self' https://${DOMAIN} wss://${DOMAIN}",
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
  url: process.env.NEXT_PUBLIC_WS_URL || 'wss://${DOMAIN}/ws',
  reconnectInterval: 1000,
  maxReconnectAttempts: 5,
}
EOF

# Compress the deploy directory
echo "Compressing application..."
tar -czf astroshield-web-deploy.tar.gz -C deploy .

# Clean up
rm -rf deploy

echo "==========================================================="
echo "Web deployment package created successfully!"
echo "==========================================================="
echo "Package: astroshield-web-deploy.tar.gz"
echo ""
echo "Upload this package to your server and follow the instructions"
echo "in the README.md file inside the package."
echo "===========================================================" 