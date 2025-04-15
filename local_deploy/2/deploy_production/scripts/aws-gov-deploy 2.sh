#!/bin/bash
# AstroShield Deployment Script for AWS Gov Cloud
# This script handles deployment to the specific AWS Gov Cloud environment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(dirname "$SCRIPT_DIR")"
SSH_HOST="astroshield"
DOMAIN="astroshield.sdataplab.com"
REMOTE_DIR="/home/stardrive/astroshield"

echo "Deploying AstroShield to AWS Gov Cloud ($DOMAIN)"
echo "---------------------------------------------------"

# Step 1: Create a tmp directory for our package
TMP_DIR=$(mktemp -d)
echo "Created temporary directory: $TMP_DIR"

# Step 2: Copy all necessary files to the tmp directory
echo "Gathering deployment files..."
mkdir -p "$TMP_DIR/backend"
mkdir -p "$TMP_DIR/frontend"
mkdir -p "$TMP_DIR/nginx"
mkdir -p "$TMP_DIR/scripts"

# Copy backend files
cp -r "$DEPLOY_DIR/backend"/* "$TMP_DIR/backend/"

# Copy frontend files
cp -r "$DEPLOY_DIR/frontend"/* "$TMP_DIR/frontend/"

# Copy nginx configuration
cp -r "$DEPLOY_DIR/nginx/conf.d" "$TMP_DIR/nginx/"

# Copy essential scripts
cp "$SCRIPT_DIR/setup-server.sh" "$TMP_DIR/scripts/"
cp "$SCRIPT_DIR/start-services.sh" "$TMP_DIR/scripts/"
cp "$SCRIPT_DIR/stop-services.sh" "$TMP_DIR/scripts/"
cp "$SCRIPT_DIR/setup-ssl.sh" "$TMP_DIR/scripts/"

# Step 3: Create deployment package
DEPLOY_PKG="astroshield-deploy.tar.gz"
echo "Creating deployment package: $DEPLOY_PKG"
tar -czf "$DEPLOY_PKG" -C "$TMP_DIR" .

# Step 4: Test SSH connection
echo "Testing SSH connection..."
if ! ssh -q "$SSH_HOST" exit; then
  echo "Error: Cannot connect to server via SSH. Check SSH configuration."
  exit 1
fi

# Step 5: Create directory on remote server
echo "Setting up remote directory structure..."
ssh "$SSH_HOST" "mkdir -p $REMOTE_DIR/backend $REMOTE_DIR/frontend $REMOTE_DIR/nginx $REMOTE_DIR/scripts"

# Step 6: Copy deployment package
echo "Copying deployment package to server..."
scp "$DEPLOY_PKG" "$SSH_HOST:$REMOTE_DIR/"

# Step 7: Unpack and set up on remote server
echo "Unpacking and setting up on remote server..."
ssh "$SSH_HOST" "cd $REMOTE_DIR && \
  tar -xzf astroshield-deploy.tar.gz && \
  chmod +x scripts/*.sh && \
  echo 'Deployment package unpacked successfully.'"

echo "Running installation on remote server..."
ssh "$SSH_HOST" "cd $REMOTE_DIR && \
  echo 'Setting up backend...' && \
  python3 -m venv backend/venv && \
  source backend/venv/bin/activate && \
  pip install -r backend/requirements.txt && \
  deactivate && \
  echo 'Setting up frontend...' && \
  cd frontend && npm install && cd .. && \
  echo 'Installation complete.'"

# Step 8: Ask user about running the application
echo "---------------------------------------------------"
echo "Deployment completed. Would you like to start the application now? (y/n)"
read -r START_APP

if [[ "$START_APP" == "y" || "$START_APP" == "Y" ]]; then
  echo "Starting application..."
  ssh "$SSH_HOST" "cd $REMOTE_DIR && nohup python3 backend/minimal_server.py > backend.log 2>&1 &"
  ssh "$SSH_HOST" "cd $REMOTE_DIR/frontend && nohup npm start > frontend.log 2>&1 &"
  echo "Application started. Access at:"
  echo "Backend: http://$DOMAIN:3001"
  echo "Frontend: http://$DOMAIN:3000"
  echo "Check logs at $REMOTE_DIR/backend.log and $REMOTE_DIR/frontend.log"
else
  echo "Application not started. To start manually, SSH to the server and run:"
  echo "  cd $REMOTE_DIR && nohup python3 backend/minimal_server.py > backend.log 2>&1 &"
  echo "  cd $REMOTE_DIR/frontend && nohup npm start > frontend.log 2>&1 &"
fi

# Step 9: Cleanup
echo "Cleaning up local temporary files..."
rm -rf "$TMP_DIR" "$DEPLOY_PKG"

echo "---------------------------------------------------"
echo "Deployment process completed!"