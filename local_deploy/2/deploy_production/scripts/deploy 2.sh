#!/bin/bash
# AstroShield Production Deployment Script
# Deploys the application to astroshield.sdataplab.com via SSH jump host

set -e

# Configuration
DOMAIN="astroshield.sdataplab.com"
SSH_HOST="astroshield"  # Uses the SSH config with ProxyJump
SSH_USER="stardrive"  # Set in SSH config
TARGET_DIR="/opt/astroshield"
REMOTE_NGINX_DIR="/etc/nginx"

# Local paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$DEPLOY_DIR/backend"
FRONTEND_DIR="$DEPLOY_DIR/frontend"
NGINX_DIR="$DEPLOY_DIR/nginx"

echo "Preparing to deploy AstroShield to $DOMAIN..."

# Create a temporary directory for packaging
TMP_DIR=$(mktemp -d)
echo "Creating temporary directory: $TMP_DIR"

mkdir -p "$TMP_DIR/backend"
mkdir -p "$TMP_DIR/frontend"
mkdir -p "$TMP_DIR/nginx"
mkdir -p "$TMP_DIR/scripts"

echo "Packaging application components..."

# Copy backend files
cp -r "$BACKEND_DIR"/* "$TMP_DIR/backend/"

# Copy frontend files
cp -r "$FRONTEND_DIR"/* "$TMP_DIR/frontend/"

# Copy nginx configuration
cp -r "$NGINX_DIR"/* "$TMP_DIR/nginx/"

# Copy scripts
cp "$SCRIPT_DIR/setup-server.sh" "$TMP_DIR/scripts/"
cp "$SCRIPT_DIR/start-services.sh" "$TMP_DIR/scripts/"
cp "$SCRIPT_DIR/stop-services.sh" "$TMP_DIR/scripts/"
cp "$SCRIPT_DIR/setup-ssl.sh" "$TMP_DIR/scripts/"

# Create the deployment package
DEPLOY_PKG="astroshield-deploy-$(date +%Y%m%d-%H%M%S).tar.gz"
echo "Creating deployment package: $DEPLOY_PKG"
tar -czf "$DEPLOY_PKG" -C "$TMP_DIR" .

echo "Testing SSH connection to server..."
if ! ssh -q "$SSH_HOST" exit; then
  echo "Error: Cannot connect to server via SSH. Please check your SSH configuration."
  exit 1
fi

echo "Copying files to server..."
scp "$DEPLOY_PKG" "$SSH_HOST:/tmp/"

echo "Setting up the application on the server..."
ssh "$SSH_HOST" "sudo mkdir -p $TARGET_DIR && \
  sudo tar -xzf /tmp/$DEPLOY_PKG -C /tmp && \
  sudo cp -r /tmp/backend $TARGET_DIR/ && \
  sudo cp -r /tmp/frontend $TARGET_DIR/ && \
  sudo mkdir -p $REMOTE_NGINX_DIR/conf.d && \
  sudo cp -r /tmp/nginx/conf.d/* $REMOTE_NGINX_DIR/conf.d/ && \
  sudo mkdir -p $TARGET_DIR/scripts && \
  sudo cp /tmp/scripts/* $TARGET_DIR/scripts/ && \
  sudo chmod +x $TARGET_DIR/scripts/*.sh && \
  sudo $TARGET_DIR/scripts/setup-server.sh && \
  sudo $TARGET_DIR/scripts/setup-ssl.sh $DOMAIN && \
  sudo $TARGET_DIR/scripts/start-services.sh && \
  sudo systemctl restart nginx || echo 'Nginx restart failed - may need manual configuration' && \
  rm -rf /tmp/$DEPLOY_PKG /tmp/backend /tmp/frontend /tmp/nginx /tmp/scripts"

echo "Cleaning up..."
rm -rf "$TMP_DIR" "$DEPLOY_PKG"

echo "======================================"
echo "Deployment to $DOMAIN completed!"
echo "Frontend URL: https://$DOMAIN"
echo "Backend API: https://$DOMAIN/api"
echo "======================================"