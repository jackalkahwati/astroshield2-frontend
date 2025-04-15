#!/bin/bash
set -e

# Configuration
EC2_USER="ec2-user"
EC2_HOST="astroshield.sdataplab.com"
SSH_KEY_PATH="$PWD/~/.ssh/ec2_key_new"
BASTION_HOST="34.238.65.173"
BASTION_USER="ec2-user"
BASTION_KEY_PATH="$PWD/~/.ssh/ec2_key_new"
DEPLOYMENT_FOLDER="/home/ec2-user/astroshield"

# Check if SSH key exists
if [ ! -f "$SSH_KEY_PATH" ]; then
  echo "SSH key not found at $SSH_KEY_PATH"
  echo "Please provide the path to your SSH key:"
  read -r SSH_KEY_PATH
fi

# Ensure key permissions are correct
chmod 600 "$SSH_KEY_PATH"
chmod 600 "$BASTION_KEY_PATH"

echo "=== Creating deployment packages ==="

# Create frontend build
echo "Building frontend..."
cd frontend
npm run build
cd ..

# Package frontend
echo "Packaging frontend..."
mkdir -p deployment/frontend
cp -r frontend/* deployment/frontend/
cp frontend/Dockerfile deployment/frontend/

# Package backend
echo "Packaging backend..."
mkdir -p deployment/backend
cp -r backend/* deployment/backend/
cp backend/Dockerfile deployment/backend/

# Package configuration files
echo "Packaging configuration files..."
mkdir -p deployment/nginx
cp -r nginx/* deployment/nginx/
cp docker-compose.yml deployment/

# Create deployment archive
echo "Creating deployment archive..."
tar -czf astroshield_deploy.tar.gz deployment

echo "=== Transferring files to EC2 ==="
# Use ProxyCommand to connect through the bastion host
scp -o "ProxyCommand ssh -i $BASTION_KEY_PATH $BASTION_USER@$BASTION_HOST -W %h:%p" \
    -i "$SSH_KEY_PATH" \
    astroshield_deploy.tar.gz \
    "$EC2_USER@$EC2_HOST:/tmp/"

echo "=== Deploying on EC2 ==="
# Connect to EC2 through bastion and run deployment commands
ssh -o "ProxyCommand ssh -i $BASTION_KEY_PATH $BASTION_USER@$BASTION_HOST -W %h:%p" \
    -i "$SSH_KEY_PATH" \
    "$EC2_USER@$EC2_HOST" "
    mkdir -p $DEPLOYMENT_FOLDER
    tar -xzf /tmp/astroshield_deploy.tar.gz -C $DEPLOYMENT_FOLDER
    cd $DEPLOYMENT_FOLDER/deployment
    
    # Generate SSL certificate (self-signed for testing)
    mkdir -p nginx/ssl
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout nginx/ssl/server.key -out nginx/ssl/server.crt \
        -subj '/CN=astroshield.sdataplab.com'
    
    # Start Docker Compose
    sudo docker-compose down || true
    sudo docker-compose up -d
    
    echo 'Deployment completed!'
"

echo "Deployment completed successfully!" 