#!/bin/bash
set -e

# Configuration
DEPLOYMENT_FOLDER="/home/stardrive/astroshield"

# Ensure key permissions are correct
chmod 600 "$HOME/.ssh/jackal_ec2_key"
chmod 600 "$HOME/.ssh/config"

# Step 1: Create the deployment package without building
echo "=== Creating deployment package ==="

# Package frontend as-is (don't rebuild)
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

# Step 2: Transfer files to EC2
echo "=== Transferring files to EC2 ==="
# Use the SSH config for easier connectivity
scp astroshield_deploy.tar.gz astroshield:~/

# Step 3: Set up and deploy on EC2
echo "=== Deploying on EC2 ==="
# Connect to EC2 through bastion and run deployment commands
ssh astroshield "
    # Stop any existing Docker containers using our ports
    echo 'Stopping any existing containers...'
    sudo docker stop \$(sudo docker ps -q) 2>/dev/null || true
    
    # Extract the deployment archive
    mkdir -p $DEPLOYMENT_FOLDER
    tar -xzf ~/astroshield_deploy.tar.gz -C $DEPLOYMENT_FOLDER
    cd $DEPLOYMENT_FOLDER/deployment
    
    # Create a minimal Dockerfile for the frontend to avoid building on the server
    cat > frontend/Dockerfile <<EOF
FROM node:18-alpine

WORKDIR /app

COPY . .

# Install dependencies
RUN npm install --production

# Install serve for static file serving
RUN npm install -g serve

EXPOSE 3000

# Serve the app with a static file server instead of Next.js
CMD [\"serve\", \"-s\", \".\", \"-p\", \"3000\"]
EOF
    
    # Generate SSL certificate (self-signed for testing)
    mkdir -p nginx/ssl
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout nginx/ssl/server.key -out nginx/ssl/server.crt \
        -subj '/CN=astroshield.sdataplab.com'
    
    # Start Docker Compose
    sudo docker-compose down || true
    sudo docker-compose up -d
    
    echo 'Deployment completed!'
    echo 'The application should now be accessible at https://astroshield.sdataplab.com/'
"

echo "Deployment completed successfully!" 