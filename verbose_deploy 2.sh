#!/bin/bash
set -e

# Configuration
DEPLOYMENT_FOLDER="/home/stardrive/astroshield"

# Ensure key permissions are correct
chmod 600 "$HOME/.ssh/jackal_ec2_key"
chmod 600 "$HOME/.ssh/config"

# Clean up any previous deployment files
echo "Cleaning up previous deployment files..."
rm -rf deployment astroshield_deploy.tar.gz 2>/dev/null || true

# Step 1: Create the deployment package with verbose output
echo "=== Creating deployment package ==="

# Package frontend with limited files
echo "Packaging frontend (selected files only)..."
mkdir -p deployment/frontend
mkdir -p deployment/frontend/app
mkdir -p deployment/frontend/components
mkdir -p deployment/frontend/lib

# Copy only essential frontend directories with progress feedback
echo "  - Copying frontend/app..."
cp -r frontend/app deployment/frontend/
echo "  - Copying frontend/components..."
cp -r frontend/components deployment/frontend/
echo "  - Copying frontend/lib..."
cp -r frontend/lib deployment/frontend/
echo "  - Copying frontend configuration files..."
cp frontend/package.json deployment/frontend/
cp frontend/next.config.mjs deployment/frontend/
cp frontend/Dockerfile deployment/frontend/
cp frontend/tailwind.config.js deployment/frontend/ 2>/dev/null || true
cp frontend/postcss.config.js deployment/frontend/ 2>/dev/null || true

# Package backend with limited files
echo "Packaging backend (selected files only)..."
mkdir -p deployment/backend
mkdir -p deployment/backend/app

# Copy only essential backend directories with progress feedback
echo "  - Copying backend/app..."
cp -r backend/app deployment/backend/
echo "  - Copying backend configuration files..."
cp backend/requirements.txt deployment/backend/
cp backend/Dockerfile deployment/backend/
cp backend/.env deployment/backend/ 2>/dev/null || true

# Package configuration files
echo "Packaging configuration files..."
mkdir -p deployment/nginx
echo "  - Copying nginx configuration..."
cp -r nginx/* deployment/nginx/
echo "  - Copying docker-compose.yml..."
cp docker-compose.yml deployment/

# Create deployment archive with progress
echo "Creating deployment archive..."
tar -czf astroshield_deploy.tar.gz deployment
echo "  - Created archive: $(du -h astroshield_deploy.tar.gz | cut -f1)"

# Step 2: Transfer files to EC2 with progress
echo "=== Transferring files to EC2 ==="
# Use the SSH config for easier connectivity
scp -v astroshield_deploy.tar.gz astroshield:~/
echo "  - File transfer completed"

# Step 3: Set up and deploy on EC2
echo "=== Deploying on EC2 ==="
# Connect to EC2 through bastion and run deployment commands
ssh -v astroshield "
    # Stop any existing Docker containers using our ports
    echo 'Stopping any existing containers...'
    sudo docker stop \$(sudo docker ps -q) 2>/dev/null || true
    
    # Extract the deployment archive
    echo 'Extracting deployment archive...'
    mkdir -p $DEPLOYMENT_FOLDER
    tar -xzf ~/astroshield_deploy.tar.gz -C $DEPLOYMENT_FOLDER
    cd $DEPLOYMENT_FOLDER/deployment
    
    # Create a minimal Dockerfile for the frontend to avoid building on the server
    echo 'Creating simplified frontend Dockerfile...'
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
    echo 'Generating SSL certificate...'
    mkdir -p nginx/ssl
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout nginx/ssl/server.key -out nginx/ssl/server.crt \
        -subj '/CN=astroshield.sdataplab.com'
    
    # Start Docker Compose
    echo 'Starting Docker containers...'
    sudo docker-compose down || true
    sudo docker-compose up -d
    
    echo 'Deployment completed!'
    echo 'The application should now be accessible at https://astroshield.sdataplab.com/'
"

echo "Deployment completed successfully!" 