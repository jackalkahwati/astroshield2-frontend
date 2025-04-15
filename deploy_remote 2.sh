#!/bin/bash

# Deploy AstroShield to remote server
SERVER="astroshield.sdataplab.com"
USER="ec2-user"
SSH_KEY="~/.ssh/id_rsa_astroshield"

echo "Deploying to $SERVER..."

if ssh -i "$SSH_KEY" -o ConnectTimeout=5 "$USER@$SERVER" "echo Connected"; then
    ssh -i "$SSH_KEY" "$USER@$SERVER" << 'EOF'
    mkdir -p ~/astroshield
    cd ~/astroshield
    
    # Check for Docker
    if command -v docker &> /dev/null; then
        echo "Using Docker deployment"
        tar -xzf ~/astroshield_docker_deploy.tar.gz -C .
        chmod +x deploy_docker.sh
        ./deploy_docker.sh
    else
        echo "Using standard deployment"
        tar -xzf ~/minimal-deploy.tar.gz -C .
        chmod +x deploy-ec2.sh
        ./deploy-ec2.sh
    fi
EOF
else
    echo "SSH connection failed. Please upload manually:"
    echo "1. Upload either astroshield_docker_deploy.tar.gz or minimal-deploy.tar.gz"
    echo "2. SSH to server and extract: tar -xzf [package].tar.gz -C ~/astroshield"
    echo "3. Run deployment script: cd ~/astroshield && ./deploy_docker.sh"
fi 