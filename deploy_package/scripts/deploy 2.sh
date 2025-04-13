#\!/bin/bash

set -e  # Exit on error

echo "========================================="
echo "Starting AstroShield Deployment"
echo "========================================="

# Choose deployment method
echo "Select deployment method:"
echo "1) Docker Compose deployment"
echo "2) Minimal Python server deployment"
read -p "Enter choice [1-2]: " DEPLOY_CHOICE

if [ "$DEPLOY_CHOICE" = "1" ]; then
    echo "Starting Docker deployment..."
    
    # Install Docker if needed
    if \! command -v docker &> /dev/null; then
        echo "Installing Docker..."
        sudo yum update -y
        sudo amazon-linux-extras install docker -y
        sudo systemctl start docker
        sudo systemctl enable docker
        sudo usermod -a -G docker $(whoami)
        echo "You may need to log out and back in for Docker permissions to take effect"
    fi
    
    # Install Docker Compose if needed
    if \! command -v docker-compose &> /dev/null; then
        echo "Installing Docker Compose..."
        sudo curl -L "https://github.com/docker/compose/releases/download/v2.5.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose
    fi
    
    # Build and start containers
    echo "Building containers..."
    docker build -t astroshield-backend -f Dockerfile.backend .
    docker build -t astroshield-frontend -f frontend/Dockerfile.frontend frontend
    
    echo "Starting services with Docker Compose..."
    docker-compose up -d
    
    echo "Docker deployment completed\!"
    
elif [ "$DEPLOY_CHOICE" = "2" ]; then
    echo "Starting minimal Python server deployment..."
    
    # Install Python if needed
    if \! command -v python3 &> /dev/null; then
        echo "Installing Python 3..."
        sudo yum update -y
        sudo yum install -y python3 python3-pip
    fi
    
    # Setup backend
    echo "Setting up backend..."
    cd backend
    python3 -m venv venv || python3 -m virtualenv venv
    source venv/bin/activate
    pip install -r requirements.txt
    
    # Start the server
    echo "Starting the server..."
    nohup python3 minimal_server.py > server.log 2>&1 &
    echo $\! > server.pid
    echo "Server started with PID $(cat server.pid)"
    cd ..
    
    echo "Minimal deployment completed\!"
    
else
    echo "Invalid selection. Exiting."
    exit 1
fi

echo "========================================="
echo "AstroShield deployment completed\!"
echo "========================================="
echo "The API should be accessible at: http://localhost:3001"
echo "Configure your web server (Nginx) to expose this service"
