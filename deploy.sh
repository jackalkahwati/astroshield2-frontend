#!/bin/bash
# AstroShield Production Deployment Script

# Color constants
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== AstroShield Production Deployment ===${NC}"

# Check if we're in the right directory
if [ ! -f "docker-compose.yml" ]; then
    echo -e "${RED}Error: This script must be run from the root directory of the project${NC}"
    exit 1
fi

# Verify environment file exists
if [ ! -f ".env.production" ]; then
    echo -e "${RED}Error: .env.production file not found. Please create it first.${NC}"
    echo -e "${YELLOW}You can use the provided template by running: cp .env.production.template .env.production${NC}"
    exit 1
fi

# Check if Docker is installed
if ! command -v docker >/dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose >/dev/null 2>&1; then
    echo -e "${RED}Error: Docker Compose is not installed. Please install Docker Compose first.${NC}"
    exit 1
fi

# Functions
function check_ssl_files() {
    echo -e "${BLUE}Checking SSL certificates...${NC}"
    if [ ! -f "nginx/ssl/server.crt" ] || [ ! -f "nginx/ssl/server.key" ]; then
        echo -e "${YELLOW}Warning: SSL certificate files not found.${NC}"
        echo -e "${YELLOW}Creating self-signed certificates for testing...${NC}"
        
        # Create directory if it doesn't exist
        mkdir -p nginx/ssl
        
        # Generate self-signed certificate
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout nginx/ssl/server.key -out nginx/ssl/server.crt \
            -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost" \
            -addext "subjectAltName = DNS:localhost,IP:127.0.0.1"
            
        echo -e "${YELLOW}Self-signed certificates created. Replace these with real certificates before going to production.${NC}"
    else
        echo -e "${GREEN}SSL certificates found.${NC}"
    fi
}

function load_env_file() {
    echo -e "${BLUE}Loading production environment variables...${NC}"
    # Check if environment contains any default placeholder values
    grep -q "replace_with_" .env.production
    if [ $? -eq 0 ]; then
        echo -e "${YELLOW}Warning: Your .env.production file appears to contain placeholder values.${NC}"
        echo -e "${YELLOW}Please update these values before deploying to production.${NC}"
        echo -e "${YELLOW}Do you want to continue anyway? (y/n)${NC}"
        read -r confirm
        if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
            echo -e "${RED}Deployment aborted.${NC}"
            exit 1
        fi
    fi
    
    # Load environment variables
    set -a
    source .env.production
    set +a
    echo -e "${GREEN}Environment variables loaded.${NC}"
}

function run_tests() {
    echo -e "${BLUE}Running tests...${NC}"
    
    # Frontend tests
    echo -e "${BLUE}Running frontend tests...${NC}"
    cd frontend
    npm test -- --watchAll=false
    if [ $? -ne 0 ]; then
        echo -e "${RED}Frontend tests failed. Fix the issues before deploying.${NC}"
        cd ..
        return 1
    fi
    cd ..
    
    # Backend tests
    echo -e "${BLUE}Running backend tests...${NC}"
    cd backend
    python -m pytest
    if [ $? -ne 0 ]; then
        echo -e "${RED}Backend tests failed. Fix the issues before deploying.${NC}"
        cd ..
        return 1
    fi
    cd ..
    
    echo -e "${GREEN}All tests passed.${NC}"
    return 0
}

function build_and_deploy() {
    echo -e "${BLUE}Building Docker images...${NC}"
    docker-compose -f docker-compose.yml build
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to build Docker images. See above errors.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Docker images built successfully.${NC}"
    
    echo -e "${BLUE}Starting services...${NC}"
    docker-compose -f docker-compose.yml up -d
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to start services. See above errors.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Services started successfully.${NC}"
}

function verify_deployment() {
    echo -e "${BLUE}Verifying deployment...${NC}"
    sleep 10 # Give services time to start
    
    # Check if containers are running
    echo -e "${BLUE}Checking containers...${NC}"
    docker-compose ps
    
    # Check backend health
    echo -e "${BLUE}Checking backend health...${NC}"
    curl -s http://localhost:3001/health
    if [ $? -ne 0 ]; then
        echo -e "${RED}Backend health check failed.${NC}"
    else
        echo -e "${GREEN}Backend is healthy.${NC}"
    fi
    
    # Check frontend health
    echo -e "${BLUE}Checking frontend health...${NC}"
    curl -s http://localhost:3000/api/health
    if [ $? -ne 0 ]; then
        echo -e "${RED}Frontend health check failed.${NC}"
    else
        echo -e "${GREEN}Frontend is healthy.${NC}"
    fi
    
    echo -e "${GREEN}Deployment verification complete.${NC}"
}

# Main deployment process
load_env_file
check_ssl_files

echo -e "${BLUE}Do you want to run tests before deploying? (y/n)${NC}"
read -r run_tests_confirm
if [[ "$run_tests_confirm" =~ ^[Yy]$ ]]; then
    run_tests
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}Do you want to continue with deployment despite test failures? (y/n)${NC}"
        read -r ignore_test_failures
        if [[ ! "$ignore_test_failures" =~ ^[Yy]$ ]]; then
            echo -e "${RED}Deployment aborted.${NC}"
            exit 1
        fi
    fi
fi

echo -e "${BLUE}Ready to deploy. This will build and start all services.${NC}"
echo -e "${BLUE}Do you want to continue? (y/n)${NC}"
read -r deploy_confirm
if [[ "$deploy_confirm" =~ ^[Yy]$ ]]; then
    build_and_deploy
    verify_deployment
    
    echo -e "${GREEN}Deployment complete!${NC}"
    echo -e "${GREEN}Your application should now be accessible at:${NC}"
    echo -e "${GREEN}Frontend: https://localhost${NC}"
    echo -e "${GREEN}Backend API: https://localhost/api/v1${NC}"
    echo -e "${GREEN}API Documentation: https://localhost/api/v1/docs${NC}"
else
    echo -e "${YELLOW}Deployment canceled.${NC}"
fi