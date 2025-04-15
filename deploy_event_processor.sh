#!/bin/bash

# Script to deploy the Astroshield Event Processor

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Astroshield Event Processor Deployment ===${NC}"
echo "Starting deployment process..."

# Check if Docker is installed and running
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is not installed. Please install Docker and try again.${NC}"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo -e "${RED}Docker is not running. Please start Docker and try again.${NC}"
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Docker Compose is not installed. Please install Docker Compose and try again.${NC}"
    exit 1
fi

# Build the Docker images
echo -e "${YELLOW}Building Docker images...${NC}"
docker-compose -f docker-compose.event-processor.yml build

# Check if build was successful
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to build Docker images. Please check the logs above for errors.${NC}"
    exit 1
fi

# Start the containers
echo -e "${YELLOW}Starting containers...${NC}"
docker-compose -f docker-compose.event-processor.yml up -d

# Check if containers started successfully
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to start containers. Please check the logs above for errors.${NC}"
    exit 1
fi

# Wait for services to be ready
echo -e "${YELLOW}Waiting for services to be ready...${NC}"
sleep 10

# Check if Kafka is running
echo -e "${YELLOW}Checking Kafka service...${NC}"
docker-compose -f docker-compose.event-processor.yml ps kafka | grep "Up" > /dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}Kafka service is not running. Please check the logs for errors.${NC}"
    echo "Run 'docker-compose -f docker-compose.event-processor.yml logs kafka' for more information."
    exit 1
fi

# Check if event processor is running
echo -e "${YELLOW}Checking Event Processor service...${NC}"
docker-compose -f docker-compose.event-processor.yml ps event-processor | grep "Up" > /dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}Event Processor service is not running. Please check the logs for errors.${NC}"
    echo "Run 'docker-compose -f docker-compose.event-processor.yml logs event-processor' for more information."
    exit 1
fi

echo -e "${GREEN}=== Deployment Successful ===${NC}"
echo "Event Processor is now running."
echo ""
echo "Kafka UI: http://localhost:8080"
echo "Event Processor: http://localhost:5005"
echo ""
echo "To view logs:"
echo "docker-compose -f docker-compose.event-processor.yml logs -f"
echo ""
echo "To stop the services:"
echo "docker-compose -f docker-compose.event-processor.yml down"
echo ""

# Make the script executable
chmod +x deploy_event_processor.sh 