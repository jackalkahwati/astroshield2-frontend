#!/bin/bash
# docker-start.sh - Start AstroShield using Docker containers

# ANSI color codes for better output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting AstroShield Docker containers...${NC}"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
  echo -e "${RED}Docker is not running. Please start Docker first.${NC}"
  exit 1
fi

# Check if docker-compose.simple.yml exists
if [ ! -f "docker-compose.simple.yml" ]; then
  echo -e "${RED}docker-compose.simple.yml not found.${NC}"
  exit 1
fi

# Stop any running containers first
if docker ps | grep -q asttroshield; then
  echo -e "${YELLOW}Stopping any running AstroShield containers...${NC}"
  ./docker-stop.sh > /dev/null 2>&1 || true
  sleep 2
fi

# Start the Docker containers
echo -e "${GREEN}Starting AstroShield containers using docker-compose.simple.yml...${NC}"
docker-compose -f docker-compose.simple.yml up -d

# Check if containers are running
if docker ps | grep -q asttroshield; then
  echo -e "${GREEN}AstroShield containers started successfully!${NC}"
  echo -e "${YELLOW}Services:${NC}"
  echo -e "  - Backend API: http://localhost:5002"
  echo -e "  - Frontend: http://localhost:3003"
  echo -e "  - Nginx: http://localhost:80"
  
  # Wait for backend to be ready
  echo -e "${YELLOW}Waiting for backend to be ready...${NC}"
  attempt=0
  max_attempts=30
  while [ $attempt -lt $max_attempts ]; do
    if curl -s http://localhost:5002/health > /dev/null; then
      echo -e "${GREEN}Backend is ready!${NC}"
      break
    fi
    attempt=$((attempt+1))
    echo -n "."
    sleep 2
  done
  
  if [ $attempt -eq $max_attempts ]; then
    echo -e "${RED}Backend did not become ready within the expected time.${NC}"
    echo -e "${YELLOW}Please check logs with: docker-compose -f docker-compose.simple.yml logs backend${NC}"
  fi
else
  echo -e "${RED}Failed to start AstroShield containers.${NC}"
  echo -e "${YELLOW}Check logs with: docker-compose -f docker-compose.simple.yml logs${NC}"
  exit 1
fi

echo -e "${GREEN}AstroShield platform is now running!${NC}"
echo -e "${YELLOW}To stop the containers, run: ./docker-stop.sh${NC}" 