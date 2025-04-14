#!/bin/bash
# docker-stop.sh - Stop AstroShield Docker containers

# ANSI color codes for better output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Stopping AstroShield Docker containers...${NC}"

# Stop and remove containers from docker-compose
if [ -f "docker-compose.simple.yml" ]; then
  echo -e "${GREEN}Stopping containers from docker-compose.simple.yml...${NC}"
  docker-compose -f docker-compose.simple.yml down
elif [ -f "docker-compose.yml" ]; then
  echo -e "${GREEN}Stopping containers from docker-compose.yml...${NC}"
  docker-compose down
else
  echo -e "${RED}No docker-compose file found.${NC}"
  
  # Try to stop containers directly
  echo -e "${YELLOW}Attempting to stop containers by name...${NC}"
  
  # Stop backend container if running
  if docker ps | grep -q asttroshield.*backend; then
    echo -e "${GREEN}Stopping backend container...${NC}"
    docker stop $(docker ps -q --filter name=asttroshield.*backend)
  fi
  
  # Stop frontend container if running
  if docker ps | grep -q asttroshield.*frontend; then
    echo -e "${GREEN}Stopping frontend container...${NC}"
    docker stop $(docker ps -q --filter name=asttroshield.*frontend)
  fi
  
  # Stop nginx container if running
  if docker ps | grep -q asttroshield.*nginx; then
    echo -e "${GREEN}Stopping nginx container...${NC}"
    docker stop $(docker ps -q --filter name=asttroshield.*nginx)
  fi
fi

# Optional: Prune unused Docker resources
echo -e "${YELLOW}Do you want to remove unused Docker resources? (y/n)${NC}"
read -r answer
if [[ "$answer" =~ ^[Yy]$ ]]; then
  echo -e "${GREEN}Removing unused Docker resources...${NC}"
  docker system prune -f
fi

echo -e "${GREEN}Done! All AstroShield containers have been stopped.${NC}" 