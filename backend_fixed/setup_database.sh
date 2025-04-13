#!/bin/bash
# Database setup script for AstroShield
# This script creates the PostgreSQL database and tables, and loads sample data

# Set up colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}AstroShield Database Setup${NC}"
echo "=================================="

# Check if PostgreSQL is running
echo -e "\n${YELLOW}Checking PostgreSQL status...${NC}"
if ! pg_isready -h localhost -p 5432 > /dev/null 2>&1; then
    echo -e "${RED}PostgreSQL is not running. Starting Docker containers...${NC}"
    
    # Run Docker Compose if PostgreSQL is not running
    if command -v docker-compose &> /dev/null; then
        docker-compose up -d postgres
    elif command -v docker &> /dev/null && docker compose version &> /dev/null; then
        docker compose up -d postgres
    else
        echo -e "${RED}Error: Docker Compose not found. Please start PostgreSQL manually.${NC}"
        exit 1
    fi
    
    # Wait for PostgreSQL to start
    echo "Waiting for PostgreSQL to start..."
    sleep 5
    
    # Check again if PostgreSQL is running
    if ! pg_isready -h localhost -p 5432 > /dev/null 2>&1; then
        echo -e "${RED}Error: PostgreSQL failed to start. Please check Docker logs.${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}PostgreSQL is running.${NC}"

# Set environment variables
export DATABASE_URL="postgresql://postgres:password@localhost:5432/astroshield"

# Create database if it doesn't exist
echo -e "\n${YELLOW}Creating database if it doesn't exist...${NC}"
if ! psql -h localhost -U postgres -lqt | cut -d \| -f 1 | grep -qw astroshield; then
    echo "Creating database astroshield..."
    psql -h localhost -U postgres -c "CREATE DATABASE astroshield;" || {
        echo -e "${RED}Error: Failed to create database. Check PostgreSQL credentials.${NC}"
        exit 1
    }
    echo -e "${GREEN}Database created successfully.${NC}"
else
    echo -e "${GREEN}Database already exists.${NC}"
fi

# Initialize database tables and sample data
echo -e "\n${YELLOW}Initializing database tables and sample data...${NC}"
python scripts/init_database.py || {
    echo -e "${RED}Error: Failed to initialize database tables.${NC}"
    exit 1
}

echo -e "\n${GREEN}Database setup completed successfully!${NC}"
echo "You can now run the application with the following environment variable:"
echo "export DATABASE_URL=\"postgresql://postgres:password@localhost:5432/astroshield\""
echo -e "${YELLOW}Note:${NC} In production, use a more secure password and consider using environment variables." 