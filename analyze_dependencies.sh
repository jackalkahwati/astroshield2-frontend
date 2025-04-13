#!/bin/bash
# AstroShield Dependency Analysis Script
# Analyzes dependencies for both frontend and backend

# Color constants
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== AstroShield Dependency Analysis ===${NC}"

# Check that npm and pip are available
if ! command -v npm &> /dev/null; then
    echo -e "${RED}Error: npm is not installed. Please install npm to continue.${NC}"
    exit 1
fi

if ! command -v pip &> /dev/null; then
    echo -e "${RED}Error: pip is not installed. Please install pip to continue.${NC}"
    exit 1
fi

# Analyze frontend dependencies
echo -e "${BLUE}Analyzing frontend dependencies...${NC}"
cd frontend || { echo -e "${RED}Error: frontend directory not found.${NC}"; exit 1; }

# Check for outdated npm packages
echo -e "${BLUE}Checking for outdated npm packages...${NC}"
npm outdated

# Audit npm packages for security vulnerabilities
echo -e "${BLUE}Auditing npm packages for security vulnerabilities...${NC}"
npm audit

# Go back to project root
cd ..

# Analyze backend dependencies
echo -e "${BLUE}Analyzing backend dependencies...${NC}"
cd backend || { echo -e "${RED}Error: backend directory not found.${NC}"; exit 1; }

# Check for outdated Python packages
echo -e "${BLUE}Checking for outdated Python packages...${NC}"
pip list --outdated

# Check for security vulnerabilities (requires safety)
echo -e "${BLUE}Checking for security vulnerabilities in Python packages...${NC}"
if command -v safety &> /dev/null; then
    safety check -r requirements.txt
else
    echo -e "${YELLOW}Warning: 'safety' not installed. Install it with 'pip install safety' to check for vulnerabilities.${NC}"
    pip install safety
    safety check -r requirements.txt
fi

# Go back to project root
cd ..

# Suggest dependency optimizations
echo -e "${BLUE}Generating dependency optimization suggestions...${NC}"

echo -e "${YELLOW}Suggestions for frontend dependencies:${NC}"
echo "1. Consider removing unused dependencies by running 'npm prune'"
echo "2. Check for duplicate dependencies with 'npm dedupe'"
echo "3. Consider using exact versions (remove ^ and ~) for production builds"
echo "4. Update all dependencies to latest versions with 'npm update'"

echo -e "${YELLOW}Suggestions for backend dependencies:${NC}"
echo "1. Consider using a virtual environment for development"
echo "2. Pin exact versions for all dependencies in requirements.txt"
echo "3. Consider organizing requirements into dev, test, and production files"
echo "4. Uncomment psycopg2-binary in requirements.txt for PostgreSQL support"

echo -e "${GREEN}Dependency analysis complete!${NC}" 