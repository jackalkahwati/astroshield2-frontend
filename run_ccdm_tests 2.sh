#!/bin/bash
# Script to run CCDM service tests

# Set up colored output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Running CCDM Service Tests ===${NC}"
echo -e "${YELLOW}Setting up test environment...${NC}"

# Check if pytest is installed
if ! command -v pytest &> /dev/null
then
    echo -e "${RED}pytest is not installed. Installing...${NC}"
    pip3 install pytest pytest-asyncio pytest-cov
fi

# Check if required packages are installed
echo -e "${YELLOW}Checking dependencies...${NC}"
if [ -f backend/requirements.txt ]; then
    pip3 install -r backend/requirements.txt 2>/dev/null
fi

echo -e "${GREEN}Running tests with coverage report...${NC}"
python3 -m pytest __tests__/services/test_ccdm_service.py -v --cov=backend.app.services.ccdm

echo -e "${BLUE}=== Tests Complete ===${NC}"