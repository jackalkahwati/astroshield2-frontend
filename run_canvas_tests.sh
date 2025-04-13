#!/bin/bash

# Define colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Running AstroShield Canvas Tests ===${NC}"

# Check if we're in the project root
if [ ! -d "frontend" ]; then
  echo -e "${RED}Error: Please run this script from the project root directory${NC}"
  echo "The directory 'frontend' was not found in the current location"
  exit 1
fi

# Navigate to frontend directory
cd frontend

# Check if setup is complete
if [ ! -f "jest.canvas.setup.js" ]; then
  echo -e "${YELLOW}Canvas test setup not found. Running setup first...${NC}"
  cd ..
  bash setup_canvas_testing.sh
  cd frontend
fi

echo -e "${YELLOW}Running canvas tests...${NC}"

# Get additional arguments
ARGS=""
if [ "$#" -gt 0 ]; then
  ARGS="-- $@"
fi

# Run the tests
npm run test:canvas $ARGS

# Check the exit code
if [ $? -eq 0 ]; then
  echo -e "\n${GREEN}✓ Canvas tests completed successfully${NC}"
else
  echo -e "\n${RED}✗ Some canvas tests failed${NC}"
fi 