#!/bin/bash

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Check if running from project root
if [ ! -d "frontend" ]; then
  echo -e "${RED}Error: Please run this script from the project root directory${NC}"
  echo "The directory 'frontend' was not found in the current location"
  exit 1
fi

echo -e "${GREEN}=== Setting up Canvas Testing Environment ===${NC}"

# Install necessary npm packages
echo -e "\n${YELLOW}Installing Canvas testing dependencies...${NC}"
cd frontend
npm install --save-dev \
  canvas \
  jest-canvas-mock \
  @testing-library/jest-dom \
  canvas-prebuilt

if [ $? -ne 0 ]; then
  echo -e "${RED}Failed to install dependencies. Please install them manually:${NC}"
  echo "npm install --save-dev canvas jest-canvas-mock @testing-library/jest-dom canvas-prebuilt"
  exit 1
fi

echo -e "${GREEN}✓ Dependencies installed successfully${NC}"

# Create a test runner script
echo -e "\n${YELLOW}Creating Canvas test runner...${NC}"
cat > frontend/run_canvas_tests.js << 'EOL'
#!/usr/bin/env node

const path = require('path');
const { execSync } = require('child_process');
const fs = require('fs');

// Check for setup file
const setupFilePath = path.join(__dirname, 'jest.canvas.setup.js');
if (!fs.existsSync(setupFilePath)) {
  console.error('Error: jest.canvas.setup.js is missing! Run setup_canvas_testing.sh again.');
  process.exit(1);
}

// Run Jest with canvas support
try {
  execSync(
    `npx jest --setupFilesAfterEnv=${setupFilePath} "$@"`,
    { stdio: 'inherit' }
  );
} catch (error) {
  // Jest will exit with non-zero code if tests fail
  process.exit(error.status);
}
EOL

chmod +x frontend/run_canvas_tests.js
echo -e "${GREEN}✓ Canvas test runner created${NC}"

# Update package.json to add a script for canvas testing
echo -e "\n${YELLOW}Updating package.json to add canvas testing script...${NC}"

if command_exists jq; then
  # Use jq if available (more reliable)
  jq '.scripts["test:canvas"] = "node run_canvas_tests.js"' package.json > package.json.tmp && mv package.json.tmp package.json
else
  # Fallback to sed (less reliable but more widely available)
  sed -i.bak '/\"scripts\": {/a \    \"test:canvas\": \"node run_canvas_tests.js\",' package.json && rm package.json.bak
fi

echo -e "${GREEN}✓ Added 'test:canvas' script to package.json${NC}"

# Final instructions
echo -e "\n${GREEN}=== Canvas Testing Setup Complete ===${NC}"
echo -e "You can now run canvas tests with: ${YELLOW}npm run test:canvas${NC}"
echo -e "Or for a specific test: ${YELLOW}npm run test:canvas -- -t \"your test name\"${NC}"
echo -e "\nHappy testing! 🎨"

cd ..