#!/bin/bash
# Test runner script for AstroShield

# Set up colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get absolute paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
FRONTEND_DIR="$SCRIPT_DIR/frontend"
BACKEND_DIR="$SCRIPT_DIR/backend_fixed"

echo -e "${YELLOW}AstroShield Test Runner${NC}"
echo "=================================="
echo "Working directory: $SCRIPT_DIR"
echo "Frontend directory: $FRONTEND_DIR"
echo "Backend directory: $BACKEND_DIR"

# Parse command line arguments
BACKEND_ONLY=false
FRONTEND_ONLY=false
UNIT_ONLY=false
INTEGRATION_ONLY=false

for arg in "$@"
do
    case $arg in
        --backend)
        BACKEND_ONLY=true
        shift
        ;;
        --frontend)
        FRONTEND_ONLY=true
        shift
        ;;
        --unit)
        UNIT_ONLY=true
        shift
        ;;
        --integration)
        INTEGRATION_ONLY=true
        shift
        ;;
    esac
done

# Check if directories exist
if [ ! -d "$BACKEND_DIR" ]; then
    echo -e "${RED}Backend directory not found: $BACKEND_DIR${NC}"
    exit 1
fi

if [ ! -d "$FRONTEND_DIR" ] && [ "$BACKEND_ONLY" = false ]; then
    echo -e "${RED}Frontend directory not found: $FRONTEND_DIR${NC}"
    exit 1
fi

# Install dependencies if needed
if [ "$BACKEND_ONLY" = false ] || [ "$FRONTEND_ONLY" = false ]; then
    echo -e "\n${YELLOW}Checking dependencies...${NC}"
    
    # Check for pytest
    if ! python3 -m pytest --version > /dev/null 2>&1; then
        echo -e "${YELLOW}Installing pytest and dependencies...${NC}"
        python3 -m pip install pytest pytest-cov fastapi sqlalchemy
    fi
    
    # Check for frontend dependencies
    if [ -d "$FRONTEND_DIR" ] && [ "$BACKEND_ONLY" = false ]; then
        if ! test -d "$FRONTEND_DIR/node_modules/@testing-library/react"; then
            echo -e "${YELLOW}Installing frontend test dependencies...${NC}"
            cd "$FRONTEND_DIR" && npm install --save-dev @testing-library/react @testing-library/jest-dom @testing-library/user-event jest-environment-jsdom next-router-mock && cd "$SCRIPT_DIR"
        fi
    fi
fi

# Run backend tests
if [ "$FRONTEND_ONLY" = false ]; then
    echo -e "\n${YELLOW}Running backend tests...${NC}"
    
    if [ "$UNIT_ONLY" = true ]; then
        echo -e "${YELLOW}Running backend unit tests only...${NC}"
        cd "$BACKEND_DIR" && python3 -m tests.test_runner unit
    elif [ "$INTEGRATION_ONLY" = true ]; then
        echo -e "${YELLOW}Running backend integration tests only...${NC}"
        cd "$BACKEND_DIR" && python3 -m tests.test_runner integration
    else
        echo -e "${YELLOW}Running all backend tests...${NC}"
        cd "$BACKEND_DIR" && python3 -m tests.test_runner
    fi
    
    BACKEND_EXIT_CODE=$?
    
    if [ $BACKEND_EXIT_CODE -eq 0 ]; then
        echo -e "\n${GREEN}Backend tests passed!${NC}"
    else
        echo -e "\n${RED}Backend tests failed with exit code $BACKEND_EXIT_CODE${NC}"
    fi
fi

# Run frontend tests
if [ "$BACKEND_ONLY" = false ]; then
    echo -e "\n${YELLOW}Running frontend tests...${NC}"
    
    cd "$FRONTEND_DIR" && npm test
    
    FRONTEND_EXIT_CODE=$?
    
    if [ $FRONTEND_EXIT_CODE -eq 0 ]; then
        echo -e "\n${GREEN}Frontend tests passed!${NC}"
    else
        echo -e "\n${RED}Frontend tests failed with exit code $FRONTEND_EXIT_CODE${NC}"
    fi
fi

# Return overall exit code
if [ "$FRONTEND_ONLY" = true ]; then
    exit $FRONTEND_EXIT_CODE
elif [ "$BACKEND_ONLY" = true ]; then
    exit $BACKEND_EXIT_CODE
else
    # If both were run, exit with failure if either failed
    if [ $BACKEND_EXIT_CODE -ne 0 ] || [ $FRONTEND_EXIT_CODE -ne 0 ]; then
        exit 1
    else
        exit 0
    fi
fi 