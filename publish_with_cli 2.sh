#!/bin/bash

# Script to publish AstroShield API to SwaggerHub using the SwaggerHub CLI
# This script assumes you have Node.js and npm installed

# Configuration
API_NAME="AstroShield"
API_VERSION="1.0.0"
SPEC_FILE="openapi.json"
VISIBILITY="public"

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Check if SwaggerHub CLI is installed
if ! command -v swaggerhub &> /dev/null; then
    echo -e "${YELLOW}SwaggerHub CLI not found. Installing...${NC}"
    npm install -g swaggerhub-cli
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to install SwaggerHub CLI. Please install it manually:${NC}"
        echo "npm install -g swaggerhub-cli"
        exit 1
    fi
    echo -e "${GREEN}SwaggerHub CLI installed successfully.${NC}"
fi

# Check if the OpenAPI spec file exists
if [ ! -f "$SPEC_FILE" ]; then
    echo -e "${RED}Error: File '$SPEC_FILE' not found.${NC}"
    
    # Try to find the file in the current directory
    ALTERNATIVE_SPEC=$(find . -name "openapi.json" -type f | head -n 1)
    
    if [ -n "$ALTERNATIVE_SPEC" ]; then
        echo -e "${YELLOW}Found alternative spec file: $ALTERNATIVE_SPEC${NC}"
        read -p "Use this file instead? (y/n): " USE_ALTERNATIVE
        
        if [[ $USE_ALTERNATIVE == "y" || $USE_ALTERNATIVE == "Y" ]]; then
            SPEC_FILE=$ALTERNATIVE_SPEC
        else
            echo -e "${RED}Please provide the path to your OpenAPI specification file.${NC}"
            exit 1
        fi
    else
        echo -e "${RED}No OpenAPI specification file found. Please create one first.${NC}"
        exit 1
    fi
fi

# Ask for the username
echo -e "${YELLOW}NOTE: Your SwaggerHub username must be a simple name, not an email address.${NC}"
echo -e "${YELLOW}For example: 'johnsmith' not 'john@example.com'${NC}"
read -p "Enter your SwaggerHub username: " USERNAME

if [ -z "$USERNAME" ]; then
    echo -e "${RED}Error: Username cannot be empty.${NC}"
    exit 1
fi

# Check if the username contains @ symbol (email format)
if [[ "$USERNAME" == *"@"* ]]; then
    echo -e "${RED}Error: Username cannot be an email address.${NC}"
    echo -e "${YELLOW}Please use your SwaggerHub username, not your email address.${NC}"
    echo -e "${YELLOW}You can find your username in your SwaggerHub profile or URL:${NC}"
    echo -e "${YELLOW}https://app.swaggerhub.com/apis/YOUR_USERNAME${NC}"
    exit 1
fi

# Check if the user has already configured the CLI
if ! swaggerhub api:list &> /dev/null; then
    echo -e "${YELLOW}SwaggerHub CLI needs to be configured.${NC}"
    echo "You will need to create an API key in your SwaggerHub account settings."
    echo "Go to https://app.swaggerhub.com/settings/apiKey to create one."
    echo ""
    
    # Configure SwaggerHub CLI
    swaggerhub configure
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to configure SwaggerHub CLI.${NC}"
        exit 1
    fi
    echo -e "${GREEN}SwaggerHub CLI configured successfully.${NC}"
fi

# Full API identifier
API_IDENTIFIER="$USERNAME/$API_NAME/$API_VERSION"

# Check if the API already exists
if swaggerhub api:get $API_IDENTIFIER &> /dev/null; then
    echo -e "${YELLOW}API '$API_IDENTIFIER' already exists.${NC}"
    read -p "Update it? (y/n): " UPDATE_API
    
    if [[ $UPDATE_API == "y" || $UPDATE_API == "Y" ]]; then
        echo -e "${YELLOW}Updating API '$API_IDENTIFIER'...${NC}"
        swaggerhub api:update $API_IDENTIFIER --file $SPEC_FILE
        
        if [ $? -ne 0 ]; then
            echo -e "${RED}Failed to update API.${NC}"
            exit 1
        fi
        echo -e "${GREEN}API updated successfully.${NC}"
    else
        echo -e "${YELLOW}Operation cancelled.${NC}"
        exit 0
    fi
else
    # Create a new API
    echo -e "${YELLOW}Creating new API '$API_IDENTIFIER'...${NC}"
    swaggerhub api:create $API_IDENTIFIER --file $SPEC_FILE --visibility $VISIBILITY
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to create API.${NC}"
        exit 1
    fi
    echo -e "${GREEN}API created successfully.${NC}"
fi

# Output success message with link
echo -e "${GREEN}Your API is now published!${NC}"
echo -e "You can view it at: ${YELLOW}https://app.swaggerhub.com/apis/$USERNAME/$API_NAME/$API_VERSION${NC}"
echo -e "To verify the publication, run: ${YELLOW}swaggerhub api:get $API_IDENTIFIER${NC}" 