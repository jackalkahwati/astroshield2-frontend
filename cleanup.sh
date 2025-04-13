#!/bin/bash
# AstroShield Cleanup Script
# Removes duplicate files and organizes project structure

# Color constants
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== AstroShield Project Cleanup ===${NC}"

# Find all files with '2' in their names (likely duplicates)
echo -e "${BLUE}Finding duplicate files...${NC}"
DUPLICATES=$(find . -type f -name "*2*" -not -path "*/node_modules/*" -not -path "*/.git/*" -not -path "*/venv/*" -not -path "*/.venv/*")

if [ -z "$DUPLICATES" ]; then
    echo -e "${GREEN}No duplicate files found.${NC}"
else
    echo -e "${YELLOW}Found the following potential duplicate files:${NC}"
    echo "$DUPLICATES"
    
    echo -e "${YELLOW}Do you want to remove these files? (y/n)${NC}"
    read -r confirm_remove
    
    if [[ "$confirm_remove" =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}Removing duplicate files...${NC}"
        # Create a temporary file with the list of duplicates
        echo "$DUPLICATES" > duplicate_files.txt
        
        # Loop through each file and remove it
        while read -r file; do
            if [ -f "$file" ]; then
                rm "$file"
                echo -e "${GREEN}Removed: $file${NC}"
            fi
        done < duplicate_files.txt
        
        # Clean up the temporary file
        rm duplicate_files.txt
    else
        echo -e "${YELLOW}Skipping removal of duplicate files.${NC}"
    fi
fi

# Clean up temporary and unnecessary files
echo -e "${BLUE}Cleaning up temporary and unnecessary files...${NC}"
TEMP_PATTERNS=(".DS_Store" "*.log" "*.pid" "*.pyc" "__pycache__" ".coverage" "coverage/" ".swc/")

for pattern in "${TEMP_PATTERNS[@]}"; do
    find . -name "$pattern" -not -path "*/node_modules/*" -not -path "*/.git/*" -not -path "*/venv/*" -not -path "*/.venv/*" -print0 | xargs -0 rm -rf 2>/dev/null
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Removed files matching pattern: $pattern${NC}"
    fi
done

# Organize project structure
echo -e "${BLUE}Organizing project structure...${NC}"

# Ensure critical directories exist
mkdir -p frontend/public
mkdir -p frontend/src/components
mkdir -p frontend/src/services
mkdir -p frontend/src/pages/api
mkdir -p backend/app/routers
mkdir -p backend/app/models
mkdir -p backend/app/services
mkdir -p nginx/ssl
mkdir -p docs/api
mkdir -p docs/user

echo -e "${GREEN}Project directory structure organized.${NC}"

# Run Git garbage collection to clean up unnecessary files
echo -e "${BLUE}Running Git garbage collection...${NC}"
git gc

echo -e "${GREEN}Project cleanup complete!${NC}" 