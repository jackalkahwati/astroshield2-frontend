#!/bin/bash
set -e

# Application Code Validation Script for AstroShield
echo "==== Application Code Validation ===="
echo "Running validation at $(date)"
echo

# Validate Frontend
echo "Frontend Validation:"

# Check for required files
echo "Checking required frontend files..."
FRONTEND_REQUIRED_FILES=(
  "frontend/package.json"
  "frontend/next.config.mjs"
  "frontend/app/layout.tsx"
  "frontend/app/page.tsx"
)

for file in "${FRONTEND_REQUIRED_FILES[@]}"; do
  if [ -f "$file" ]; then
    echo "✅ $file exists"
  else
    echo "❌ $file missing"
  fi
done
echo

# Parse package.json for correct dependencies
echo "Checking package.json dependencies..."
if [ -f "frontend/package.json" ]; then
  # Check for required dependencies
  REQUIRED_DEPS=("react" "react-dom" "next")
  for dep in "${REQUIRED_DEPS[@]}"; do
    if grep -q "\"$dep\":" frontend/package.json; then
      echo "✅ $dep dependency found"
    else
      echo "❌ $dep dependency missing"
    fi
  done
else
  echo "❌ Cannot check dependencies, package.json missing"
fi
echo

# Validate Backend
echo "Backend Validation:"

# Check for required files
echo "Checking required backend files..."
BACKEND_REQUIRED_FILES=(
  "backend/requirements.txt"
  "backend/app/main.py"
)

for file in "${BACKEND_REQUIRED_FILES[@]}"; do
  if [ -f "$file" ]; then
    echo "✅ $file exists"
  else
    echo "❌ $file missing"
  fi
done
echo

# Check for required Python packages
echo "Checking requirements.txt..."
if [ -f "backend/requirements.txt" ]; then
  # Check for required dependencies
  REQUIRED_PKGS=("fastapi" "uvicorn" "sqlalchemy")
  for pkg in "${REQUIRED_PKGS[@]}"; do
    if grep -q -i "^$pkg" backend/requirements.txt; then
      echo "✅ $pkg package found"
    else
      echo "❌ $pkg package missing"
    fi
  done
else
  echo "❌ Cannot check Python dependencies, requirements.txt missing"
fi
echo

# Validate Database
echo "Database Setup Validation:"
if [ -f "docker-compose.yml" ]; then
  if grep -q "postgres" docker-compose.yml; then
    echo "✅ PostgreSQL service defined in docker-compose.yml"
  else
    echo "❌ PostgreSQL service not found in docker-compose.yml"
  fi
else
  echo "❌ docker-compose.yml not found"
fi
echo

# Validate API Configuration
echo "API Configuration Validation:"
if [ -f "frontend/lib/api-config.ts" ]; then
  echo "✅ API configuration file exists"
  
  # Extract the API URL from the config file
  API_URL=$(grep "baseURL" frontend/lib/api-config.ts | head -1)
  echo "API URL configuration: $API_URL"
  
  # Check if it's referring to the correct backend service
  if echo "$API_URL" | grep -q "localhost:3001" || echo "$API_URL" | grep -q "backend:3001"; then
    echo "✅ API URL configured correctly"
  else
    echo "❌ API URL may not be properly configured (should point to backend:3001 or localhost:3001)"
  fi
else
  echo "❌ API configuration file not found"
fi
echo

# Validate Docker Configuration
echo "Docker Configuration Validation:"
if [ -f "frontend/Dockerfile" ]; then
  echo "✅ Frontend Dockerfile exists"
else
  echo "❌ Frontend Dockerfile missing"
fi

if [ -f "backend/Dockerfile" ]; then
  echo "✅ Backend Dockerfile exists"
else
  echo "❌ Backend Dockerfile missing"
fi

if [ -f "docker-compose.yml" ]; then
  echo "✅ docker-compose.yml exists"
  
  # Check for required services
  REQUIRED_SERVICES=("frontend" "backend" "postgres" "nginx")
  for service in "${REQUIRED_SERVICES[@]}"; do
    if grep -q "^  $service:" docker-compose.yml; then
      echo "✅ $service service defined"
    else
      echo "❌ $service service missing"
    fi
  done
else
  echo "❌ docker-compose.yml missing"
fi
echo

echo "==== Validation Complete ====" 