#!/bin/bash
set -e

# Master Deployment Testing Script for AstroShield
echo "======================================================"
echo "AstroShield AWS Deployment Validation Suite"
echo "Started at: $(date)"
echo "======================================================"
echo

echo "Step 1: Application Code Validation"
echo "----------------------------------------------------"
if ./app_validate.sh; then
  echo "✅ Application code validation passed"
else
  echo "❌ Application code validation failed"
  echo "Please fix the issues above before proceeding"
  exit 1
fi
echo

echo "Step 2: Connectivity and Environment Tests"
echo "----------------------------------------------------"
if ./test_deployment.sh; then
  echo "✅ Connectivity tests passed"
else
  echo "❌ Connectivity tests failed"
  echo "Please fix the connectivity issues before proceeding"
  exit 1
fi
echo

echo "Step 3: Testing EC2 Environment"
echo "----------------------------------------------------"
echo "Uploading and running EC2 validation script..."

# Configuration
DEPLOYMENT_FOLDER="/home/stardrive/astroshield"

# Upload the EC2 validation script
scp ec2_validate.sh astroshield:~/

# Run the validation script on EC2
ssh astroshield "chmod +x ~/ec2_validate.sh && ~/ec2_validate.sh"

if [ $? -eq 0 ]; then
  echo "✅ EC2 environment validation passed"
else
  echo "❌ EC2 environment validation had issues"
  echo "Please review the output above"
  read -p "Continue with deployment anyway? (y/n) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
  fi
fi
echo

echo "Step 4: Creating a Deployment Package"
echo "----------------------------------------------------"
# Clean up old deployment files
rm -rf deployment astroshield_deploy.tar.gz 2>/dev/null || true

# Package frontend as-is (don't rebuild)
echo "Packaging frontend..."
mkdir -p deployment/frontend
cp -r frontend/* deployment/frontend/
cp frontend/Dockerfile deployment/frontend/

# Package backend
echo "Packaging backend..."
mkdir -p deployment/backend
cp -r backend/* deployment/backend/
cp backend/Dockerfile deployment/backend/

# Package configuration files
echo "Packaging configuration files..."
mkdir -p deployment/nginx
cp -r nginx/* deployment/nginx/
cp docker-compose.yml deployment/

# Create deployment archive
echo "Creating deployment archive..."
tar -czf astroshield_deploy.tar.gz deployment

echo "✅ Deployment package created successfully: astroshield_deploy.tar.gz ($(du -h astroshield_deploy.tar.gz | cut -f1))"
echo

echo "======================================================"
echo "All validation tests have passed!"
echo "The system is ready for deployment."
echo "======================================================"
echo
read -p "Do you want to proceed with the deployment now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
  echo "Starting deployment process..."
  ./simplified_deploy.sh
else
  echo "Deployment deferred. Run ./simplified_deploy.sh when you're ready."
fi 