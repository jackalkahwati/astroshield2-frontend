#!/bin/bash

echo "Uploading files to astroshield instance..."

# Create a temporary directory for deployment files
mkdir -p deploy_temp

# Copy backend files
cp -r backend deploy_temp/
cp -r venv deploy_temp/ 2>/dev/null || echo "No virtual environment to copy"

# Copy frontend files
cp -r frontend deploy_temp/

# Copy deployment script
cp deploy-aws.sh deploy_temp/

# Upload to EC2 using the SSH config
scp -r deploy_temp/* astroshield:~/astroshield/

# Clean up
rm -rf deploy_temp

echo "Upload complete. Now run the deployment script on the EC2 instance."
echo "SSH into your instance with: ssh astroshield"
echo "Then run: cd ~/astroshield && ./deploy-aws.sh" 