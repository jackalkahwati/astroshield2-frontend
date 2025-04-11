#!/bin/bash

echo "Creating deployment directory on the server..."
ssh astroshield "mkdir -p ~/astroshield"

echo "Uploading essential backend files..."
ssh astroshield "mkdir -p ~/astroshield/backend/app"
scp -r backend/app astroshield:~/astroshield/backend/
scp backend/requirements.minimal.txt astroshield:~/astroshield/backend/

echo "Uploading essential frontend files..."
ssh astroshield "mkdir -p ~/astroshield/frontend"
scp frontend/package.json astroshield:~/astroshield/frontend/
scp -r frontend/src astroshield:~/astroshield/frontend/
scp -r frontend/public astroshield:~/astroshield/frontend/
scp frontend/next.config.js astroshield:~/astroshield/frontend/ 2>/dev/null || echo "No next.config.js file"
scp frontend/tsconfig.json astroshield:~/astroshield/frontend/ 2>/dev/null || echo "No tsconfig.json file"
scp frontend/tailwind.config.js astroshield:~/astroshield/frontend/ 2>/dev/null || echo "No tailwind.config.js file"
scp frontend/postcss.config.js astroshield:~/astroshield/frontend/ 2>/dev/null || echo "No postcss.config.js file"

echo "Uploading deployment script..."
scp deploy-aws.sh astroshield:~/astroshield/

echo "Upload complete. Now run the deployment script on the EC2 instance."
echo "SSH into your instance with: ssh astroshield"
echo "Then run: cd ~/astroshield && ./deploy-aws.sh" 