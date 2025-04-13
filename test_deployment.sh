#!/bin/bash
set -e

# Configuration
SSH_KEY_PATH="$HOME/.ssh/jackal_ec2_key"

echo "=== AstroShield Deployment Test Script ==="

# Test 1: Verify SSH keys and config exist
echo "Testing SSH keys and config..."
if [ ! -f "$SSH_KEY_PATH" ]; then
  echo "❌ Error: SSH key not found at $SSH_KEY_PATH"
  exit 1
else
  chmod 600 "$SSH_KEY_PATH"
  echo "✅ SSH key found and permissions set"
fi

if [ ! -f "$HOME/.ssh/config" ]; then
  echo "❌ Error: SSH config not found at $HOME/.ssh/config"
  exit 1
else
  chmod 600 "$HOME/.ssh/config"
  echo "✅ SSH config found and permissions set"
fi

# Test 2: Check SSH connectivity to bastion host
echo "Testing SSH connectivity to bastion host..."
if ! ssh -o "BatchMode=yes" -o "ConnectTimeout=5" ub "echo 'Connected to bastion'"; then
  echo "❌ Error: Cannot connect to bastion host"
  exit 1
else
  echo "✅ Successfully connected to bastion host"
fi

# Test 3: Check SSH connectivity to EC2 instance via bastion
echo "Testing SSH connectivity to EC2 instance via bastion..."
if ! ssh -o "BatchMode=yes" -o "ConnectTimeout=5" astroshield "echo 'Connected to EC2'"; then
  echo "❌ Error: Cannot connect to EC2 instance via bastion"
  exit 1
else
  echo "✅ Successfully connected to EC2 instance via bastion"
fi

# Test 4: Check Docker and Docker Compose on EC2
echo "Testing Docker and Docker Compose on EC2..."
if ! ssh astroshield "docker --version && docker-compose --version"; then
  echo "❌ Error: Docker or Docker Compose not available on EC2"
  exit 1
else
  echo "✅ Docker and Docker Compose are available on EC2"
fi

# Test 5: Validate configuration files locally
echo "Validating configuration files..."

echo "Checking docker-compose.yml..."
if [ ! -f "docker-compose.yml" ]; then
  echo "❌ Error: docker-compose.yml not found"
  exit 1
else
  echo "✅ docker-compose.yml found"
fi

echo "Checking nginx configuration..."
if [ ! -d "nginx" ]; then
  echo "❌ Error: nginx directory not found"
  exit 1
else
  echo "✅ nginx directory found"
fi

# Test 6: Validate Dockerfiles
echo "Validating Dockerfiles..."

echo "Checking frontend Dockerfile..."
if [ ! -f "frontend/Dockerfile" ]; then
  echo "❌ Error: frontend/Dockerfile not found"
  exit 1
else
  echo "✅ frontend/Dockerfile found"
fi

echo "Checking backend Dockerfile..."
if [ ! -f "backend/Dockerfile" ]; then
  echo "❌ Error: backend/Dockerfile not found"
  exit 1
else
  echo "✅ backend/Dockerfile found"
fi

# Test 7: Check package.json and requirements.txt
echo "Checking package dependencies..."

echo "Checking frontend/package.json..."
if [ ! -f "frontend/package.json" ]; then
  echo "❌ Error: frontend/package.json not found"
  exit 1
else
  echo "✅ frontend/package.json found"
fi

echo "Checking backend/requirements.txt..."
if [ ! -f "backend/requirements.txt" ]; then
  echo "❌ Error: backend/requirements.txt not found"
  exit 1
else
  echo "✅ backend/requirements.txt found"
fi

# Test 8: Create a small test package and try scp to confirm transfer works
echo "Testing file transfer to EC2..."
echo "test content" > test_file.txt
if ! scp test_file.txt astroshield:~/test_file.txt; then
  echo "❌ Error: Cannot transfer files to EC2"
  rm -f test_file.txt
  exit 1
else
  echo "✅ Successfully transferred test file to EC2"
  rm -f test_file.txt
  ssh astroshield "rm -f ~/test_file.txt"
fi

echo "=== All tests passed! ==="
echo "The system is ready for deployment. Run ./simplified_deploy.sh to start the deployment process." 