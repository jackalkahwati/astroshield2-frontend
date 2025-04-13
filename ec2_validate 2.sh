#!/bin/bash
set -e

# EC2 Environment Validation Script for AstroShield
echo "==== EC2 Environment Validation ===="
echo "Running diagnostics on $(hostname) @ $(date)"
echo

# Check operating system
echo "OS Information:"
cat /etc/os-release
echo

# Check system resources
echo "System Resources:"
echo "- CPU: $(nproc) cores"
echo "- Memory: $(free -h | grep Mem | awk '{print $2}')"
echo "- Disk Space: $(df -h / | awk 'NR==2 {print $4}') available"
echo

# Check Docker and Docker Compose
echo "Docker Environment:"
if command -v docker &> /dev/null; then
    echo "✅ Docker installed: $(docker --version)"
else
    echo "❌ Docker not installed"
fi

if command -v docker-compose &> /dev/null; then
    echo "✅ Docker Compose installed: $(docker-compose --version)"
else
    echo "❌ Docker Compose not installed"
fi
echo

# Check ports in use
echo "Network Ports:"
echo "Checking required ports (80, 443, 3000, 3001, 5432)..."
for port in 80 443 3000 3001 5432; do
    if netstat -tuln | grep ":$port " > /dev/null; then
        echo "❌ Port $port is already in use"
    else
        echo "✅ Port $port is available"
    fi
done
echo

# Check if PostgreSQL client is available
echo "Database Tools:"
if command -v psql &> /dev/null; then
    echo "✅ PostgreSQL client installed: $(psql --version)"
else
    echo "ℹ️ PostgreSQL client not installed (optional, will use Docker container)"
fi
echo

# Check if Nginx is already running
echo "Web Server:"
if systemctl is-active --quiet nginx; then
    echo "ℹ️ Nginx is running as a system service (might conflict with Docker Nginx)"
else
    echo "✅ No system Nginx running"
fi
echo

# Check if we have permissions to write to deployment folder
echo "Permissions:"
DEPLOYMENT_FOLDER="/home/ec2-user/astroshield"
mkdir -p $DEPLOYMENT_FOLDER 2>/dev/null || true
if [ -w "$DEPLOYMENT_FOLDER" ]; then
    echo "✅ Can write to deployment folder: $DEPLOYMENT_FOLDER"
else
    echo "❌ Cannot write to deployment folder: $DEPLOYMENT_FOLDER"
fi
echo

# Check current docker containers
echo "Docker Status:"
if command -v docker &> /dev/null; then
    echo "Current containers:"
    docker ps -a
    echo
    echo "Current images:"
    docker images
else
    echo "Docker not available, skipping container check"
fi
echo

echo "==== Validation Complete ====" 