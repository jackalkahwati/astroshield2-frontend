#!/bin/bash
set -e

echo "=== Modifying Backend Dockerfile ==="
echo "This script will modify the Dockerfile CMD to use app.main:app"

# Create the script to run on the EC2 instance
cat > ec2_modify_dockerfile.sh << 'EOF'
#!/bin/bash
set -e

echo "=== Modifying backend Dockerfile ==="

cd /home/stardrive/astroshield/deployment/backend

# Backup original Dockerfile if not already done
if [ ! -f Dockerfile.bak ]; then
  cp Dockerfile Dockerfile.bak
fi

# Modify the Dockerfile to use app.main instead of main
cat > Dockerfile << 'EOT'
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 3001

# Start the application using app.main instead of main
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3001"]
EOT

# Rebuild the backend container
echo "Rebuilding the backend container with the modified Dockerfile..."
cd /home/stardrive/astroshield/deployment
sudo docker-compose down
sudo docker-compose build backend
sudo docker-compose up -d

# Check the logs after restart
echo "Checking backend logs after restart:"
sleep 5  # Wait for service to restart
sudo docker logs deployment-backend-1

echo "=== Dockerfile modification completed ==="
EOF

# Transfer the script to EC2
echo "Transferring script to EC2..."
chmod +x ec2_modify_dockerfile.sh
scp ec2_modify_dockerfile.sh astroshield:~/

# Run the script on EC2
echo "Running script on EC2..."
ssh astroshield "chmod +x ~/ec2_modify_dockerfile.sh && ~/ec2_modify_dockerfile.sh"

echo "Backend Dockerfile has been modified."
echo "Try accessing https://astroshield.sdataplab.com/ again." 