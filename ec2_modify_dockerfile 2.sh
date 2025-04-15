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
