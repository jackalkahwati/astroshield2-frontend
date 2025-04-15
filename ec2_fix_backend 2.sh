#!/bin/bash
set -e

echo "=== Fixing backend Dockerfile ==="

cd /home/stardrive/astroshield/deployment/backend

# Backup original Dockerfile
cp Dockerfile Dockerfile.bak

# Check if there's a main.py file in app directory
if ls app/main.py 2>/dev/null; then
  echo "Found app/main.py - creating symlink"
  ln -sf app/main.py main.py
elif ls app/app.py 2>/dev/null; then
  echo "Found app/app.py - creating symlink"
  ln -sf app/app.py main.py
fi

# Let's try to find the correct entry point
echo "Looking for possible entry points in app directory..."
find app -name "*.py" | sort

# Based on FastAPI conventions, let's check for typical entry files
for file in app/__init__.py app/main.py app/app.py; do
  if [ -f "$file" ]; then
    echo "Found potential entry point: $file"
    cat "$file" | grep -A 5 "app ="
  fi
done

# Create a simple entry point file
echo "Creating a simple entry point file that imports from app package"
cat > main.py << 'EOT'
# Entry point for the FastAPI application
# Import the FastAPI app from the app package
from app.main import app

# This file is needed because Dockerfile CMD specifies main:app
# It simply re-exports the app object from the app package

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=3001, reload=True)
EOT

# Restart the backend container
echo "Restarting backend container..."
cd /home/stardrive/astroshield/deployment
sudo docker-compose restart backend

# Check the logs after restart
echo "Checking backend logs after restart:"
sleep 5  # Wait for service to restart
sudo docker logs deployment-backend-1

echo "=== Backend fix applied ==="
