#!/bin/bash

# Exit on error
set -e

echo "========================================="
echo "AstroShield Docker Deployment Script"
echo "========================================="

# Build the backend and frontend images
echo "Building backend image..."
docker build -t astroshield-backend -f Dockerfile.backend .

echo "Building frontend image..."
cd frontend
docker build -t astroshield-frontend -f Dockerfile.frontend .
cd ..

# Start the application using Docker Compose
echo "Starting AstroShield using Docker Compose..."
docker-compose up -d

echo "========================================="
echo "AstroShield deployment completed!"
echo "========================================="
echo "Frontend is accessible at: http://localhost:3000"
echo "Backend API is accessible at: http://localhost:3001"
echo "=========================================" 