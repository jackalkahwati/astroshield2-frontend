#!/bin/bash

# Exit on any error
set -e

# Parse arguments
VERSION=$1
if [ -z "$VERSION" ]; then
    echo "Error: Version argument is required"
    echo "Usage: $0 <version>"
    exit 1
fi

# Load environment variables
source .env

echo "Starting rollback to version $VERSION..."

# Rollback database
echo "Rolling back database migrations..."
npm run migrate:down

# Rollback application version
echo "Rolling back application to version $VERSION..."
git checkout $VERSION

# Install dependencies for the rolled back version
echo "Installing dependencies..."
npm install

# Run database migrations for the rolled back version
echo "Running database migrations..."
npm run migrate:up

# Restart the application
echo "Restarting application..."
pm2 restart all

echo "Rollback completed successfully" 