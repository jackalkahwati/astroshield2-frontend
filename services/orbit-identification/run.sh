#!/bin/bash

echo "Starting Orbit Family Identification API..."

# Check if Python is installed
if ! command -v python3 &>/dev/null; then
    echo "Python 3 is not installed. Please install Python 3."
    exit 1
fi

# Check if the virtual environment exists, if not create it
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate the virtual environment
source venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Run the API
echo "Starting the API server..."
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Deactivate the virtual environment when the server stops
deactivate 