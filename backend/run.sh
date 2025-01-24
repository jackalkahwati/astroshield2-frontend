#!/bin/bash

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port 3001 --reload 