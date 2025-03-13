#!/bin/bash

# Run the Track Classification API
echo "Starting Track Classification API..."
uvicorn main:app --host 0.0.0.0 --port 8000 --reload 