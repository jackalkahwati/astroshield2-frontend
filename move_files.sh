#!/bin/bash

# Move all files from backend directory to root
mv backend/* .

# Remove empty backend directory
rmdir backend

# Update imports in main.py to remove 'app.' prefix
sed -i '' 's/from app\./from /g' main.py

# Update imports in other Python files
find . -type f -name "*.py" -exec sed -i '' 's/from app\./from /g' {} + 