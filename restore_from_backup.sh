#!/bin/bash

# Script to restore files from the backup directory

echo "Restoring files from backup directory..."

# Create directories if they don't exist
mkdir -p models
mkdir -p checkpoints/launch_evaluator

# Restore Kafka binaries if needed
if [ ! -d "kafka_2.13-3.9.0" ]; then
  echo "Restoring Kafka binaries..."
  cp -r backup/kafka_2.13-3.9.0 ./
  cp backup/kafka_2.13-3.9.0.tgz ./
fi

# Restore model files if needed
echo "Restoring model files..."
cp -r backup/models/* ./models/

echo "Restoration complete!"
echo "Note: Large files are now managed with Git LFS. Run 'git lfs pull' to download them." 