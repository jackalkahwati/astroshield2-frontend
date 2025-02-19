#!/bin/bash
# Create .env file from example
cp config/kafka_example.env .env

# Secure the .env file permissions
chmod 600 .env

echo 'Please edit .env with actual credentials and source it before use:'
echo 'source .env'
