#!/bin/bash
# Validate required environment variables
required_vars=(KAFKA_BOOTSTRAP_SERVERS KAFKA_USERNAME KAFKA_PASSWORD)

for var in "${required_vars[@]}"; do
  if [[ -z "${!var}" ]]; then
    echo "ERROR: Missing required environment variable $var" >&2
    exit 1
  fi
done

echo "Environment validation successful"
