#!/bin/bash

# Create test database
psql -U postgres -c "DROP DATABASE IF EXISTS astroshield_test;"
psql -U postgres -c "CREATE DATABASE astroshield_test;"

# Run migrations
NODE_ENV=test npx knex migrate:latest

echo "Test database setup complete" 