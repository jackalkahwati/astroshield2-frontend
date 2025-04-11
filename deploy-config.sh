#!/bin/bash

# EC2 configuration
SSH_KEY_PATH="~/.ssh/id_rsa_astroshield_new"  # Using the astroshield_new key found in the SSH directory
EC2_USER="stardrive"
EC2_HOST="astroshield"  # Using the SSH alias from the config file instead of the domain
APP_NAME="astroshield"

# Application configuration
BACKEND_PORT=3001
FRONTEND_PORT=3000

# Directories
BACKEND_DIR="."
FRONTEND_DIR="./frontend"
REMOTE_DIR="/home/${EC2_USER}/${APP_NAME}"

# Dependencies and versions
NODE_VERSION="16"
PYTHON_VERSION="3"