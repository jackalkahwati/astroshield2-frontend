# AstroShield Deployment Package

This package contains everything needed to deploy AstroShield to a server.

## Deployment Options

1. **Docker Compose Deployment**: Full deployment with Docker and multiple containers
2. **Minimal Server Deployment**: Simple Python FastAPI server for demo purposes

## Quick Start

1. Upload this package to your server
2. Extract the package: `tar -xzf astroshield_deploy.tar.gz`
3. Run the deployment script: `cd astroshield_deploy && bash scripts/deploy.sh`
4. Follow the prompts to choose your deployment method

## Server Requirements

- Python 3.6+ (for minimal deployment)
- Docker and Docker Compose (for container deployment)
- 1GB+ RAM
- 10GB+ disk space

## Accessing the Application

After deployment:
- API will be available at http://localhost:3001
- Configure Nginx or other web server to expose the service

## Configuration

To customize the deployment, edit the docker-compose.yml file or modify the minimal_server.py file.
