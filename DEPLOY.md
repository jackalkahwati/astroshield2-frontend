# AstroShield Deployment Guide

This guide explains how to deploy the AstroShield application to an EC2 server.

## Prerequisites

- SSH access to an EC2 instance
- Python 3.x installed on your local machine
- Node.js and npm installed on your local machine

## Deployment Steps

1. **Configure the deployment settings**

   Edit the `deploy-config.sh` file to match your EC2 instance details:

   ```bash
   # EC2 configuration
   SSH_KEY="astroshield_key"
   EC2_USER="ec2-user"         # EC2 username (e.g., ec2-user, ubuntu)
   EC2_HOST="your-ec2-hostname-or-ip"  # Your EC2 hostname or IP address
   APP_NAME="astroshield"      # Application name for the remote directory

   # Application configuration
   BACKEND_PORT=3001           # Port for the backend API
   FRONTEND_PORT=3000          # Port for the Next.js frontend

   # Directories
   BACKEND_DIR="."             # Local backend directory
   FRONTEND_DIR="./frontend"   # Local frontend directory
   ```

2. **Run the deployment script**

   ```bash
   ./deploy.sh
   ```

   This script will:
   - Package both the backend and frontend code
   - Upload the files to your EC2 instance
   - Install necessary dependencies
   - Set up systemd services for automatic startup
   - Start the application

3. **Access the application**

   After successful deployment, you can access:
   - Frontend: `http://your-ec2-hostname-or-ip:3000`
   - Backend API: `http://your-ec2-hostname-or-ip:3001`

## Managing the Deployed Application

The deployment creates several scripts on the server to help manage the application:

- **Start the application**: 
  ```bash
  cd /home/ec2-user/astroshield && ./start.sh
  ```

- **Stop the application**: 
  ```bash
  cd /home/ec2-user/astroshield && ./stop.sh
  ```

- **Restart the application (using systemd)**: 
  ```bash
  sudo systemctl restart astroshield
  ```

- **View application logs**:
  ```bash
  # Backend logs
  cat /home/ec2-user/astroshield/backend/backend.log
  
  # Frontend logs
  cat /home/ec2-user/astroshield/frontend/frontend.log
  ```

## Troubleshooting

1. **Check service status**:
   ```bash
   sudo systemctl status astroshield
   ```

2. **Check if ports are in use**:
   ```bash
   sudo lsof -i :3000,3001
   ```

3. **Check firewall settings**:
   Make sure the EC2 security group allows incoming traffic on ports 3000 and 3001.

4. **Check application logs**:
   See the log files in the application directory for more detailed errors.

## Security Considerations

- The deployment script temporarily stores the SSH key on the local machine. It is removed after deployment.
- Consider adding HTTPS/SSL support for production environments.
- Review EC2 security groups to restrict access to only necessary ports. 