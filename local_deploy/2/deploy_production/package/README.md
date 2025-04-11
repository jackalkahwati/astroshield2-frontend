# AstroShield Production Deployment

This directory contains all necessary files and configurations for deploying AstroShield to the production environment at https://astroshield.sdataplab.com.

## Overview

This deployment package includes:

- A FastAPI-based backend service
- A Next.js frontend application
- Nginx configuration for serving both applications
- SSL/TLS configuration via Let's Encrypt
- Systemd service definitions for process management

## Directory Structure

```
deploy_production/
├── backend/              # Backend Python application
│   ├── minimal_server.py # The FastAPI server
│   └── requirements.txt  # Python dependencies
├── frontend/             # Frontend Next.js application
│   ├── lib/              # Shared libraries
│   ├── next.config.js    # Next.js configuration
│   └── package.json      # Node.js dependencies
├── nginx/                # Nginx configuration
│   └── conf.d/           # Site configuration
│       └── astroshield.conf
└── scripts/              # Deployment scripts
    ├── deploy.sh         # Main deployment script
    ├── setup-server.sh   # Server setup script
    ├── setup-ssl.sh      # SSL configuration script
    ├── start-services.sh # Service start script
    └── stop-services.sh  # Service stop script
```

## Deployment Instructions

1. **Prerequisites**:
   - An EC2 instance or other Linux server with SSH access
   - Domain astroshield.sdataplab.com pointing to your server's IP
   - SSH key for authentication

2. **Deployment**:
   
   Run the deployment script:
   
   ```bash
   cd scripts
   ./deploy.sh
   ```
   
   The script will prompt for the server IP if not specified.

3. **Manual Setup (if needed)**:

   If you need to perform manual setup:
   
   a. Copy all files to the server:
   ```bash
   scp -r deploy_production/* user@server:/tmp/
   ```
   
   b. On the server:
   ```bash
   sudo mkdir -p /opt/astroshield
   sudo cp -r /tmp/backend /tmp/frontend /opt/astroshield/
   sudo cp -r /tmp/nginx/conf.d/* /etc/nginx/conf.d/
   sudo cp /tmp/scripts/* /opt/astroshield/
   sudo chmod +x /opt/astroshield/*.sh
   cd /opt/astroshield
   sudo ./setup-server.sh
   sudo ./setup-ssl.sh astroshield.sdataplab.com
   sudo ./start-services.sh
   ```

## Post-Deployment

After deployment, verify the services are running:

- Frontend: https://astroshield.sdataplab.com
- Backend API: https://astroshield.sdataplab.com/api/v1/health

## Troubleshooting

If you encounter issues, check the logs:

- Backend logs: `journalctl -u astroshield-backend`
- Frontend logs: `journalctl -u astroshield-frontend`
- Nginx logs: 
  - `/var/log/nginx/astroshield.access.log`
  - `/var/log/nginx/astroshield.error.log`

## Maintenance

- To stop services: `sudo /opt/astroshield/stop-services.sh`
- To start services: `sudo /opt/astroshield/start-services.sh`
- To update SSL certificate: `sudo certbot renew`