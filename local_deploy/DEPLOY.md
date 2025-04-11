# AstroShield Deployment Guide

This guide provides instructions for deploying AstroShield to different environments.

## Local Development Deployment

For local development and testing:

1. Navigate to the local_deploy directory:
   ```bash
   cd /Users/jackal-kahwati/asttroshield_v0\ 2/local_deploy
   ```

2. Start the local deployment:
   ```bash
   ./start.sh
   ```

3. Access the application:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:3001

4. Stop the local deployment:
   ```bash
   ./stop.sh
   ```

## Production Deployment to astroshield.sdataplab.com

For deploying to the production environment:

1. Navigate to the production deployment directory:
   ```bash
   cd /Users/jackal-kahwati/asttroshield_v0\ 2/local_deploy/2/deploy_production
   ```

2. Create the deployment package:
   ```bash
   chmod +x scripts/create-package.sh
   scripts/create-package.sh
   ```

3. Execute the deployment script:
   ```bash
   chmod +x scripts/deploy.sh
   scripts/deploy.sh
   ```
   - When prompted, enter the IP address of your production server.

4. Verify the deployment:
   - Frontend: https://astroshield.sdataplab.com
   - Backend API: https://astroshield.sdataplab.com/api/v1/health

## Production Server Requirements

- Ubuntu 20.04 LTS or Amazon Linux 2
- At least 2GB RAM and 2 vCPUs
- Port 80 and 443 open in security group/firewall
- Python 3.8+ and Node.js 16+ available or permissions to install them
- Domain astroshield.sdataplab.com pointing to the server's IP

## Manual Deployment Steps

If the automated deployment script fails, follow these manual steps:

1. Copy the deployment package to the server:
   ```bash
   scp astroshield-*.zip user@server-ip:/tmp/
   ```

2. SSH into the server:
   ```bash
   ssh user@server-ip
   ```

3. Unzip and set up the application:
   ```bash
   cd /tmp
   unzip astroshield-*.zip
   sudo mkdir -p /opt/astroshield
   sudo cp -r package/backend package/frontend /opt/astroshield/
   sudo cp -r package/nginx/conf.d/* /etc/nginx/conf.d/
   sudo cp package/scripts/* /opt/astroshield/
   sudo chmod +x /opt/astroshield/*.sh
   ```

4. Set up the server environment:
   ```bash
   cd /opt/astroshield
   sudo ./setup-server.sh
   ```

5. Configure SSL:
   ```bash
   sudo ./setup-ssl.sh astroshield.sdataplab.com
   ```

6. Start the services:
   ```bash
   sudo ./start-services.sh
   ```

## Troubleshooting

### Common Issues

1. **SSL Certificate Issues**:
   ```bash
   sudo certbot --nginx -d astroshield.sdataplab.com --force-renewal
   ```

2. **Backend Service Not Starting**:
   ```bash
   sudo systemctl status astroshield-backend
   sudo journalctl -u astroshield-backend
   ```

3. **Frontend Service Not Starting**:
   ```bash
   sudo systemctl status astroshield-frontend
   sudo journalctl -u astroshield-frontend
   ```

4. **Nginx Configuration Issues**:
   ```bash
   sudo nginx -t
   sudo systemctl restart nginx
   ```

## Maintenance

### Service Management

- Start all services: `sudo /opt/astroshield/start-services.sh`
- Stop all services: `sudo /opt/astroshield/stop-services.sh`
- Restart backend: `sudo systemctl restart astroshield-backend`
- Restart frontend: `sudo systemctl restart astroshield-frontend`
- Restart Nginx: `sudo systemctl restart nginx`

### Log Monitoring

- Backend logs: `sudo journalctl -u astroshield-backend -f`
- Frontend logs: `sudo journalctl -u astroshield-frontend -f`
- Nginx access logs: `sudo tail -f /var/log/nginx/astroshield.access.log`
- Nginx error logs: `sudo tail -f /var/log/nginx/astroshield.error.log`

### Certificate Renewal

SSL certificates from Let's Encrypt are automatically renewed. To manually renew:

```bash
sudo certbot renew
```