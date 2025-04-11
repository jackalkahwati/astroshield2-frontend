# AstroShield Deployment Guide

This guide provides instructions for deploying the AstroShield application to astroshield.sdataplab.com.

## Prerequisites

- Ubuntu 20.04 LTS or later
- Python 3.8 or later
- Node.js 16.x or later
- Nginx
- SSL certificate (will be automatically configured)

## Deployment Steps

1. **Extract the Deployment Package**
   ```bash
   tar -xzf astroshield-web-deploy.tar.gz
   cd astroshield_deployment_package
   ```

2. **Make the Deployment Script Executable**
   ```bash
   chmod +x deploy_astroshield.sh
   ```

3. **Run the Deployment Script**
   ```bash
   sudo ./deploy_astroshield.sh
   ```

4. **Verify the Installation**
   - Check the application at https://astroshield.sdataplab.com
   - Verify SSL certificate is working
   - Test the API endpoints

## Directory Structure

```
/var/www/astroshield/
├── venv/                 # Python virtual environment
├── frontend/            # Next.js frontend application
├── backend/             # FastAPI backend application
├── nginx/               # Nginx configuration
├── systemd/             # Systemd service files
└── scripts/             # Monitoring and maintenance scripts
```

## Monitoring

The application includes a monitoring script that checks:
- Service status
- Disk space usage
- Log file sizes

To start monitoring:
```bash
sudo systemctl start astroshield-monitor
```

## Maintenance

### Logs
- Application logs: `/var/log/astroshield/`
- Nginx logs: `/var/log/nginx/astroshield.*.log`

### Backup
Backups are stored in `/var/backups/astroshield/`

### Restarting the Application
```bash
sudo systemctl restart astroshield
```

## Troubleshooting

1. **Service Not Starting**
   - Check logs: `sudo journalctl -u astroshield`
   - Verify permissions: `sudo chown -R www-data:www-data /var/www/astroshield`

2. **SSL Issues**
   - Check certificate: `sudo certbot certificates`
   - Renew certificate: `sudo certbot renew`

3. **Nginx Issues**
   - Test configuration: `sudo nginx -t`
   - Check error logs: `sudo tail -f /var/log/nginx/error.log`

## Support

For support, contact:
- Email: admin@sdataplab.com
- Issue Tracker: https://github.com/sdataplab/astroshield/issues 