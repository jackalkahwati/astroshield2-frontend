# AstroShield Deployment Guide

This document outlines the steps required to deploy AstroShield to a production environment.

## Prerequisites

Before you begin, you'll need the following:

1. Docker and Docker Compose installed on the target system
2. SSL certificates (for production deployment)
3. A domain name pointing to your server (for production deployment)
4. Access to a PostgreSQL database (can be deployed with Docker Compose)

## Deployment Steps

### 1. Clone the Repository

```bash
git clone https://github.com/your-organization/asttroshield.git
cd asttroshield
```

### 2. Configure Environment Variables

Create a production environment file:

```bash
cp .env.production.template .env.production
```

Edit the `.env.production` file to set all required environment variables:

- Replace placeholder passwords with strong, secure passwords
- Update UDL integration settings with real credentials
- Set domain name for CORS settings
- Configure environment-specific settings

### 3. SSL Certificates

Place your SSL certificates in the `nginx/ssl` directory:

```bash
mkdir -p nginx/ssl
# Copy your SSL certificate and private key
cp your-certificate.crt nginx/ssl/server.crt
cp your-private-key.key nginx/ssl/server.key
```

For testing purposes, you can generate self-signed certificates (the deployment script will do this automatically if no certificates are found).

### 4. Run the Deployment Script

Execute the deployment script:

```bash
./deploy.sh
```

The script will:
- Validate your environment configuration
- Check for SSL certificates
- (Optional) Run tests to ensure everything is working
- Build Docker images
- Start all services
- Verify the deployment

### 5. Verify Deployment

After the deployment script completes, verify the application is working correctly:

- Frontend: `https://your-domain/`
- Backend API: `https://your-domain/api/v1/`
- API Documentation: `https://your-domain/api/v1/docs`

## Manual Deployment

If you prefer to deploy manually or need to customize the deployment, follow these steps:

### 1. Configure Environment

Configure the `.env.production` file as described above.

### 2. Build Docker Images

```bash
docker-compose -f docker-compose.yml build
```

### 3. Start Services

```bash
docker-compose -f docker-compose.yml up -d
```

### 4. Verify Deployment

```bash
# Check container status
docker-compose ps

# Check backend health
curl http://localhost:3001/health

# Check frontend health
curl http://localhost:3000/api/health
```

## Production Considerations

### Database Backups

Set up regular database backups:

```bash
# Example cron job to backup PostgreSQL database daily
0 2 * * * docker exec postgres pg_dump -U postgres astroshield > /path/to/backups/astroshield_$(date +\%Y\%m\%d).sql
```

### Monitoring

Enable monitoring with Prometheus and Grafana:

1. Access Grafana: `http://your-domain:3002`
2. Default credentials: admin/admin (change on first login)
3. Configure dashboards for monitoring system health

### Security Best Practices

1. Regularly update SSL certificates
2. Keep Docker images updated
3. Regularly rotate secrets and passwords
4. Configure a firewall to restrict access to necessary ports
5. Set up log monitoring and automated security scanning

## Troubleshooting

### Common Issues

1. **Connection refused errors**: Check if the containers are running with `docker-compose ps`
2. **Database connection issues**: Verify database settings in `.env.production`
3. **SSL certificate problems**: Ensure certificates are correctly placed in `nginx/ssl/`

### Logs Access

View logs for troubleshooting:

```bash
# View logs for all services
docker-compose logs

# View logs for a specific service
docker-compose logs backend
docker-compose logs frontend
docker-compose logs nginx
```

## Updating the Application

To update the application:

```bash
# Pull latest changes
git pull

# Rebuild and restart
docker-compose -f docker-compose.yml build
docker-compose -f docker-compose.yml up -d
```

## Rollback Procedure

If you need to roll back to a previous version:

```bash
# Check out the previous version
git checkout <previous-commit-or-tag>

# Rebuild and restart
docker-compose -f docker-compose.yml build
docker-compose -f docker-compose.yml up -d
``` 