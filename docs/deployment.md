# Deployment Documentation

## Overview

This document outlines the deployment procedures and configurations for the AstroShield platform.

## Infrastructure Setup

### AWS Infrastructure

1. EKS Cluster
   ```bash
   # Create EKS cluster
   eksctl create cluster \
     --name astroshield \
     --region us-west-2 \
     --nodegroup-name standard-workers \
     --node-type t3.medium \
     --nodes 3 \
     --nodes-min 1 \
     --nodes-max 4 \
     --managed
   ```

2. RDS Database
   ```bash
   # Create RDS instance
   aws rds create-db-instance \
     --db-instance-identifier astroshield-db \
     --db-instance-class db.t3.medium \
     --engine postgres \
     --master-username admin \
     --master-user-password <password> \
     --allocated-storage 20
   ```

3. ElastiCache
   ```bash
   # Create Redis cluster
   aws elasticache create-cache-cluster \
     --cache-cluster-id astroshield-cache \
     --engine redis \
     --cache-node-type cache.t3.micro \
     --num-cache-nodes 1
   ```

### Network Configuration

1. VPC Setup
   ```bash
   # Create VPC
   aws ec2 create-vpc \
     --cidr-block 10.0.0.0/16 \
     --tag-specifications 'ResourceType=vpc,Tags=[{Key=Name,Value=astroshield-vpc}]'
   ```

2. Subnets
   ```bash
   # Create public subnet
   aws ec2 create-subnet \
     --vpc-id <vpc-id> \
     --cidr-block 10.0.1.0/24 \
     --availability-zone us-west-2a
   
   # Create private subnet
   aws ec2 create-subnet \
     --vpc-id <vpc-id> \
     --cidr-block 10.0.2.0/24 \
     --availability-zone us-west-2b
   ```

## Application Deployment

### Backend Deployment

1. Build Image
   ```bash
   # Build backend image
   docker build -t astroshield-backend:latest -f backend/Dockerfile .
   
   # Push to registry
   docker tag astroshield-backend:latest <registry>/astroshield-backend:latest
   docker push <registry>/astroshield-backend:latest
   ```

2. Deploy Backend
   ```bash
   # Apply backend manifests
   kubectl apply -f k8s/backend/
   
   # Verify deployment
   kubectl get pods -l app=astroshield,component=backend
   ```

### Frontend Deployment

1. Build Image
   ```bash
   # Build frontend image
   docker build -t astroshield-frontend:latest -f frontend/Dockerfile .
   
   # Push to registry
   docker tag astroshield-frontend:latest <registry>/astroshield-frontend:latest
   docker push <registry>/astroshield-frontend:latest
   ```

2. Deploy Frontend
   ```bash
   # Apply frontend manifests
   kubectl apply -f k8s/frontend/
   
   # Verify deployment
   kubectl get pods -l app=astroshield,component=frontend
   ```

### Database Migration

1. Initial Setup
   ```bash
   # Apply migrations
   kubectl create job --from=cronjob/migrate db-migrate
   
   # Verify migration status
   kubectl logs job/db-migrate
   ```

2. Seed Data
   ```bash
   # Apply seed data
   kubectl create job --from=cronjob/seed db-seed
   
   # Verify seed status
   kubectl logs job/db-seed
   ```

## Monitoring Setup

### Prometheus & Grafana

1. Install Prometheus
   ```bash
   # Add Prometheus repo
   helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
   
   # Install Prometheus
   helm install prometheus prometheus-community/kube-prometheus-stack \
     -f k8s/config/prometheus-values.yaml
   ```

2. Configure Grafana
   ```bash
   # Apply dashboard configs
   kubectl apply -f k8s/config/grafana-dashboards/
   
   # Get Grafana password
   kubectl get secret prometheus-grafana -o jsonpath="{.data.admin-password}" | base64 --decode
   ```

### Logging Setup

1. Install EFK Stack
   ```bash
   # Add Elastic repo
   helm repo add elastic https://helm.elastic.co
   
   # Install Elasticsearch
   helm install elasticsearch elastic/elasticsearch
   
   # Install Fluentd
   kubectl apply -f k8s/logging/fluentd/
   
   # Install Kibana
   helm install kibana elastic/kibana
   ```

2. Configure Log Shipping
   ```bash
   # Apply logging config
   kubectl apply -f k8s/logging/config/
   
   # Verify log shipping
   kubectl logs -l app=fluentd
   ```

## SSL/TLS Configuration

### Certificate Management

1. Install cert-manager
   ```bash
   # Install cert-manager
   kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.8.0/cert-manager.yaml
   
   # Verify installation
   kubectl get pods -n cert-manager
   ```

2. Configure Certificates
   ```bash
   # Apply certificate configs
   kubectl apply -f k8s/cert-manager/
   
   # Verify certificates
   kubectl get certificates
   ```

## Deployment Verification

### Health Checks

1. Backend Health
   ```bash
   # Check backend health
   curl https://api.astroshield.com/health
   
   # Check backend metrics
   curl https://api.astroshield.com/metrics
   ```

2. Frontend Health
   ```bash
   # Check frontend health
   curl -I https://astroshield.com
   
   # Check frontend assets
   curl -I https://astroshield.com/static/
   ```

### Performance Tests

1. Load Testing
   ```bash
   # Run k6 load test
   k6 run k8s/tests/load-test.js
   
   # View results
   k6 report
   ```

2. Stress Testing
   ```bash
   # Run stress test
   artillery run k8s/tests/stress-test.yml
   
   # View results
   artillery report
   ```

## Rollback Procedures

### Application Rollback

1. Backend Rollback
   ```bash
   # Rollback deployment
   kubectl rollout undo deployment/astroshield-backend
   
   # Verify rollback
   kubectl rollout status deployment/astroshield-backend
   ```

2. Frontend Rollback
   ```bash
   # Rollback deployment
   kubectl rollout undo deployment/astroshield-frontend
   
   # Verify rollback
   kubectl rollout status deployment/astroshield-frontend
   ```

### Database Rollback

1. Schema Rollback
   ```bash
   # Rollback migration
   kubectl create job --from=cronjob/migrate db-rollback
   
   # Verify rollback
   kubectl logs job/db-rollback
   ```

2. Data Rollback
   ```bash
   # Restore from backup
   kubectl create job --from=cronjob/restore db-restore
   
   # Verify restore
   kubectl logs job/db-restore
   ```

## Maintenance Procedures

### Regular Updates

1. Backend Updates
   ```bash
   # Update backend
   kubectl set image deployment/astroshield-backend \
     backend=<registry>/astroshield-backend:new-tag
   
   # Monitor update
   kubectl rollout status deployment/astroshield-backend
   ```

2. Frontend Updates
   ```bash
   # Update frontend
   kubectl set image deployment/astroshield-frontend \
     frontend=<registry>/astroshield-frontend:new-tag
   
   # Monitor update
   kubectl rollout status deployment/astroshield-frontend
   ```

### Scaling

1. Horizontal Scaling
   ```bash
   # Scale backend
   kubectl scale deployment/astroshield-backend --replicas=5
   
   # Scale frontend
   kubectl scale deployment/astroshield-frontend --replicas=3
   ```

2. Vertical Scaling
   ```bash
   # Update resource requests/limits
   kubectl patch deployment astroshield-backend -p \
     '{"spec":{"template":{"spec":{"containers":[{"name":"backend","resources":{"requests":{"cpu":"2","memory":"4Gi"}}}]}}}}'
   ```

# CCDM Service Deployment Guide

This document provides instructions for deploying the CCDM (Conjunction and Collision Data Management) service in different environments.

## Deployment Prerequisites

Before deploying the CCDM service, ensure you have the following:

- Docker and Docker Compose (for containerized deployment)
- Kubernetes cluster (for production deployment)
- PostgreSQL database (version 13+)
- Redis (for caching and distributed locking)
- Node.js 18+ (for local development)
- Access to external data sources (Space-Track, JSPOC)

## Deployment Options

The CCDM service can be deployed in several ways:

1. **Local Development**: Running directly on a developer machine
2. **Docker Containers**: Single-host containerized deployment
3. **Kubernetes**: Production-grade scalable deployment
4. **Serverless**: For specific components (data processing pipelines)

## Local Development Deployment

For local development and testing:

```bash
# Clone the repository
git clone https://github.com/astroshield/ccdm-service.git
cd ccdm-service

# Install dependencies
npm install

# Setup local environment
cp .env.example .env
# Edit .env file with your configuration

# Start development services
docker-compose -f docker-compose.dev.yml up -d

# Run database migrations
npm run migrate

# Start the service in development mode
npm run dev
```

The service will be available at http://localhost:8080.

## Docker Deployment

For containerized deployment:

```bash
# Build the Docker image
docker build -t astroshield/ccdm-service:latest .

# Run the container with environment variables
docker run -d \
  --name ccdm-service \
  -p 8080:8080 \
  -e DEPLOYMENT_ENV=production \
  -e DATABASE_URL=postgresql://user:password@db-host:5432/ccdm \
  -e REDIS_URL=redis://redis-host:6379 \
  -e JWT_SECRET=your-secure-jwt-secret \
  astroshield/ccdm-service:latest
```

For multi-container deployment with Docker Compose:

```bash
# Start the entire stack
docker-compose -f docker-compose.yml up -d
```

## Kubernetes Deployment

For production deployment on Kubernetes:

1. Ensure you have `kubectl` configured to access your cluster
2. Apply the Kubernetes manifests:

```bash
# Apply ConfigMaps and Secrets
kubectl apply -f k8s/configmaps.yaml
kubectl apply -f k8s/secrets.yaml

# Apply Database resources (if not using external DB)
kubectl apply -f k8s/database/

# Apply Redis resources (if not using external Redis)
kubectl apply -f k8s/redis/

# Apply CCDM service resources
kubectl apply -f k8s/ccdm-service/
```

The typical Kubernetes deployment includes:

- Deployment for the CCDM service
- Services for internal and external access
- ConfigMaps for configuration
- Secrets for sensitive data
- HorizontalPodAutoscaler for scaling
- Ingress for external access
- PersistentVolumeClaims for persistent storage

## Configuration

Review the [Configuration Guide](./configuration.md) for detailed configuration options.

For deployment-specific configuration:

```bash
# Set required environment variables
export DEPLOYMENT_ENV=production
export DATABASE_URL=postgresql://user:password@db-host:5432/ccdm
export JWT_SECRET=your-secure-jwt-secret
export LOG_LEVEL=info
```

## Database Setup

### Initial Setup

```bash
# Create the database
createdb ccdm

# Run migrations
npm run migrate
```

### Database Migrations

```bash
# Run pending migrations
npm run migrate

# Roll back a migration
npm run migrate:down

# Create a new migration
npm run migrate:create -- --name add_new_table
```

## Scaling Considerations

The CCDM service is designed to scale horizontally. Consider the following:

- Database connection pooling is configured in the database section of the config
- Redis is used for distributed caching and locking
- Stateless service design allows multiple instances
- Rate limiting is applied per instance by default

For Kubernetes, configure the HorizontalPodAutoscaler:

```yaml
apiVersion: autoscaling/v2beta2
kind: HorizontalPodAutoscaler
metadata:
  name: ccdm-service
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ccdm-service
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Monitoring and Logging

### Logging

The service logs to stdout/stderr in JSON format by default. In production, use a log aggregation solution:

- ELK Stack (Elasticsearch, Logstash, Kibana)
- Prometheus and Grafana for metrics
- Loki for log aggregation
- CloudWatch Logs (if on AWS)

### Metrics

The service exposes metrics at the `/metrics` endpoint in Prometheus format. Configure Prometheus to scrape this endpoint.

### Health Checks

The service provides health check endpoints:

- `/health/liveness` - Liveness probe
- `/health/readiness` - Readiness probe

Configure these in your container orchestration:

```yaml
# Kubernetes example
livenessProbe:
  httpGet:
    path: /health/liveness
    port: 8080
  initialDelaySeconds: 30
  periodSeconds: 10
readinessProbe:
  httpGet:
    path: /health/readiness
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 5
```

## Backup and Disaster Recovery

### Database Backups

Schedule regular database backups:

```bash
# Create a PostgreSQL backup
pg_dump -U username -h hostname -d ccdm > ccdm_backup_$(date +%Y%m%d).sql

# Automated daily backups (add to crontab)
0 2 * * * pg_dump -U username -h hostname -d ccdm | gzip > /backups/ccdm_$(date +\%Y\%m\%d).sql.gz
```

### Disaster Recovery

1. Maintain database backups in multiple locations
2. Document recovery procedures
3. Periodically test restoration process
4. Use database replication for high availability
5. Consider multi-region deployment for critical services

## Security Considerations

1. **Network Security**:
   - Use TLS for all communication
   - Implement proper network policies in Kubernetes
   - Use VPC/subnet isolation for cloud deployments

2. **Authentication and Authorization**:
   - JWT tokens with appropriate expiration
   - Role-based access control
   - API keys for service-to-service communication

3. **Secrets Management**:
   - Use Kubernetes Secrets or a dedicated secrets manager
   - Avoid hardcoding secrets in configuration files
   - Rotate secrets regularly

4. **Vulnerability Scanning**:
   - Regularly scan container images
   - Update dependencies to address vulnerabilities
   - Implement security scanning in CI/CD pipeline

## Continuous Deployment

Set up a CI/CD pipeline for automated deployment:

1. Automated testing (unit, integration, e2e)
2. Image building and vulnerability scanning
3. Deployment to staging environment
4. Automated acceptance tests
5. Promotion to production

Example GitHub Actions workflow:

```yaml
name: Deploy CCDM Service

on:
  push:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Node.js
      uses: actions/setup-node@v2
      with:
        node-version: '18'
    
    - name: Install dependencies
      run: npm ci
    
    - name: Run tests
      run: npm test
    
    - name: Build Docker image
      run: |
        docker build -t astroshield/ccdm-service:${{ github.sha }} .
        docker tag astroshield/ccdm-service:${{ github.sha }} astroshield/ccdm-service:latest
    
    - name: Push Docker image
      run: |
        echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
        docker push astroshield/ccdm-service:${{ github.sha }}
        docker push astroshield/ccdm-service:latest
    
    - name: Deploy to Kubernetes
      uses: steebchen/kubectl@v2
      env:
        KUBE_CONFIG_DATA: ${{ secrets.KUBE_CONFIG_DATA }}
      with:
        args: set image deployment/ccdm-service ccdm-service=astroshield/ccdm-service:${{ github.sha }} --record
```

## Maintenance

### Version Updates

When updating the service version:

1. Review the changelog for breaking changes
2. Test in a staging environment first
3. Plan for database migrations
4. Consider a blue/green deployment strategy
5. Monitor closely after deployment

### Database Maintenance

Regular database maintenance tasks:

1. Index optimization
2. Vacuum operations for PostgreSQL
3. Performance monitoring
4. Scaling database resources as needed

## Troubleshooting

Common issues and solutions:

1. **Service fails to start**:
   - Check environment variables and configuration
   - Verify database connectivity
   - Check logs for startup errors

2. **Database connection issues**:
   - Verify connection string and credentials
   - Check database server status
   - Examine connection pool configuration

3. **High memory usage**:
   - Review and adjust Node.js memory limits
   - Check for memory leaks using profiling tools
   - Consider scaling horizontally instead of vertically

4. **Slow response times**:
   - Check database query performance
   - Review Redis caching configuration
   - Monitor external API call performance
   - Enable request tracing to identify bottlenecks

## Support and Contact

For deployment issues:

- File an issue on the GitHub repository
- Contact the AstroShield development team at devops@astroshield.space
- Consult the #ccdm-service channel on the internal Slack 