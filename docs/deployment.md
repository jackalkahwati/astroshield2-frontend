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