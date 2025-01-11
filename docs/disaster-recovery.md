# Disaster Recovery Procedures

## Overview

This document outlines the procedures for backup, restore, and disaster recovery for the AstroShield platform.

## Backup Schedule

- Database backups: Daily at 1 AM UTC
- Backup verification: Daily at 3 AM UTC
- Retention period: 30 days

## Backup Storage

- Primary storage: AWS S3 (astroshield-backups bucket)
- Region: us-west-2
- Versioning: Enabled
- Encryption: AWS KMS

## Recovery Time Objectives (RTO)

- Critical systems: 1 hour
- Non-critical systems: 4 hours

## Recovery Point Objectives (RPO)

- Database: 24 hours
- Application state: Real-time (stateless)

## Disaster Recovery Procedures

### 1. Database Recovery

```bash
# View available backups
aws s3 ls s3://astroshield-backups/database/

# Restore from latest backup
kubectl create job --from=cronjob/disaster-recovery dr-latest

# Restore from specific backup
kubectl create job --from=cronjob/disaster-recovery dr-specific -- specific-backup-20240101.dump
```

### 2. Application Recovery

```bash
# Redeploy application
kubectl apply -f k8s/

# Verify deployment
kubectl get pods
kubectl get services
```

### 3. DNS/SSL Recovery

```bash
# Verify DNS records
kubectl get ingress

# Reapply SSL certificates
kubectl apply -f k8s/cert-manager/
```

## Verification Procedures

### 1. Database Verification

```bash
# Check backup job status
kubectl get jobs

# View backup logs
kubectl logs job/backup-verification-<timestamp>
```

### 2. Application Health

```bash
# Check application health
curl https://api.astroshield.com/health

# Verify metrics
curl https://api.astroshield.com/metrics
```

### 3. System Verification

```bash
# Verify all components
kubectl get pods --all-namespaces
kubectl get services --all-namespaces
```

## Emergency Contacts

1. Database Team:
   - Primary: db-team@astroshield.com
   - Secondary: db-oncall@astroshield.com

2. Infrastructure Team:
   - Primary: infra-team@astroshield.com
   - Secondary: infra-oncall@astroshield.com

3. Security Team:
   - Primary: security@astroshield.com
   - Emergency: security-oncall@astroshield.com

## Recovery Runbook

### Complete System Recovery

1. Assess the Situation
   - Identify affected systems
   - Determine cause of failure
   - Notify stakeholders

2. Infrastructure Recovery
   ```bash
   # Verify AWS resources
   aws eks describe-cluster --name astroshield
   aws rds describe-db-instances
   ```

3. Database Recovery
   ```bash
   # Start recovery job
   kubectl create job --from=cronjob/disaster-recovery dr-latest
   
   # Monitor progress
   kubectl logs -f job/dr-latest
   ```

4. Application Recovery
   ```bash
   # Deploy latest version
   kubectl apply -f k8s/
   
   # Scale services
   kubectl scale deployment/astroshield-backend --replicas=3
   kubectl scale deployment/astroshield-frontend --replicas=3
   ```

5. Verification
   ```bash
   # Verify all services
   kubectl get pods
   kubectl get services
   kubectl get ingress
   
   # Check application health
   curl https://api.astroshield.com/health/details
   ```

### Partial Recovery

1. Single Service Recovery
   ```bash
   # Redeploy specific service
   kubectl rollout restart deployment/astroshield-backend
   
   # Verify service
   kubectl get pods -l app=astroshield,component=backend
   ```

2. Database Recovery
   ```bash
   # Restore specific backup
   kubectl create job --from=cronjob/disaster-recovery dr-specific -- backup-20240101.dump
   ```

## Post-Recovery Procedures

1. Root Cause Analysis
   - Document incident timeline
   - Identify cause
   - Implement preventive measures

2. System Verification
   - Run full test suite
   - Verify data integrity
   - Check monitoring systems

3. Documentation Update
   - Update recovery procedures
   - Document lessons learned
   - Update system architecture 