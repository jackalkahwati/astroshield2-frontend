# AstroShield Deployment Guide

## Prerequisites

- Docker and Docker Compose
- Kubernetes cluster (for production)
- AWS account with EKS access
- Domain name and SSL certificates
- Environment variables configured

## Local Development Setup

1. Clone the repository:
```bash
git clone https://github.com/your-org/astroshield.git
cd astroshield
```

2. Create environment files:
```bash
cp .env.example .env
cp frontend/.env.example frontend/.env
```

3. Start the development environment:
```bash
docker-compose up -d
```

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Production Deployment

### Infrastructure Setup

1. Create EKS cluster:
```bash
eksctl create cluster -f cluster.yaml
```

2. Configure DNS and SSL:
```bash
kubectl apply -f cert-manager.yaml
kubectl apply -f ingress.yaml
```

3. Set up monitoring:
```bash
helm install prometheus prometheus-community/kube-prometheus-stack
helm install grafana grafana/grafana
```

### Application Deployment

1. Configure environment:
```bash
kubectl create secret generic astroshield-secrets --from-env-file=.env.prod
```

2. Deploy application:
```bash
kubectl apply -f k8s/
```

3. Verify deployment:
```bash
kubectl get pods
kubectl get services
```

### Database Setup

1. Create RDS instance:
```bash
terraform apply -target=module.database
```

2. Run migrations:
```bash
kubectl exec -it deployment/astroshield-backend -- alembic upgrade head
```

## Monitoring Setup

1. Access Grafana:
```bash
kubectl port-forward service/grafana 3000:80
```

2. Import dashboards:
- Application metrics dashboard
- Node metrics dashboard
- PostgreSQL metrics dashboard

3. Configure alerts in Grafana

## Backup and Recovery

1. Database backups:
```bash
# Automated daily backups
aws rds create-db-cluster-snapshot

# Manual backup
pg_dump -h $DB_HOST -U $DB_USER -d $DB_NAME > backup.sql
```

2. Application state:
- All state is stored in the database
- Container images are stored in ECR
- Configuration in Kubernetes secrets

## Security Considerations

1. Network Security:
- Use private subnets for backend services
- Enable AWS WAF
- Configure security groups

2. Application Security:
- Regular dependency updates
- Security scanning in CI/CD
- JWT token rotation

3. Monitoring:
- Enable AWS CloudTrail
- Configure CloudWatch alerts
- Set up log aggregation

## Troubleshooting

### Common Issues

1. Database Connection:
```bash
kubectl logs deployment/astroshield-backend
kubectl describe pod astroshield-backend
```

2. Frontend Issues:
```bash
kubectl logs deployment/astroshield-frontend
```

3. Scaling Issues:
```bash
kubectl describe hpa astroshield-backend
```

### Health Checks

1. Backend API:
```bash
curl https://api.astroshield.com/health
```

2. Database:
```bash
kubectl exec -it deployment/astroshield-backend -- python -c "from app.db import check_db; check_db()"
```

## Maintenance

### Updates

1. Backend updates:
```bash
kubectl set image deployment/astroshield-backend backend=your-registry/astroshield-backend:new-tag
```

2. Frontend updates:
```bash
kubectl set image deployment/astroshield-frontend frontend=your-registry/astroshield-frontend:new-tag
```

### Scaling

1. Horizontal scaling:
```bash
kubectl scale deployment astroshield-backend --replicas=3
```

2. Vertical scaling:
```bash
kubectl edit deployment astroshield-backend # Update resource limits
```

## Rollback Procedures

1. Application rollback:
```bash
kubectl rollout undo deployment/astroshield-backend
```

2. Database rollback:
```bash
alembic downgrade -1
``` 